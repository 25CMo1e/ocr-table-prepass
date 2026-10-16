# ocr-table-prepass

**表格 OCR 之前的修正层。** 在把 PDF 交给 PP-StructureV3（或任何 OCR 流水线）**之前**，
先做三件事：把横躺的页转正、把跨页残条擦掉、给上游的多表格单元格错配打个补丁。

三层互相独立，可以只用其中一层。核心层零硬依赖：`numpy` / `PyMuPDF` / `paddleocr`
任何一个缺失，都只是"这一层不生效"，主流程照常跑。

```python
from ocr_table_prepass import prepare_pdf

report = prepare_pdf("scan.pdf", dpi=200)   # 转正 + 擦残条，得到一份修好的 PDF
pipeline.predict(input=report.path)         # 把它交给你的 OCR
report.cleanup()
```

> This is the preprocessing layer we extracted from an internal document-processing
> pipeline. It fixes three table-OCR failures we hit in production: (1) rotated scan
> pages make PP-StructureV3's table headers land in the wrong columns, because the
> orientation correction inside the pipeline never reaches the table-structure step;
> (2) a thin "cross-page fragment row" at the top of a page makes the row boundaries
> cut through rows, so two companies' data interleave; (3) a batch-order bug upstream
> misassigns table cells when a page has multiple tables. Each fix is independent and
> degrades to a no-op if its optional dependency is missing. Full write-up (in Chinese)
> below; the plug-and-play entry points are [§5](#5-即插即用).

---

## 0. 它解决什么问题

三个故障都来自同一批扫描件（并购 / 尽调材料），都是"识别模型本身没错、但送进去的东西
不对"。

| # | 故障 | 你能看到的现象 | 根因 | 修法 |
|---|---|---|---|---|
| 1 | 扫描页横躺 | 跨页表格的**表头整列错位**（9 列表格里「序号」跑到第 7 列）；正文有时还逐字读对，所以特别难怀疑到方向上去 | PP-StructureV3 内部的整页方向分类**是在工作**的，但它纠正后的坐标系**没有传给表格结构那一步**——单元格顺序仍按原始朝向算 | [§2](#2-方法一流水线之前把页面转正)：在流水线**之前**自己转正 |
| 2 | 跨页残条 | 同一张表里**两家公司的内容互相串行**、公司名被拼在一起；只差一两行，肉眼看像"模型瞎了" | 表格跨页时，本页顶端只剩上一页切走的行尾（很薄、几乎无字），这种几何会让行界切到行**内部** | [§3](#3-方法二跨页残条检测)：擦掉残条带上方的表格边框线 |
| 3 | 一页多表 | **单表 PDF 全对、整份报告错**；单元格跑到别的表 / 别的列 | 多张表格的裁剪被放进同一批次做检测，NMS 打乱顺序后，上游按「扁平顺序 + box_nums 切片」分组 | [§4](#4-方法三多表格单元格错配补丁)：运行时补丁（上游 PR 未合并） |

这三个都已经在真机上复现、定位、修复并回归过。实测数据和"哪些验证过、哪些没有"写在
[§7](#7-验证到什么程度)。

---

## 1. 模型：方向判定用的是什么

**模型**：PaddleOCR 3.x 的 `DocImgOrientationClassification`，即 PP-LCNet 文档图像方向
分类（四分类：0° / 90° / 180° / 270°）。这个模型就是 PP-StructureV3 内部用的那个。

我们和流水线的区别只有一点：**单独调用它，并且在流水线之外完成转正**。

为什么必须绕开内置的那一环：

* 内置方向模块确实在纠正方向——实测一张 180° 倒置的正文页，41 行里只有 3 个标点不同，
  其余逐字读对；没有方向纠正是做不到的。
* 但它的纠正结果**只影响文字识别，不影响表格结构**。于是横躺页上的表格列序仍然按横躺
  的坐标计算，表头整列错位。只要"原始朝向"是正的，表头与原文列序一字不差。

想先确认这个模型在你这台机器上能不能单独调起来、返回什么结构，用仓库里的探测脚本
（只读文件、只打印，不写盘、不联网）：

```bash
python tools/probe_orientation.py 某页.pdf 1
```

**踩过的坑，写在这里省你十分钟**：`predict()` 的入参是**可迭代对象**，直接喂 `bytes`
会被当成"一批输入"逐字节跑一遍，输出直接爆掉。必须传**文件路径**。本库已经处理好。

---

## 2. 方法一：流水线之前把页面转正

对每一页：

1. 以 **100 DPI** 渲染一张缩略图（够判方向，且快）；
2. 用方向模型判角度；
3. 判出非 0 的页才重建：按角度栅格化 → 写成一个新页；
4. **自检**：把转正后的图再判一次，确认模型这次说的是 0；不行就试反方向；两个方向都不行
   就保留原样（宁可不动，也不要把页面搞歪）。

没有歪页时**原样返回原文件**，一点开销都不多花，也不会多产出一个临时文件。

### 重建页的两个参数是实打实测出来的

**渲染 DPI 刻意用 110，低于调用方的 200。** 同一张会被判错的横躺表格（源件是 264 DPI 的
JPEG 扫描件）做多个对照，结果非常挑：

| 流水线把页面放大到 | 1.20x | 1.44x | 1:1 | 0.54x | 4.0x |
|---|---|---|---|---|---|
| 判定 | ✅ | ✅ | ❌ | ❌ | ❌ |

放大相当于一次低通滤波，把源扫描件的 JPEG 振铃抹掉；1:1 或缩小会把振铃原样带进去。
所以按 110 DPI 渲染位图、页面写成**真实物理尺寸**，流水线按它固定的 144 DPI 渲染就得到
约 **1.31x**，落在能判对的窗口里。

**页面尺寸的单位是点（1/72 英寸），不是像素。** 像素数要除以渲染倍率才是物理尺寸；直接
拿像素当点会让页面被放大 zoom 倍（实测 A4 变 A2），流水线再渲染时超出它的 4000px 上限、
被强制降采样，那一页的精度和尺度就跟别的页对不上。本库已经按点写，并有单元测试锁住。

整个过程每页约多花一次 100 DPI 渲染 + 一次分类推理；只有需要转正的页才会多一次位图重建。

---

## 3. 方法二：跨页残条检测

### 判据（刻意收紧：宁可不修，也不误修）

同时满足两条才算命中：

1. 页面**最上面两条长横线**的间距 ≤ **20 磅** —— 残条带很薄；
2. 这条带子里的**暗像素占比 ≤ 5%** —— 带内基本没字。

其中"长横线"= 行内允许 ≤2px 断口（扫描件线条常有小缺口）、长度至少占页宽 30% 的连续
暗像素行。

### 阈值是标定过的

这份报告里那张表的**正常表头行**，实测下来是 **26.8 磅 / 带内 9.2% 有字**——两条判据都
不满足，不会被误判。即便万一误擦，对照实验（人为擦掉第 33 页的表格上边框）显示表格结构
不受影响。

### 擦除方式

把残条带**上方那条表格边框线**用一个白色矩形盖掉（`draw_rect(fill=white)`），页面尺寸和
内容都不动。这一步本身出错也只是"不修"：记录进报告，继续跑 OCR，绝不把整次任务拖下水。

### 作用范围（`fragment_mode`）

| 取值 | 含义 | 什么时候用 |
|---|---|---|
| `rotated`（默认） | 只对**被转正重建**的页做残条修复 | 我们真机端到端验证过的就是这条路径 |
| `all` | 每一页都查一遍残条（检测在低分辨率渲染图上做，擦除直接盖在原页面上） | 文档整体是正的、但顶端同样出现残条 |
| `off` | 只转正，不修残条 | 做 A/B 对照 |

---

## 4. 方法三：多表格单元格错配补丁

上游问题：PaddleOCR issue **#17133** / PaddleX PR **#4756**。PP-StructureV3 处理**多张
表格**时，多张表的裁剪被放进同一个批次做检测，NMS 之后顺序可能变化；上游按「扁平化顺序 +
`box_nums` 切片」分组，于是单元格被分给了错误的表格或列。**单张表格不触发**（没有批处理），
所以症状是「单表 PDF 对、整份报告错」。

本库照搬 PR #4756 的思路：优先用模型输出的 `batch_inds`（记录每个检测框来自批次里哪张表）
分组，拿不到就退回原实现（那种情况下原实现本来就是对的）。

安全边界：

* 任何一步不满足前提就**原样交回原实现**——补丁失效可以，把 OCR 弄挂不行；
* 只有顺序**真的**被打乱过才打印一行，其余时候保持安静；
* `TABLE_PREPASS_PADDLEX_FIX=0` 关闭；`revert()` 撤回。

> ⚠️ 该 PR **未被上游合并、也未经验证**。我们只在自己这批材料上观察到问题消失，
> 所以把它做成可关、可撤的补丁，而不是硬改依赖。

---

## 5. 即插即用

### 5.1 装

```bash
# 只用这个修正层（要 numpy + PyMuPDF）
pip install ".[lib]"          # 从源码装

# 还要跑自带的 HTTP OCR 服务
pip install ".[service]"      # 另见 §6 的 GPU / CPU 两条线
```

不想装包也行：把 `ocr_table_prepass/` 目录整个拷进你的项目即可，核心层只有标准库 +
numpy + PyMuPDF。

### 5.2 插进你自己的流水线（推荐）

```python
from ocr_table_prepass import prepare_pdf

report = prepare_pdf("scan.pdf", dpi=200)     # 你的流水线打算用的 DPI
try:
    print(report.angles)        # 逐页判定角度，-1 = 没判出来
    print(report.rotated)       # {页码: 角度}，空字典 = 这页都是正的
    print(report.strips)        # {页码: 擦除矩形(pt)}
    result = my_pipeline(report.path)         # ← 换成这个文件
finally:
    report.cleanup()                          # 临时文件用完就删
```

完整可运行的例子：`examples/integrate_into_your_pipeline.py`。

要点：

* 没有歪页时 `report.path` **就是**原文件，`report.changed` 为 `False`，不会有额外开销；
* 显式传 `out_path=` 就把结果写到你要的位置（那时 `cleanup()` 不会删它）；
* 只要报告、不要文件：看 `report.as_dict()`，可以直接塞进 JSON 日志。

### 5.3 命令行

```bash
# 本地修正，输出修好的 PDF + JSON 报告
python -m ocr_table_prepass fix scan.pdf -o scan.fixed.pdf --report r.json

# 只想看"哪些页被改了"，不改文件
python -m ocr_table_prepass fix scan.pdf            # 结束时会删掉临时文件

# 对整页都是正的文档也查残条
python -m ocr_table_prepass fix scan.pdf --mode all -o scan.fixed.pdf
```

`--fail-if-unchanged` 会让"什么都没改"时返回退出码 1，方便放进批处理里筛要人工看的文件。

### 5.4 起服务 + 客户端（跨机器部署时用）

```bash
# 服务端（Windows / Linux 各一台，模型跑在有 GPU 的那台）
python -m ocr_table_prepass.service
# 或：uvicorn ocr_table_prepass.service:app --host 0.0.0.0 --port 8766

curl http://127.0.0.1:8766/health
```

```python
from ocr_table_prepass import OcrClient

client = OcrClient("http://ocr-box:8766", token="换一个口令")
print(client.health())

result = client.ocr_file("scan.pdf")          # 完整 OCR
print(result.ok, result.pages, result.meta.get("rotated_pages"))

fixed, report = client.prepass_file("scan.pdf", "scan.fixed.pdf")   # 只要修正层
```

`OcrClient` 只用标准库（`urllib`），可以当协议模板抄。四个端点：

| 端点 | 作用 |
|---|---|
| `GET /health` | 引擎是否就绪 + 前置修正层的开关与参数 |
| `GET /version` | 服务与引擎版本 |
| `POST /jobs` → `GET /jobs/{id}` | 异步 OCR（跨网络 + 几十秒的扫描件，同步请求容易被中间设备掐断） |
| `POST /prepass` | **只要修正层**：返回修正后的 PDF（base64）+ 逐页报告 |

### 5.5 只想要"跨页残条检测"这一块

```python
from ocr_table_prepass import find_fragment_strip

strip = find_fragment_strip(page_image, zoom=dpi / 72.0)   # 单位：磅
if strip:
    # strip = (x0, y0, x1, y1)，自己决定怎么用
    ...
```

`page_image` 可以是 PyMuPDF 的 Pixmap、PIL 的 `Image`、或者 numpy 灰度 / RGB 数组——不挑
渲染库。认不出的输入返回 `None`，不会抛异常。例子见
`examples/find_fragment_on_a_render.py`。

### 5.6 主要接口

| 接口 | 作用 |
|---|---|
| `prepare_pdf(path, dpi=200, out_path=None, fragment_mode="rotated")` | 一行搞定：转正 + 残条，返回 `OrientationReport` |
| `OrientationPrepass.apply(...)` | 同上，但要自己的实例（可以注入自定义分类器、改 DPI） |
| `OrientationReport` | `path` / `changed` / `angles` / `rotated` / `strips` / `notes` / `as_dict()` / `cleanup()` |
| `find_fragment_strip(image, zoom=…)` | 单页残条检测，返回擦除矩形（磅） |
| `find_horizontal_lines(dark)` | 底层：从二值图里找长横线 |
| `erase_fragment_strip(page, rect)` | 在 PyMuPDF 的 Page 上盖白 |
| `paddlex_fix.apply() / revert()` | 多表格单元格错配补丁 |
| `PaddleStructureEngine` / `RapidOCREngine` / `build_engine()` | 引擎适配层（PP-StructureV3 / RapidOCR） |
| `OcrClient` / `OcrResult` | 远端 OCR 与 `/prepass` 的客户端 |

### 5.7 开关、阈值与配置

全部环境变量，**默认值都是我们真机上用的那套**，不改也能跑。为了兼容抽取前的部署，
旧的 `COMPS_OCR_*` 前缀同样有效（新前缀优先）。

| 变量 | 默认 | 作用 |
|---|---|---|
| `TABLE_PREPASS_ORIENT` | `1` | 方向转正总开关（`0` = 整层关掉，做对照用） |
| `TABLE_PREPASS_PROBE_DPI` | `100` | 判角度用的渲染 DPI |
| `TABLE_PREPASS_UPRIGHT_DPI` | `110` | 重建转正页的渲染 DPI（见 §2 的对照表） |
| `TABLE_PREPASS_FRAGMENT_FIX` | `1` | 残条修复总开关 |
| `TABLE_PREPASS_FRAGMENT_GAP_PT` | `20` | 残条带最大高度（磅） |
| `TABLE_PREPASS_FRAGMENT_MAX_DARK` | `0.05` | 残条带内暗像素占比上限 |
| `TABLE_PREPASS_PADDLEX_FIX` | `1` | 多表格单元格错配补丁 |
| `TABLE_PREPASS_DISABLE` | 空 | 关掉 PP-StructureV3 的子模块：`formula,chart,seal,orientation,...` |
| `TABLE_PREPASS_ENGINE` | `auto` | `paddle` / `rapid` / `auto` |
| `TABLE_PREPASS_DPI` | `200` | 渲染 DPI（6GB 显存别往上加） |
| `TABLE_PREPASS_MAX_PAGES` | `50` | 单文件页数上限（超出部分不解析） |
| `TABLE_PREPASS_MAX_UPLOAD_MB` | `80` | 上传体积上限 |
| `TABLE_PREPASS_TOKEN` | 空 | 服务端 Bearer 口令（留空 = 不校验，仅限本机调试） |
| `TABLE_PREPASS_HOST` / `_PORT` | `0.0.0.0` / `8766` | 服务监听地址 |
| `TABLE_PREPASS_URL` | 空 | 客户端要连的服务地址 |
| `TABLE_PREPASS_TIMEOUT` | 按页数算 | 单文件总预算（默认 `max(300, 页数×12 + 120)` 秒） |
| `TABLE_PREPASS_CACHE` | 空（关闭） | 客户端结果缓存目录；`off` = 关 |

---

## 6. 环境与部署

先探测，再决定装哪条线（脚本纯标准库，不装任何东西就能跑）：

```bash
python tools/check_env.py
```

它会报告：NVIDIA 驱动与显存、torch 能否看到 GPU、PaddlePaddle 能否看到 GPU、paddleocr /
rapidocr 的可用性、以及模型权重缓存目录实际落在哪个盘（权重动辄几个 GB，默认在用户目录，
本库会把它**默认改到当前工作目录**下的 `.paddlex/` 与 `.hf_cache/`），最后给一句"下一步
装什么"的结论。

| 路线 | 装什么 | 适合 |
|---|---|---|
| **A. PP-StructureV3（首选）** | `paddlepaddle-gpu` + `paddleocr` + `paddlex[ocr]` + `pymupdf` | 中文 + 表格 + 版面还原，输出 Markdown |
| **C. RapidOCR（兜底）** | `rapidocr-onnxruntime` + `pymupdf` | CUDA 装不动时保证服务可用（只有逐行文本） |

```bash
python -m venv .venv && .venv/bin/activate          # Windows: .venv\Scripts\activate
# A：paddlepaddle-gpu 要走官方 CUDA 索引，wheel 的 CUDA 版本要和本机驱动对齐
python -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -r requirements-gpu.txt
python tools/check_env.py                            # device_count() > 0 才算成功
```

起服务后的自检三连（连不上时按这个顺序查）：

```bash
netstat -ano | findstr :8766                  # 在监听吗？期望 0.0.0.0:8766 LISTENING
curl http://127.0.0.1:8766/health             # 服务本身好不好？（绕开网络）
curl http://<本机组网 IP>:8766/health          # 绑定范围 / 防火墙对不对？
```

`--host 0.0.0.0` 别省：省掉它 uvicorn 默认只绑 `127.0.0.1`，本机 curl 通、从别的机器一律
连不上（现象是 curl 直接 "Couldn't connect"）。首次启动会把模型加载进显存（几十秒），
所以第一次 `/health` 慢是正常的——服务里已有预热线程。

---

## 7. 验证到什么程度

写清楚边界，比写得漂亮重要。

**真机验证过：**

* 跨页残条：一份 43 页并购协议，同一张表跨第 33/34 页。修复前 110/120 档下第 34 页两家
  公司的内容互相串行，100/132/144 档下各有一两行错开；擦掉残条带上方的边框线后，
  110/120/144 三档 + **整份 43 页文档端到端**均恢复正确。
* 阈值标定：正常表头行 26.8 磅 / 9.2% 有字（不误判）；误擦对照（人为擦掉第 33 页上边框）
  显示表格结构不受影响。
* 方向转正：横躺页的表格表头列序从"整列错位"恢复到与原文一字不差；DPI 对照见 §2 的表。
* 多表格补丁：自己这批材料上问题消失（补丁本身照搬未合并的上游 PR）。

**没验证过 / 已知边界：**

* `fragment_mode="all"`（在不需要转正的页上擦残条）：检测逻辑与擦除都是同一套，但我们
  只在 `rotated` 这条路径上做过端到端验证，所以默认值是 `rotated`。
* 数字版 PDF（有文字层）的原页擦除：我们是先栅格化再重建，所以不受影响；`mode=all` 会
  直接盖在原页上，那一路没做过对照实验。
* 带 `page.rotation` 标记的 PDF 页：坐标换算按未旋转页面处理，没实测过这种输入。
* 被转正的页会**位图重建**，因此原本的文字层会丢失。扫描件本来就没有文字层，不受影响；
  数字版 PDF 只是"页面被转了一下"的话，你要心里有数（那种文件通常也不需要 OCR）。
* 上游那个补丁未经 PaddleX 官方合并或验证。

---

## 8. 代码结构

```
ocr_table_prepass/
├── _compat.py       可选依赖降级 + 环境变量（新前缀 / 旧前缀）
├── orient.py        方向模型封装 + 逐页转正重建        ← 方法一
├── fragment.py      跨页残条检测与擦除（纯像素）        ← 方法二
├── paddlex_fix.py   多表格单元格错配补丁                ← 方法三
├── engine.py        引擎适配层（PP-StructureV3 / RapidOCR），把前置修正接进流水线
├── service.py       FastAPI 服务（/health /version /jobs /prepass）
├── client.py        只用标准库的 HTTP 客户端
└── __main__.py      命令行：fix / ocr / health
tools/
├── check_env.py         环境探测（GPU / CUDA / 缓存目录）
└── probe_orientation.py 方向模型的单独探测（确认能调起来、返回什么结构）
examples/                两段可以直接跑的接入示例
tests/                   86 个用例：82 个纯离线（不装 paddleocr、不碰 GPU、不联网）
                         + 4 个真 PyMuPDF 的集成用例（真渲染、真画矩形；没装就跳过）
```

## 9. 跑测试

```bash
python -m unittest discover -v        # 或 pytest
```

绝大部分用例离线：用假的 `pymupdf` / `paddleocr` 换掉 `sys.modules`，覆盖的是"该生效时
生效、该降级时降级"——例如模型缺失、PyMuPDF 缺失、擦除本身失败、引擎不接受子模块开关
参数时，都必须安静地退化，不能让 OCR 挂掉。

还有一组用**真 PyMuPDF** 的集成用例（`tests/test_integration_pymupdf.py`，装了 pymupdf 才
跑）：现场造一份带横躺页和顶端残条的 PDF，验证重建页的尺寸是真实物理尺寸（A4 不能变成
A2）、白色矩形真的盖掉了那根边框线、没毛病的页一根汗毛都不动。

## 10. 出处与许可

从我们的文档处理流水线里抽出来的（原路径：Windows 侧 OCR 服务 + Mac 侧调用层），
只保留了"表格 OCR 之前的那一层"。条款见 [LICENSE](LICENSE)（MIT）。

补丁参照 PaddleX PR #4756 / PaddleOCR issue #17133；模型来自 PaddleOCR 3.x 的
`DocImgOrientationClassification`。
