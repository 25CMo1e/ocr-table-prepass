"""探测：能不能单独调用方向分类模型，拿到页面的旋转角度。

为什么需要它
------------
PP-StructureV3 内部的方向模块**会在工作**（实测：一张 180° 倒置的正文页，41 行里
只有 3 个标点不同，其余逐字读对——没有方向纠正是做不到的），但它纠正后的坐标系
**没有传给表格结构那一步**：表格的单元格顺序仍然按原始朝向计算。

所以我们要在**送进流水线之前**自己把页面转正——而前提是能单独拿到"转多少度"这个数。
这个脚本就是在真机上确认这件事：依次尝试几种入口，把能用的那个和它的返回结构打印出来。

用法（在仓库根目录下，虚拟环境已激活）：

    python tools/probe_orientation.py "某页.pdf" [页码]

脚本只读文件、只打印信息，不写盘、不上传、不改任何配置。
"""

from __future__ import annotations

import tempfile
import sys
import traceback
from pathlib import Path

# 报告同时写盘：控制台一旦滚过去就看不回来了，文件可以慢慢翻。
REPORT = Path(__file__).with_name("probe_orientation.out.txt")
_LINES: list[str] = []


def say(text: str = "") -> None:
    _LINES.append(text)


def _preview(value, limit: int = 200) -> str:
    """把任意值压成一行短摘要；二进制和数组只报体积，绝不打内容。"""

    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} 字节的二进制>"
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"<数组 shape={getattr(value, 'shape', '?')} dtype={getattr(value, 'dtype', '?')}>"
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def render(path: str, pno: int, rotate: int = 0):
    """把某一页渲染成 PNG 字节；rotate 是像素级旋转（0/90/180/270）。"""

    try:
        import pymupdf as fitz          # 新版包名，老写法会打 deprecation 警告
    except ImportError:
        import fitz

    doc = fitz.open(path)
    try:
        page = doc[pno - 1]
        zoom = 200 / 72.0                       # 200 DPI，和服务的默认值一致
        matrix = fitz.Matrix(zoom, zoom)
        if rotate:
            matrix = matrix.prerotate(rotate)   # 先在光栅层旋转，等价于"把图转正/转歪"
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def describe(result, depth: int = 0) -> None:
    """把一个预测结果的结构摊开，便于判断怎么取角度。"""

    pad = "      " + "  " * depth
    if depth > 3:
        say(f"{pad}（层级太深，略）")
        return
    say(f"{pad}类型 : {type(result).__name__}")
    if isinstance(result, (list, tuple)):
        say(f"{pad}条数 : {len(result)}")
        for item in result[:2]:
            describe(item, depth + 1)
        return
    for attr in ("json", "res", "label_names", "scores", "class_ids",
                 "label", "score", "input_path"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if callable(value):
                try:
                    value = value()
                except Exception as exc:  # noqa: BLE001
                    value = f"<调用失败 {type(exc).__name__}>"
            say(f"{pad}{attr:12s}: {_preview(value)}")
    names = [n for n in dir(result) if not n.startswith("_")]
    if names:
        say(f"{pad}可用属性 : {', '.join(names[:26])}")


def try_entry(label: str, build):
    """尝试一个入口；成功就打印结果结构并返回 True。"""

    say(f"\n  [{label}]")
    try:
        model = build()
    except Exception as exc:  # noqa: BLE001
        say(f"      构造失败：{type(exc).__name__}: {exc}")
        return None
    say(f"      构造成功 : {type(model).__name__}")
    return model


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    pdf = argv[0]
    pno = int(argv[1]) if len(argv) > 1 else 1
    if not Path(pdf).exists():
        print("[错误] 找不到文件：", pdf)
        return 1

    say(f"目标：{Path(pdf).name} 第 {pno} 页")
    say("\n=== 1. 看看 paddleocr / paddlex 里有哪些方向相关的入口 ===")
    for mod_name in ("paddleocr", "paddlex"):
        try:
            mod = __import__(mod_name)
        except ImportError as exc:
            say(f"  {mod_name}: 导入失败 {exc}")
            continue
        names = [n for n in dir(mod)
                 if any(k in n.lower() for k in ("orient", "docimg", "unwarp"))]
        say(f"  {mod_name}: {names or '（没有匹配的名字）'}")

    say("\n=== 2. 渲染两版图：原样、像素级转 90°（都落成临时文件再喂） ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="probe-ori-"))
    files = {}
    for label, rot in (("原样", 0), ("转90", 90), ("转180", 180)):
        blob = render(pdf, pno, rot)
        path = tmpdir / f"page-{rot}.png"
        path.write_bytes(blob)
        files[label] = path
        say(f"  {label}: {path.name}  {len(blob)/1024:.0f} KB")

    say("\n=== 3. 尝试几个可能的入口 ===")
    model = None
    attempts = [
        ("paddleocr.DocImgOrientationClassification()",
         lambda: __import__("paddleocr").DocImgOrientationClassification()),
        ("paddlex.create_model('PP-LCNet_x1_0_doc_ori')",
         lambda: __import__("paddlex").create_model("PP-LCNet_x1_0_doc_ori")),
        ("paddleocr.TextLineOrientationClassification()",
         lambda: __import__("paddleocr").TextLineOrientationClassification()),
    ]
    for label, build in attempts:
        model = try_entry(label, build)
        if model is not None:
            break

    if model is None:
        say("\n[结论] 三个入口都没构造成功。把这份报告发回来，我据此换写法。")
        _flush()
        return 1

    say("\n=== 4. 喂三种朝向，看判断是否随之改变 ===")
    for label, path in files.items():
        say(f"\n  --- {label} ---")
        try:
            # 只传路径、不传 bytes：bytes 是可迭代的，模型可能把它当成"一批输入"，
            # 于是逐字节跑一遍，输出直接爆掉（上一版就是这么翻车的）。
            result = model.predict(str(path))
            if hasattr(result, "__iter__") and not isinstance(
                    result, (list, tuple, dict, str, bytes)):
                result = list(result)       # predict 返回生成器，先摊开
            describe(result)
        except Exception:  # noqa: BLE001
            say("      预测失败：")
            say(traceback.format_exc())

    say("\n[提示] 三种朝向给出的 label/class_id 不同 → 模型确实在判断方向，")
    say("       那个数就是我们要的。把这份报告发回来即可。")
    _flush()
    return 0


def _flush() -> None:
    """把报告写盘，控制台只留摘要——控制台滚过去就看不回来了。"""

    REPORT.write_text("\n".join(_LINES), encoding="utf-8")
    print(f"\n完整输出已写入：{REPORT}")
    print("—— 下面只是前 25 行预览，完整内容请打开上面那个文件 ——\n")
    for line in _LINES[:25]:
        print(line, flush=True)
    print(f"\n（共 {len(_LINES)} 行，其余见 {REPORT.name}）")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
