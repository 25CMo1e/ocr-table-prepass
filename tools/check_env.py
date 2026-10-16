"""OCR 环境探测（纯标准库，无需先装任何第三方包）。

在要跑 OCR 的那台机器上先跑这个，确认 GPU / CUDA 栈到底能不能用，再决定装哪个引擎：

    python tools/check_env.py

检查内容：
  1. NVIDIA 驱动与显卡型号（nvidia-smi）——顺便确认显存够不够；
  2. PyTorch 是否已装、能否看到 GPU（PaddleOCR-VL 这类 torch 路线要先过这关）；
  3. PaddlePaddle 是否已装、能否看到 GPU（PP-StructureV3 路线要先过这关）；
  4. PaddleOCR / RapidOCR 两个候选引擎的可用性与版本；
  5. Python 环境与模型缓存目录实际落在哪个盘（权重默认在用户目录，动辄几个 GB）。

最后给出「下一步该装什么」的结论。脚本不上传、不写盘、不发网络请求。
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], timeout: int = 25) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def probe_nvidia() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"ok": False, "note": "PATH 里找不到 nvidia-smi（驱动没装或没进 PATH）"}
    code, out = _run([exe, "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"])
    if code != 0:
        return {"ok": False, "note": out.strip()[:200] or "nvidia-smi 执行失败"}
    gpus = [line.strip() for line in out.strip().splitlines() if line.strip()]
    return {"ok": bool(gpus), "gpus": gpus}


def probe_torch() -> dict:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return {"installed": False}
    info = {"installed": True, "version": getattr(torch, "__version__", "?"),
            "cuda_built": bool(getattr(torch.version, "cuda", None)),
            "cuda_version": getattr(torch.version, "cuda", None)}
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / 1024 ** 3, 1)
    except Exception as exc:  # noqa: BLE001 - 探测脚本，任何异常都只报告
        info["cuda_available"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def probe_paddle() -> dict:
    try:
        paddle = importlib.import_module("paddle")
    except ImportError:
        return {"installed": False}
    info = {"installed": True, "version": getattr(paddle, "__version__", "?")}
    try:
        import paddle.device as paddle_device  # noqa: PLC0415

        info["cuda_compiled"] = bool(
            getattr(paddle.device, "is_compiled_with_cuda", lambda: False)())
        info["device_count"] = int(paddle_device.cuda.device_count())
        info["current_device"] = str(getattr(paddle.device, "get_device", lambda: "?")())
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def probe_engines() -> dict:
    out: dict = {}
    try:
        mod = importlib.import_module("paddleocr")
        out["paddleocr"] = {"installed": True, "version": getattr(mod, "__version__", "?")}
        out["paddleocr"]["has_structure_v3"] = hasattr(mod, "PPStructureV3")
    except ImportError:
        out["paddleocr"] = {"installed": False}
    try:
        mod = importlib.import_module("rapidocr_onnxruntime")
        out["rapidocr"] = {"installed": True, "version": getattr(mod, "__version__", "?")}
    except ImportError:
        try:
            mod = importlib.import_module("rapidocr")
            out["rapidocr"] = {"installed": True, "version": getattr(mod, "__version__", "?")}
        except ImportError:
            out["rapidocr"] = {"installed": False}
    return out


def probe_paddlex_extra() -> dict:
    """PP-StructureV3 的真正的依赖在 ``paddlex[ocr]`` 这个 extra 里。

    只装 ``paddleocr`` 时能 import，但构造流水线会抛 DependencyError。这里直接借用
    paddlex 自己的依赖检查做一次廉价探测（不下载模型、不建流水线）。
    """

    try:
        paddlex = importlib.import_module("paddlex")
    except ImportError:
        return {"installed": False}
    info = {"installed": True, "version": getattr(paddlex, "__version__", "?")}
    try:
        from paddlex.utils.deps import require_extra  # noqa: PLC0415
    except ImportError:
        info["extra_check"] = "unknown（paddlex 版本过新/过旧，探测不到）"
        return info
    try:
        require_extra("ocr", obj_name="PP-StructureV3")
        info["ocr_extra_ok"] = True
    except Exception as exc:  # noqa: BLE001 - 缺依赖就是这里抛
        info["ocr_extra_ok"] = False
        info["ocr_extra_error"] = str(exc).splitlines()[0][:200]
    return info


def _dir_size_mb(path: Path, limit_files: int = 40000) -> float:
    total = 0
    count = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
                count += 1
                if count >= limit_files:
                    return round(total / 1024 ** 2, 1)
    except OSError:
        pass
    return round(total / 1024 ** 2, 1)


def probe_storage() -> dict:
    """Python 环境与各类缓存目录实际落在哪个盘。

    pip 装的依赖进虚拟环境（``sys.prefix``），但**模型权重默认不在项目目录里**，
    而是落在用户目录的缓存中。这里把候选位置和占用都列出来，便于决定要不要搬到别的盘。
    """

    home = Path.home()
    known_env = ("PADDLE_PDX_CACHE_HOME", "PADDLE_PDX_MODELS_HOME", "PADDLEOCR_HOME",
                 "PADDLE_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE")
    env_set = {k: os.environ[k] for k in known_env if os.environ.get(k)}
    env_set.update({k: v for k, v in os.environ.items()
                    if k.upper().startswith("PADDLE") and k not in env_set})

    candidates = [
        home / ".paddlex",                      # PaddleX 3.x / PaddleOCR 3.x 默认位置
        home / ".paddleocr",                    # PaddleOCR 2.x 时代的位置
        home / ".cache" / "paddle",
        home / ".cache" / "huggingface",
        Path(os.environ.get("LOCALAPPDATA") or home) / "pip" / "Cache",
    ]
    temp_dir = os.environ.get("TEMP") or os.environ.get("TMP")
    if temp_dir:
        candidates.append(Path(temp_dir))
    # 注意：不要在这里回退到 home——那会把整个用户目录遍历一遍，既慢又没意义。
    dirs, seen = [], set()
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        exists = path.exists()
        dirs.append({"path": str(path), "exists": exists,
                     "size_mb": _dir_size_mb(path) if exists else 0.0})
    return {"python_env": sys.prefix, "env": env_set, "dirs": dirs}


def verdict(report: dict) -> list[str]:
    tips: list[str] = []
    nv, torch_info, paddle_info, engines, paddlex_info = (
        report["nvidia"], report["torch"], report["paddle"], report["engines"],
        report["paddlex"])

    if not nv.get("ok"):
        tips.append("显卡驱动没就绪：先去 nvidia.com 装驱动，否则谈不上 GPU 加速。")
    else:
        tips.append(f"显卡识别正常：{'; '.join(nv.get('gpus', []))}"
                    "（6GB 显存这一类要用 batch=1、渲染 200 DPI 左右；再往上加容易 OOM）。")

    paddle_gpu = (paddle_info.get("installed")
                  and (paddle_info.get("cuda_compiled") or paddle_info.get("device_count", 0) > 0))
    if paddle_gpu and engines["paddleocr"].get("installed"):
        if paddlex_info.get("installed") and paddlex_info.get("ocr_extra_ok") is False:
            tips.append(f"⚠️ 装了 paddleocr {engines['paddleocr'].get('version')}，但缺 paddlex 的 "
                        f"ocr 依赖（否则构造 PP-StructureV3 会抛 DependencyError）。"
                        f"补装：python -m pip install \"paddlex[ocr]=={paddlex_info.get('version')}\"")
        else:
            tips.append("✅ 已满足 PP-StructureV3 的 GPU 条件，可以直接起服务。")
    elif paddle_info.get("installed") and not paddle_gpu:
        tips.append("⚠️ 装了 PaddlePaddle 但是 CPU 版：需卸载后装 paddlepaddle-gpu"
                    "（注意 wheel 的 CUDA 版本要和本机 CUDA/cuDNN 对齐）。")
    else:
        tips.append("→ 首选路线：装 paddlepaddle-gpu + paddleocr，走 PP-StructureV3（GPU）。")

    if torch_info.get("installed") and torch_info.get("cuda_available"):
        tips.append(f"✅ torch 也能看到 GPU（{torch_info.get('device')}），"
                    "PaddleOCR-VL 这类 torch 路线同样可行。")
    elif torch_info.get("installed"):
        tips.append("torch 装了但看不到 GPU（多半是装成了 CPU 版 wheel）。")

    if not engines["paddleocr"].get("installed") and not engines["rapidocr"].get("installed"):
        tips.append("→ 兜底路线：pip install rapidocr-onnxruntime（CPU 也能跑，"
                    "CUDA 装不动时用它保证服务可用）。")

    storage = report["storage"]
    tips.append(f"Python 环境：{storage['python_env']}（pip 依赖装在这里）")
    used = [d for d in storage["dirs"] if d["exists"] and d["size_mb"] > 0]
    if used:
        tips.append("已占用缓存目录：" + "; ".join(
            f"{d['path']} ({d['size_mb']}MB)" for d in used))
    if storage["env"]:
        tips.append("已设置的模型缓存变量：" + ", ".join(
            f"{k}={v}" for k, v in storage["env"].items()))
    else:
        tips.append("未设置模型缓存变量：权重默认落在 C 盘用户目录（见上面 dirs），"
                    "要搬去 E 盘就设 PADDLE_PDX_CACHE_HOME / PADDLE_PDX_MODELS_HOME。")
    return tips


def main() -> int:
    report = {
        "python": sys.version.split()[0],
        "nvidia": probe_nvidia(),
        "torch": probe_torch(),
        "paddle": probe_paddle(),
        "paddlex": probe_paddlex_extra(),
        "engines": probe_engines(),
        "storage": probe_storage(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n=== 结论 ===")
    for line in verdict(report):
        print(" -", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
