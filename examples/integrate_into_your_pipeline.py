"""把前置修正插进任何 OCR 流水线：三行代码，换掉你原来送进去的那个文件。

这里用 PP-StructureV3 举例——它是我们实测过的那条线；换成别的引擎（MinerU、Surya、
云 API）也一样，因为这一层只做文件级修正，不碰识别逻辑。

    python examples/integrate_into_your_pipeline.py scan.pdf
"""

from __future__ import annotations

import sys

from ocr_table_prepass import prepare_pdf


def ocr(path: str) -> str:
    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(use_doc_orientation_classify=True)

    # ---- 插进去的就是这段：转正歪页 + 擦掉跨页残条 ------------------------ #
    report = prepare_pdf(path, dpi=200)
    try:
        print(f"逐页角度：{report.angles}")
        if report.rotated:
            print(f"转正了第 {sorted(report.rotated)} 页")
        if report.strips:
            print(f"第 {sorted(report.strips)} 页擦掉了顶部跨页残条")
        result = pipeline.predict(input=report.path)      # ← 必须是修正后的文件
    finally:
        report.cleanup()                                  # 临时文件用完就删

    items = result if isinstance(result, (list, tuple)) else [result]
    return "\n\n".join(getattr(item, "markdown", "") or "" for item in items)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print(ocr(sys.argv[1]))
