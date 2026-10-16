"""只用「跨页残条检测」这一块：自己渲染页面，自己决定怎么擦。

给这样的人用：已经有自己的渲染 / 版面流程，只想问问「这一页顶端是不是跨页残条」。
检测不依赖 PyMuPDF，喂 numpy 灰度数组、PIL 图、PyMuPDF Pixmap 都行。

    python examples/find_fragment_on_a_render.py scan.pdf
"""

from __future__ import annotations

import sys

from ocr_table_prepass import find_fragment_strip

DPI = 100      # 检测在低分辨率下做就够了，还快


def main(path: str) -> int:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    zoom = DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    doc = fitz.open(path)
    try:
        for number, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            strip = find_fragment_strip(pix, zoom=zoom)
            if strip is None:
                continue
            x0, y0, x1, y1 = strip
            print(f"第 {number} 页：顶端跨页残条，"
                  f"要擦的矩形 = ({x0:.1f}, {y0:.1f}) ~ ({x1:.1f}, {y1:.1f}) pt")
            # 要动手改，就盖上白色矩形（数字版 PDF 直接盖原页也行）：
            # page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=None, fill=(1, 1, 1))
    finally:
        doc.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1]))
