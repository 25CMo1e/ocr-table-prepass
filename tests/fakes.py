"""离线测试用的假件：假的 PyMuPDF、假的 paddleocr、假的方向分类模型。

这一层修正全是"能降级"的代码路径，真正难测的是它们**该生效时生效、该降级时降级**，
所以测试一律不碰 GPU、不装 paddleocr、不联网。
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# 假的 paddleocr / PP-StructureV3
# --------------------------------------------------------------------------- #


class FakePipeline:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def predict(self, input=None):  # pragma: no cover - 本套测试不跑真推理
        return []


def fake_paddleocr(record: list, fail_on_kwargs: bool = False,
                   fail_error: Optional[Exception] = None) -> types.ModuleType:
    module = types.ModuleType("paddleocr")
    module.__version__ = "3.7.0"

    def PPStructureV3(**kwargs):
        record.append(kwargs)
        if kwargs and fail_error is not None:
            raise fail_error
        if kwargs and fail_on_kwargs:
            raise TypeError("unexpected keyword argument")
        return FakePipeline(**kwargs)

    module.PPStructureV3 = PPStructureV3
    return module


# --------------------------------------------------------------------------- #
# 假的 PyMuPDF + 假的页面
# --------------------------------------------------------------------------- #


def gray_with_fragment(width: int = 595, height: int = 842, top: int = 20,
                       bottom: int = 33) -> np.ndarray:
    """白底 + 顶部两条长横线（相距 13px）+ 带内零星墨点：标准的跨页残条几何。"""

    gray = np.full((height, width), 255, dtype=np.uint8)
    gray[top:top + 1, 10:width - 10] = 40
    gray[bottom:bottom + 1, 10:width - 10] = 40
    gray[top + 6, 200:216:4] = 40          # 残条里那一点点字尾
    return gray


class FakePix:
    """够像 PyMuPDF 的 Pixmap：samples / width / height / n / stride 都能读到。"""

    def __init__(self, width: int = 595, height: int = 842, tag: bytes = b"angle:0",
                 gray: Optional[np.ndarray] = None) -> None:
        self.width, self.height, self.tag = width, height, tag
        if gray is None:
            gray = np.full((height, width), 255, dtype=np.uint8)
        self.n = 3
        self.stride = width * 3
        self.samples = np.repeat(gray[:, :, None], 3, axis=2).tobytes()

    def tobytes(self, fmt: str) -> bytes:
        return self.tag


class FakeMatrix:
    def __init__(self, a: float = 1, b: float = 1) -> None:
        self.rot = 0

    def prerotate(self, deg: int):
        self.rot = deg
        return self


class FakePage:
    def __init__(self, number: int, angle: int, gray: Optional[np.ndarray] = None) -> None:
        self.number, self.angle, self.gray = number, angle, gray

    def get_pixmap(self, matrix=None, alpha: bool = False) -> FakePix:
        """把"页面的固有角度 + 本次请求的旋转"合起来算，结果写进 tag。

        这样假的方向模型读 tag 就能给出**真实**的角度判定，180 / 270 也不会被简化成 90。
        """

        turned = (self.angle + getattr(matrix, "rot", 0)) % 360
        return FakePix(tag=f"angle:{turned}".encode("ascii"), gray=self.gray)


class FakeNewPage:
    """``new_page()`` 建出来的页：只记下被盖上去的图像和矩形。"""

    def __init__(self) -> None:
        self.images: list = []
        self.rects: list = []

    def insert_image(self, rect, stream=None) -> None:
        self.images.append((rect, stream))

    def draw_rect(self, rect, **kwargs) -> None:
        self.rects.append((rect, kwargs))


class FakeCopiedPage(FakeNewPage):
    """``insert_pdf`` 拷进来的页：矩形画在原页面上。"""


class FakeDoc:
    def __init__(self, angles: tuple = (), gray: Optional[np.ndarray] = None) -> None:
        self._pages = [FakePage(i, a, gray) for i, a in enumerate(angles)]
        self.pages_out: list = []
        self.copied: list = []
        self.saved: list = []

    # -- 源文档 ------------------------------------------------------------ #
    def __iter__(self):
        return iter(self._pages)

    def __len__(self) -> int:
        return len(self._pages)

    def close(self) -> None:
        pass

    # -- 输出文档 ---------------------------------------------------------- #
    @property
    def page_count(self) -> int:
        return len(self.pages_out)

    def insert_pdf(self, src, from_page: int = 0, to_page: int = 0) -> None:
        self.copied.append((from_page, to_page))
        self.pages_out.append(FakeCopiedPage())

    def new_page(self, width: float = 0, height: float = 0) -> FakeNewPage:
        self.page_size = (width, height)      # 记下来，供断言页面尺寸对不对
        page = FakeNewPage()
        self.pages_out.append(page)
        return page

    def page_at(self, index: int) -> FakeNewPage:
        return self.pages_out[index]

    def __getitem__(self, index: int) -> FakeNewPage:
        """和 PyMuPDF 一样按下标取**输出文档**里的页（``out[i]``）。"""

        return self.pages_out[index]

    def save(self, path: str) -> None:
        self.saved.append(path)
        Path(path).write_bytes(b"%PDF-1.4 rebuilt")


def fake_pymupdf(angles: tuple = (), doc: Optional[FakeDoc] = None,
                 gray: Optional[np.ndarray] = None) -> types.ModuleType:
    """假 PyMuPDF：``open()`` 返回一个按给定角度搭出来的假文档。"""

    module = types.ModuleType("pymupdf")
    built = doc if doc is not None else FakeDoc(angles, gray)
    module.open = lambda path=None: built
    module.Matrix = FakeMatrix
    module.Rect = lambda *args: tuple(args)
    return module


# --------------------------------------------------------------------------- #
# 假的方向分类模型
# --------------------------------------------------------------------------- #


def fake_orientation_classifier() -> type:
    """读 pixmap 的 tag（``angle:NN``）当角度：够真，180 / 270 都区分得开。"""

    class _Result:
        def __init__(self, label: int) -> None:
            self.json = {"res": {"label_names": [label], "scores": [0.9]}}

    class _Classifier:
        def predict(self, path):
            data = Path(path).read_bytes()
            try:
                label = int(data.decode("ascii").split(":", 1)[1])
            except (ValueError, UnicodeDecodeError, IndexError):
                label = 90
            return [_Result(label)]

    return _Classifier


# --------------------------------------------------------------------------- #
# 假的 PyMuPDF：只给"按页数裁剪"用
# --------------------------------------------------------------------------- #


class FakeLimitDoc:
    def __init__(self, page_count: int) -> None:
        self.page_count = page_count
        self.sliced: list = []
        self.saved: list = []

    def insert_pdf(self, src, from_page: int = 0, to_page: int = 0) -> None:
        self.sliced.append((from_page, to_page))

    def save(self, path: str) -> None:
        self.saved.append(path)
        Path(path).write_bytes(b"%PDF-1.4 sliced")

    def close(self) -> None:
        pass


def fake_limit_fitz(page_count: int, opened: Optional[list] = None):
    """``fitz.open(path)`` → 页数为 ``page_count`` 的假文档；``fitz.open()`` → 空文档。"""

    module = types.ModuleType("pymupdf")

    def open_(path=None):
        doc = FakeLimitDoc(page_count if path else 0)
        if opened is not None:
            opened.append((path, doc))
        return doc

    module.open = open_
    return module
