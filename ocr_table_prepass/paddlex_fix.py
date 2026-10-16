"""给 paddlex 打的运行时补丁：修复「多表格批处理导致单元格错配」。

上游问题
========
PaddleOCR issue #17133 / PaddleX PR #4756 描述的现象是：PP-StructureV3 处理**多张
表格**时单元格检测结果会混乱——因为多张表格的裁剪区域被放进**同一个批次**做检测，
NMS 之后检测结果的顺序可能变化，而原实现是按「扁平化后的顺序 + ``box_nums`` 切片」
分组的，于是单元格被分给了错误的表格或列。

**单张表格不会触发**（没有批处理），所以症状是「单表 PDF 对、整份报告错」——
排查时特别容易怀疑到别的地方去。

修复思路（照搬 PR #4756；注意该 PR **未被上游合并、也未经验证**）
================================================================
优先用模型输出的 ``batch_inds`` 分组，它记录每个检测框来自批次里的哪张表；拿不到
时才退回原来的切片方式（那种情况下原实现本来就是对的）。

安全边界
========
* ``TABLE_PREPASS_PADDLEX_FIX=0`` 可关闭，默认开启。
* 任何一步不满足前提就**原样交回原实现**：补丁失效可以，把 OCR 弄挂不行。
* 只有当分组结果与 ``box_nums`` **不一致**（即顺序真的被打乱过）时才打印一行，
  其余时候保持安静。
"""

from __future__ import annotations

from ._compat import env_flag


def enabled() -> bool:
    return env_flag("PADDLEX_FIX", True)


def _group_by_batch_index(pred):
    """按批次索引重新分组检测框；前提不满足时返回 None（交回原实现）。"""

    import numpy as np

    if len(pred) < 3:                       # 没有第三个输出，不可能有 batch_inds
        return None
    boxes, box_nums = pred[0], pred[1]
    if not isinstance(boxes, np.ndarray) or len(boxes) == 0:
        return None
    if not isinstance(box_nums, np.ndarray) or box_nums.ndim != 1 or len(box_nums) == 0:
        return None

    batch_inds = None
    for candidate in pred[2:]:
        # batch_inds：一维整型、长度等于框数、取值落在 [0, 批次数)
        if not isinstance(candidate, np.ndarray) or candidate.ndim != 1:
            continue                        # 三维或浮点的是分割 mask，不是批次索引
        if candidate.dtype.kind not in "iu" or len(candidate) != len(boxes):
            continue
        if int(candidate.min()) < 0 or int(candidate.max()) >= len(box_nums):
            continue
        batch_inds = candidate.astype(np.int32)
        break
    if batch_inds is None:
        return None

    ids = np.unique(batch_inds)
    groups = [np.asarray(boxes[batch_inds == i]) for i in ids]
    # 「没被打乱」时 batch_inds 应该正好是 0×n0, 1×n1, …；不相等就说明顺序被 NMS 改过。
    expected_seq = np.repeat(np.arange(len(box_nums)), box_nums.astype(int))
    if not np.array_equal(expected_seq, batch_inds):
        # 只有顺序真的被打乱过才出声——这一行就是补丁在起作用的证据。
        print(f"[paddlex-fix] 检测顺序被打乱：批次数 {len(box_nums)}，框数 {len(boxes)}，"
              f"已按 batch_inds 重新分组")
    return [{"boxes": g} for g in groups]


def apply() -> str:
    """给 ``DetRunnerPredictor._format_output`` 打补丁，返回一句说明。永不抛异常。

    必须在**流水线构造之前**调用：补丁要早于 paddlex 建立内部预测器。
    """

    if not enabled():
        return "TABLE_PREPASS_PADDLEX_FIX=0，未打补丁"
    try:
        from paddlex.inference.models.object_detection.predictor import DetRunnerPredictor
    except Exception as exc:  # noqa: BLE001 - 路径变了或没装，都只是"不生效"
        return f"找不到目标类，跳过（{type(exc).__name__}: {exc}）"

    original = getattr(DetRunnerPredictor, "_format_output", None)
    if original is None:
        return "目标类没有 _format_output，跳过"
    if getattr(original, "_prepass_patched", False):
        return "已打过补丁"

    def patched(self, pred):
        try:
            grouped = _group_by_batch_index(pred)
        except Exception:  # noqa: BLE001 - 补丁出问题绝不能影响识别
            grouped = None
        if grouped is not None:
            return grouped
        return original(self, pred)

    patched._prepass_patched = True            # type: ignore[attr-defined]
    patched._prepass_original = original       # type: ignore[attr-defined]
    DetRunnerPredictor._format_output = patched
    return "已打补丁（多表格单元格错配，参照 PaddleX PR #4756）"


def revert() -> str:
    """撤回补丁（给 A/B 对照用）。"""

    try:
        from paddlex.inference.models.object_detection.predictor import DetRunnerPredictor
    except Exception as exc:  # noqa: BLE001
        return f"找不到目标类（{type(exc).__name__}: {exc}）"
    current = getattr(DetRunnerPredictor, "_format_output", None)
    original = getattr(current, "_prepass_original", None)
    if original is None:
        return "没打过补丁"
    DetRunnerPredictor._format_output = original
    return "已撤回补丁"
