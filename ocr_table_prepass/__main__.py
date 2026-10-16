"""命令行：本地跑前置修正，或调远端服务做 OCR。

    # 1) 只做前置修正：转正歪页 + 擦掉跨页残条，输出修好的 PDF 和 JSON 报告
    python -m ocr_table_prepass fix scan.pdf -o scan.fixed.pdf --report r.json

    # 2) 调远端服务做完整 OCR（远端没起也能跑第 1 条）
    python -m ocr_table_prepass ocr scan.pdf --url http://127.0.0.1:8766

    # 3) 看一眼远端服务状态
    python -m ocr_table_prepass health --url http://127.0.0.1:8766
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._compat import env_text
from .client import OcrClient
from .orient import OrientationPrepass


def _add_client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default="", help="服务地址，默认取 TABLE_PREPASS_URL")
    parser.add_argument("--token", default="", help="Bearer 口令，默认取 TABLE_PREPASS_TOKEN")
    parser.add_argument("--allow-public", action="store_true",
                        help="允许把文件发往公网端点（默认不拦，见 --guard-public）")
    parser.add_argument("--guard-public", action="store_true",
                        help="只允许发往内网 / 组网地址（内部材料外发前的保险）")


def _client(args) -> OcrClient:
    return OcrClient(args.url, args.token,
                     allow_public=not getattr(args, "guard_public", False))


def _cmd_fix(args) -> int:
    prepass = OrientationPrepass()
    report = prepass.apply(args.path, out_path=args.output or None,
                           fragment_mode=args.mode)
    try:
        if not report.changed:
            print("这份文件不需要修正（没有歪页，也没有检测到跨页残条）")
        else:
            print(f"已写出修正后的文件：{report.path}")
        print(f"页数 {report.pages}，逐页角度 {report.angles}")
        for page, angle in sorted(report.rotated.items()):
            print(f"  第 {page} 页：旋转 {angle}°，已转正")
        for page, rect in sorted(report.strips.items()):
            print(f"  第 {page} 页：擦了顶部跨页残条 y={rect[1]:.1f}~{rect[3]:.1f}pt")
        for note in report.notes:
            print(f"  注：{note}")
        if args.report:
            with open(args.report, "w", encoding="utf-8") as fh:
                json.dump(report.as_dict(), fh, ensure_ascii=False, indent=2)
            print(f"报告已写入：{args.report}")
        return 0 if report.changed or not args.fail_if_unchanged else 1
    finally:
        # 只有临时文件才删；显式给了 -o 就把结果留给用户。
        if not args.output:
            report.cleanup()


def _cmd_ocr(args) -> int:
    client = _client(args)
    if args.cache:
        client.cache_dir = None if args.cache == "off" else Path(args.cache)
    print("health:", json.dumps(client.health(), ensure_ascii=False))
    result = client.ocr_file(args.path)
    if not result.ok:
        print("失败：", result.error)
        return 1
    print(f"完成：{result.pages} 页 / {len(result.text)} 字符 / {result.elapsed:.1f}s / "
          f"引擎 {result.engine}" + ("（缓存命中）" if result.cached else ""))
    if result.meta:
        print("元信息：", json.dumps(result.meta, ensure_ascii=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(result.text)
        print("正文已写入：", args.output)
    else:
        print("---- 正文前 500 字 ----")
        print(result.text[:500])
    return 0


def _cmd_health(args) -> int:
    client = _client(args)
    info = client.health()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if info.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-table-prepass", description="表格 OCR 的前置修正层")
    sub = parser.add_subparsers(dest="command", required=True)

    fix = sub.add_parser("fix", help="本地：转正歪页 + 擦掉跨页残条")
    fix.add_argument("path")
    fix.add_argument("-o", "--output", default="", help="输出 PDF 路径")
    fix.add_argument("--mode", default="rotated",
                     choices=("rotated", "all", "off"),
                     help="残条修复范围：rotated（默认，只修转正过的页）/ all / off")
    fix.add_argument("--report", default="", help="把 JSON 报告写到这个文件")
    fix.add_argument("--fail-if-unchanged", action="store_true",
                     help="没有改动时返回退出码 1")
    fix.set_defaults(func=_cmd_fix)

    ocr = sub.add_parser("ocr", help="远端：调服务做完整 OCR")
    ocr.add_argument("path")
    ocr.add_argument("-o", "--output", default="", help="把 Markdown 写到这个文件")
    ocr.add_argument("--cache", default="",
                     help="结果缓存目录；off = 关闭（默认就是关）")
    _add_client_args(ocr)
    ocr.set_defaults(func=_cmd_ocr)

    health = sub.add_parser("health", help="远端：看一眼服务状态")
    _add_client_args(health)
    health.set_defaults(func=_cmd_health)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "url", None):
        args.url = env_text("URL")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
