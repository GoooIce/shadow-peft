#!/usr/bin/env python3
"""Audit real development applicability on GB10 CPU; no formal data or budget DB."""

import argparse
import html
import json
import platform
import resource
import socket
import time
from pathlib import Path

from shadow_lab.mage_w4_search.human_transcription import read_json_artifact
from shadow_lab.mage_w4_search.statistical_applicability import load_applicability_report


def render_html(report):
    escape = html.escape
    cost = report["cost"]
    scale = report["scale"]
    rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{:.4f}</td><td>{:,}</td></tr>".format(
            escape(row["stage"]),
            row["attempt_alpha"],
            row["required_empirical_ocr_improvement"],
            row["zero_effect_single_prompt_source_requirement"],
        )
        for row in report["conditional_gate_diagnostics"] or []
    )
    return (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>07 统计适用性 · 开发证据审核</title>
<style>body{max-width:960px;margin:auto;padding:32px;font:16px/1.7 system-ui;color:#243047;background:#f5f7fa}
h1{font-size:28px}section{background:white;padding:22px;margin:18px 0;border:1px solid #dce2eb;border-radius:12px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid #ddd;padding:8px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.scroll{overflow-x:auto}.badge{color:#963c13;font-weight:bold}</style>
<h1>07 统计适用性 · 开发证据审核</h1>
<p class="badge">正式证据不足 · 未开启确认或终检</p>
<section><h2>已得到的证据</h2><p>固定尺度：SCALE。开发候选初筛：SCREEN。
已核验完整批次的唯一占用：COST。</p><p>这是组件阶段审核，不宣称模型质量已被独立确认。</p></section>
<section><h2>当前方法的前提</h2><p>OCR_BOUND
CLIP / HPS 尚缺实际浮点路径的支持界证明；观测最小值和最大值不用于缩窄它。
来源编号互异、排序选样、固定种子都不能单独证明独立抽样。联合正式裁决仍被阻塞。</p></section>
<section><h2>条件性 OCR 门槛</h2><p>下表假设来源簇独立，仅解释现有界的保守程度。
“所需实测改善”是跨过区间门需要的点估计，超过 1 表示当前文字样本构成连最大可能改善也无法通过。
它不是统计功效，也不是推荐的正式样本量。零改善所需源数另假设每源一个文字提示词。</p>
<div class="scroll"><table><thead><tr><th>阶段情景</th><th>单次 α</th><th>所需实测 OCR 改善</th><th>零改善达 1pp 非劣的源数</th></tr></thead><tbody>ROWS</tbody></table></div></section>
<section><h2>仍需补齐</h2><p>实际归一化及点积的数值界；目标总体、独立来源抽样与来源追溯；
可靠的固定尺度和完整配对开发结果；正式尝试次数、样本构成、成本与授权。
没有候选初筛时，效应与配对方差保留为缺失；没有授权时不生成正式协议。</p></section>
<details><summary>完整身份、依据文件、条件与原始技术结果</summary><pre>JSON</pre></details></html>""".replace(
            "SCALE", escape(scale["status"])
        )
        .replace(
            "SCREEN", "尚未完成" if report["screen"] is None else escape(report["screen"]["status"])
        )
        .replace(
            "COST",
            "尚无可核验完整批次成本"
            if cost is None
            else f"{cost['unique_occupied_seconds']:.3f} 秒",
        )
        .replace(
            "OCR_BOUND",
            "已审核的 OCR 字符正确度实现在成功且适用时有 [0,1] 支持界；实际请求的源码身份绑定见原始结果。"
            if report["support"]["axes"]["ocr"]["bound"] == [0, 1]
            else "OCR 源码不再对应已审核实现，当前支持界需要重新核对。",
        )
        .replace("ROWS", rows)
        .replace("JSON", escape(json.dumps(report, ensure_ascii=False, indent=2)))
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.monotonic(), time.process_time()
    summary = {
        "schema": "mage-w4-statistical-applicability-cli/v1",
        "started_at": time.time(),
        "hostname": socket.gethostname(),
        "status": "failed",
        "formal_execution_authorized": False,
        "accounting": "current host process; no new lease; overlap with existing reservations is charged once",
    }
    try:
        inputs, sha, size = read_json_artifact(args.inputs)
        summary.update(input_file_sha256=sha, input_size_bytes=size)
        report = load_applicability_report(inputs)
        (args.output / "report.json").write_text(
            json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        )
        (args.output / "report.html").write_text(render_html(report))
        summary.update(status=report["status"], report_sha256=report["sha256"])
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary.update(
            finished_at=time.time(),
            wall_seconds=time.monotonic() - wall,
            host_cpu_seconds=time.process_time() - cpu,
            max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            max_rss_unit="bytes" if platform.system() == "Darwin" else "KiB",
        )
        (args.output / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            + "\n"
        )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
