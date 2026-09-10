#!/usr/bin/env python3
"""Ticket-14 pilot runner: the end-to-end NVFP4 W4A4 self-improvement loop.

One generation, in the order frozen by tickets 09/11/13/14:

1. propose a mirror-pair population (rank ``+/-1`` sparse edits, one of the 13
   layer groups per generation, deterministic seeds);
2. per candidate: patch the host-resident kernel-format tensors, KF refit
   (``is_checkpoint_format=False``), flush prefix cache, score the dev pool
   (TF accuracy + ``gold_prob``);
3. aggregate in the realised-displacement space (rank shape reward, mirror
   pairs dropped as a whole when either side breaches the gate);
4. centre step ``u = alpha*g + gamma*e``, project onto the E2M1 grid, book-keep
   the residual ``e = u - delta_realised`` (clamp +/-1.5, gamma 1.0, FP32);
5. per-generation guards: confirm open for the centre, engine-dump vs offline
   export 1008/1008 hard gate, KL monitor.

Every step is written to ``<out>/records``; ``<out>/state.json`` is the
resumable centre (sparse code delta + residual).  ``--replay-records`` re-derives
the populations from the records alone.

Usage (gooo, ``~/venvs/minicpm5-vllm``):
    python pilot_search.py --model ~/minicpm5-payload/out/MiniCPM5-2B-NVFP4-W4A4 \
        --taskset artifacts/taskset_dev.jsonl --out out/pilot \
        --generations 4 --candidates 12 --edits 16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import pilot_lib as pl  # noqa: E402
import pilot_worker as pw  # noqa: E402
import retention_lib as rl  # noqa: E402

IM_END = 130073
PROMPT_VARIANT_DEFAULT = "frozen"


def log(*parts) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def load_items(path: Path, limit: int = 0) -> list[dict]:
    items = [json.loads(line) for line in Path(path).open() if line.strip()]
    return items[:limit] if limit else items


def attach_episodes(rows: list[dict], items: list[dict]) -> list[dict]:
    episode = {item["id"]: item["episode"] for item in items}
    for row in rows:
        row["episode"] = episode.get(row["id"], row["id"])
    return rows


def engine_edits(
    target: dict[tuple[str, int], int],
    live: dict[tuple[str, int], int],
    base_of,
    shapes: dict[str, tuple[int, int]],
    layout: dict[str, tuple[str, int]],
    raw_of=None,
) -> dict[str, list[list[int]]]:
    """Diff two sparse code maps and express the change in engine coordinates."""
    edits: dict[str, list[list[int]]] = {}
    for coordinate, code in target.items():
        if live.get(coordinate) == code:
            continue
        layer, flat = coordinate
        module, offset = layout[layer]
        in_features = shapes[layer][1]
        row, col = divmod(flat, in_features)
        edits.setdefault(module, []).append([(offset + row) * in_features + col, int(code)])
    for coordinate in live:
        if coordinate in target:
            continue
        layer, flat = coordinate
        base = int(raw_of(layer, flat)) if raw_of else int(base_of(layer).reshape(-1)[flat])
        module, offset = layout[layer]
        in_features = shapes[layer][1]
        row, col = divmod(flat, in_features)
        edits.setdefault(module, []).append([(offset + row) * in_features + col, base])
    return edits


def center_delta(center: pl.CentreState) -> dict[tuple[str, int], int]:
    return {
        (layer, flat): code
        for layer, bucket in center.codes.items()
        for flat, code in bucket.items()
    }


def swap_variant(variant: str) -> str:
    """The other half of the harness swap (frozen <-> plain)."""
    from taskset_lib import PROMPT_VARIANTS

    others = [name for name in PROMPT_VARIANTS if name != variant]
    return others[0] if others else variant


def _prompt_variants() -> tuple:
    from taskset_lib import PROMPT_VARIANTS

    return PROMPT_VARIANTS


def terminal_variants(args) -> list[str]:
    """Prompt forms to measure on the sealed pool at the terminal step.

    The headline (`--prompt-variant`, default the frozen form) always runs; the
    swap form runs too unless `--no-terminal-swap`, because §3's pre-registered
    rule ("a positive claim requires the gain not to vanish under the swap") is
    only evaluable if both forms were measured on the same sealed pool in the
    same process.
    """
    headline = getattr(args, "prompt_variant", PROMPT_VARIANT_DEFAULT)
    if not getattr(args, "terminal_swap", False) or not getattr(args, "confirm", None):
        return [headline]
    return [headline, swap_variant(headline)]


class Runner:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out).expanduser()
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "records").mkdir(exist_ok=True)
        self.reader = pl.PayloadReader(Path(args.model).expanduser())
        if self.reader.kind == "int8":
            pl.configure_grid("int8-uniform")
        self.layers = (
            [
                key[: -len(".weight")]
                for key in sorted(self.reader._keys)
                if key.endswith(".weight")
                and f"{key[: -len('.weight')]}.weight_scale" in self.reader._keys
            ]
            if self.reader.kind == "int8"
            else [
                key[: -len(".weight_packed")]
                for key in sorted(self.reader._keys)
                if key.endswith(".weight_packed")
            ]
        )
        self.groups = pl.build_groups(self.layers)
        self.masks, self.census = pl.group_census(self.groups, self.reader)
        self.shapes = {layer: self.reader.shape(layer) for layer in self.layers}
        # row offsets for the fused modules (ticket 18: layer counts were wrong)
        self.layout = pl.engine_layout(self.layers, self.shapes)
        self.base_codes: dict[str, np.ndarray] = {}
        self.base_hashes: dict[str, str] = {}
        self.edits_plan = json.loads(Path(args.edits_plan).read_text()) if args.edits_plan else {}
        self.timings: list[dict] = []
        self.seen_coordinates: set = set()
        # 19 票：q0 的"未作答率"基线（启动时实测一次，见 q0_reference()）
        self.q0_reference: dict | None = None
        # 第 2 轮（retention）：反馈噪声带的证据与轨迹读数
        self.noise_replicates: list[dict] = []        # 启动时空转重测的 score_block
        self.slice_noise_replicates: list[dict] = []  # 同上，针对 feedback slice
        self.centre_readings: list[dict] = []         # {"sha", "block", "tag"}，代内重复读数
        self.slice_reading: dict | None = None        # 上一代中心的 slice 读数（slice 门用）
        self.feedback_noise: dict | None = None
        self.slice_noise: dict | None = None
        self.retention_flags: dict | None = None
        self.feedback_slice_reqs: list[dict] | None = None

    # -- base code access --------------------------------------------------- #
    def _base(self, layer: str) -> np.ndarray:
        if layer not in self.base_codes:
            self.base_codes[layer] = self.reader.codes(layer)
        return self.base_codes[layer]

    def _base_hash(self, layer: str) -> str:
        import hashlib

        if layer not in self.base_hashes:
            if self.reader.kind == "int8":
                weight = self.reader._handle.get_tensor(f"{layer}.weight")
                self.base_hashes[layer] = hashlib.sha256(
                    weight.contiguous().numpy().tobytes()
                ).hexdigest()[:24]
            else:
                packed = self.reader._handle.get_tensor(f"{layer}.weight_packed")
                self.base_hashes[layer] = hashlib.sha256(
                    packed.reshape(-1).numpy().tobytes()
                ).hexdigest()[:24]
        return self.base_hashes[layer]

    def edits_for(self, group_index: int) -> int:
        group_id = self.groups[group_index % len(self.groups)].group_id
        return int(self.edits_plan.get(group_id, self.args.edits))

    # -- engine ------------------------------------------------------------- #
    def start_engine(self):
        from vllm import LLM

        args = self.args
        started = time.perf_counter()
        kwargs = dict(
            model=args.model,
            quantization=args.quantization,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem,
            enable_prefix_caching=False,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        if args.linear_backend:
            kwargs["linear_backend"] = args.linear_backend
        self.llm = LLM(**kwargs)
        self.engine_load_seconds = round(time.perf_counter() - started, 2)
        log(f"engine loaded in {self.engine_load_seconds}s (prefix caching off)")
        self.llm.collective_rpc(pw.search_load, kwargs={
            "path": str(self.kernel_base), "cache_key": "base", "device": "cpu",
        })
        log("kernel-format base loaded into worker cache")

    def ensure_kernel_base(self) -> Path:
        path = Path(self.args.kernel_base).expanduser()
        if path.exists() and not self.args.rebuild_kernel:
            return path
        if self.args.kernel_base_script:
            cmd = [
                sys.executable,
                self.args.kernel_base_script,
                "--payload", str(Path(self.args.model).expanduser()),
                "--out", str(path),
                "--manifest", str(path.with_suffix(".manifest.json")),
            ]
            log(f"building kernel-format base: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        return path

    def verify_layout(self) -> dict:
        """Static guard: the fused row offsets must tile each engine weight.

        Ticket 18's residual hard-gate failure came from an offset *unit* bug
        (layer counts read as row counts), which no readback could see because
        the readback used the same formula.  This check compares the layout
        against the kernel file's own shapes, so the class cannot recur
        silently: it raises before the engine is started.
        """
        manifest_path = Path(self.args.kernel_base).expanduser().with_suffix(".manifest.json")
        rows_by_module = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            rows_by_module = {
                row["engine_layer"]: int(row["weight_shape"][0]) for row in manifest["rows"]
            }
        grouped: dict[str, list[tuple[int, str]]] = {}
        for layer, (module, offset) in self.layout.items():
            grouped.setdefault(module, []).append((offset, layer))
        report = {"modules": 0, "fused_modules": 0, "ok": True, "errors": [],
                  "manifest": str(manifest_path)}
        for module, entries in sorted(grouped.items()):
            entries.sort()
            cursor = 0
            for offset, layer in entries:
                if offset != cursor:
                    report["ok"] = False
                    report["errors"].append(
                        f"{module}: {layer} offset {offset} != expected row {cursor}")
                cursor = offset + int(self.shapes[layer][0])
            report["modules"] += 1
            if len(entries) > 1:
                report["fused_modules"] += 1
            if rows_by_module and module in rows_by_module and rows_by_module[module] != cursor:
                report["ok"] = False
                report["errors"].append(
                    f"{module}: layout rows {cursor} != kernel weight rows "
                    f"{rows_by_module[module]}")
        if not report["ok"]:
            raise RuntimeError(f"engine layout check failed: {report['errors'][:4]}")
        log(f"layout check: {report['modules']} modules "
            f"({report['fused_modules']} fused) tile their kernel weights")
        return report

    def patch(self, live: dict, target: dict) -> dict:
        edits = engine_edits(target, live, self._base, self.shapes, self.layout,
                             raw_of=self.reader.raw_code)
        if not edits:
            return {"modules": 0, "bytes_written": 0, "seconds": 0.0}
        result = self.llm.collective_rpc(
            pw.search_patch, kwargs={"cache_key": "base", "edits": edits, "verify": True}
        )[0]
        expected = {}
        for module, items in edits.items():
            expected[module] = [int(code) for _flat, code in items]
        readback = result.pop("readback", {})
        for module, wanted in expected.items():
            got = readback.get(module)
            if got != wanted:
                raise RuntimeError(f"patch readback mismatch on {module}: {got} != {wanted}")
        reload_result = self.llm.collective_rpc(
            pw.search_reload, kwargs={"cache_key": "base"}
        )[0]
        result["refit_seconds"] = round(reload_result["seconds"], 4)
        self.reset_prefix_cache()
        live.clear()
        live.update(target)
        return result

    def verify_live(self, live: dict, tag: str, extra=()) -> dict:
        """Read every live coordinate back from the engine and compare.

        This is the leak detector for the byte-level hard gate: the patch path
        only verifies the coordinates it just wrote, so a coordinate that the
        worker cache silently keeps (or an unintended write) stays invisible
        until the kernel dump.  Cheap: a few hundred coordinates.
        """
        live = dict(live)
        for coordinate in extra:
            layer, flat = coordinate
            live.setdefault(coordinate, int(self._base(layer).reshape(-1)[flat]))
        if not live:
            return {"coordinates": 0}
        queries: dict[str, list[int]] = {}
        for (layer, flat) in live:
            module, offset = self.layout[layer]
            in_features = self.shapes[layer][1]
            row, col = divmod(flat, in_features)
            queries.setdefault(module, []).append((offset + row) * in_features + col)
        results = {}
        for source in ("cache", "engine"):
            results[source] = self.llm.collective_rpc(
                pw.search_readback,
                kwargs={"cache_key": "base", "queries": queries, "source": source},
            )[0]
        mismatches = []          # value calibre: the decoded weight differs
        raw_mismatches = []      # code calibre: kept apart (ticket 18)
        engine_cache_raw = 0
        zero8 = 0
        for (layer, flat), code in live.items():
            module, offset = self.layout[layer]
            in_features = self.shapes[layer][1]
            row, col = divmod(flat, in_features)
            position = (offset + row) * in_features + col
            position_index = queries[module].index(position)
            actual = results["engine"][module][position_index]
            cached = results["cache"][module][position_index]
            expected = int(code)
            if actual != cached:
                # the worker cache and the live parameter must be the same
                # bytes: a difference here is its own finding, not a 0<->8
                # question, so it is counted separately from both calibres.
                engine_cache_raw += 1
            expected_value = pl.value_of(expected)
            if (pl.value_of(actual) != expected_value
                    or pl.value_of(cached) != expected_value):
                mismatches.append({"layer": layer, "flat": int(flat),
                                   "expected": expected, "engine": int(actual),
                                   "cache": int(cached)})
            elif actual != expected or cached != expected:
                zero8 += 1
                raw_mismatches.append({"layer": layer, "flat": int(flat),
                                       "expected": expected, "engine": int(actual),
                                       "cache": int(cached), "class": "zero8"})
        summary = {"coordinates": len(live),
                   "value_mismatches": len(mismatches),
                   "raw_mismatches": len(raw_mismatches),
                   "zero8_representation": zero8,
                   "engine_cache_raw_mismatches": engine_cache_raw,
                   "pass_value": not mismatches,
                   "pass_engine_cache": not engine_cache_raw,
                   "tag": tag,
                   "examples": mismatches[:8],
                   "raw_examples": raw_mismatches[:8]}
        if mismatches:
            log(f"LIVE-READBACK VALUE MISMATCH ({tag}): {len(mismatches)}/{len(live)} "
                f"coordinates differ by value; first: {mismatches[:3]}")
        else:
            log(f"live readback ({tag}): value-clean over {len(live)} coordinates "
                f"({zero8} in the numerically empty 0<->8 channel, "
                f"{engine_cache_raw} engine-vs-cache code diffs)")
        return summary

    def reset_prefix_cache(self) -> bool:
        try:
            return bool(self.llm.reset_prefix_cache())
        except Exception as error:  # pragma: no cover - engine-side guard
            log(f"reset_prefix_cache failed: {error}")
            return False

    # -- scoring ------------------------------------------------------------ #
    def build_reqs(self, items: list[dict], variant: str, strict: bool = True) -> list[dict]:
        """题集 -> 请求（按 prompt 变体渲染）。

        `frozen` 变体与冻结路径逐字一致（`render_prompt` + `answer_span`），不稳定前缀
        照旧 `raise`。swap 臂（`plain`）允许不稳定：`answer_span` 已把 prompt 截到公共
        前缀，剩余 gold 仍是合法的续写；不稳定条数在记录里报数（`unstable_prefix`）。
        """
        from taskset_lib import render_span

        reqs, unstable = [], 0
        for item in items:
            prompt, p_ids, g_ids, stable = render_span(self.tokenizer, item, variant=variant)
            unstable += int(not stable)
            reqs.append({"item": item, "prompt": prompt, "p_ids": p_ids, "g_ids": g_ids,
                         "stable": bool(stable), "variant": variant})
        if unstable and strict:
            raise RuntimeError(f"unstable prompt prefix on variant {variant}: "
                               f"{unstable}/{len(reqs)} items")
        return reqs

    def prepare_requests(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        self.items = load_items(Path(self.args.taskset), self.args.limit)
        # 反馈通道（dev 池）永远是冻结形态：它是适应度与 retention 判据的通道，
        # 换 prompt 形态的是终端 swap 臂，不是这里。
        self.dev_reqs = self.build_reqs(self.items, PROMPT_VARIANT_DEFAULT, strict=True)
        log(f"dev items={len(self.dev_reqs)} unstable_prefix=0 "
            f"(variant={PROMPT_VARIANT_DEFAULT})")
        self.confirm_reqs = None
        self.confirm_reqs_by_variant: dict[str, list[dict]] = {}
        if self.args.confirm:
            c_items = load_items(Path(self.args.confirm))
            self.confirm_reqs = self.build_reqs(c_items, PROMPT_VARIANT_DEFAULT, strict=True)
            self.confirm_reqs_by_variant[PROMPT_VARIANT_DEFAULT] = self.confirm_reqs
            log(f"confirm items={len(self.confirm_reqs)} (variant={PROMPT_VARIANT_DEFAULT})")
            if self.args.terminal_swap:
                for variant in terminal_variants(self.args):
                    reqs = self.build_reqs(c_items, variant, strict=False)
                    self.confirm_reqs_by_variant[variant] = reqs
                    unstable = sum(1 for r in reqs if not r["stable"])
                    log(f"confirm swap arm: variant={variant} unstable_prefix={unstable}"
                        f"/{len(reqs)}")
        self.feedback_slice_reqs = None
        if self.args.feedback_slice:
            s_items = load_items(Path(self.args.feedback_slice))
            self.feedback_slice_reqs = self.build_reqs(s_items, PROMPT_VARIANT_DEFAULT, strict=True)
            log(f"feedback slice items={len(self.feedback_slice_reqs)} "
                f"(variant={PROMPT_VARIANT_DEFAULT}, monitoring channel)")

    def score(self, reqs: list[dict], tag: str) -> tuple[list[dict], dict]:
        from vllm import SamplingParams

        from score_taskset import score_tf

        started = time.perf_counter()
        rows, timing = score_tf(self.llm, SamplingParams, self.tokenizer, reqs, self.args.chunk, tag)
        attach_episodes(rows, [r["item"] for r in reqs])
        timing["wall_seconds"] = round(time.perf_counter() - started, 3)
        return rows, timing

    # -- 反馈噪声带与轨迹读数（第 2 轮 §4a）----------------------------------- #
    def measure_feedback_noise(self, live: dict, state: pl.CentreState) -> dict | None:
        """启动时空转重测：把**同一个中心**在完整管道上测 k 遍。

        每一遍都走真实部署路径（patch -> KF refit -> flush prefix cache -> 评分），
        所以这是 HarnessDev 口径的"同 commit 重评估噪声"在本运行、本精度下的实测值，
        而不是记忆里的 ±4.75。k < 2 时不做，噪声带退回到量化下限（1/n）并如实标注。

        只在 `--retention` / `--feedback-noise` / `--feedback-slice` 打开时执行：默认路径
        一个字节都不加。
        """
        k = int(self.args.noop_rescores)
        want_slice = self.feedback_slice_reqs is not None and bool(self.args.retention or
                                                                  self.args.feedback_noise)
        if k < 2:
            log(f"no-op re-scores disabled (--noop-rescores {k}): the feedback band falls back "
                f"to the quantisation floor")
            return None
        target = center_delta(state)
        blocks, slice_blocks = [], []
        for index in range(k):
            patch = self.patch(live, target)          # 真实往返：同一个中心，重新 refit
            rows, timing = self.score(self.dev_reqs, f"noop{index}")
            block = rl.score_block(rows)
            blocks.append(block)
            pl.json_dump(self.out / "records" / f"noop_rescore_{index:02d}.json",
                         {"index": index, "kind": "noop_rescore", "centre_sha256": None,
                          "block": block, "patch": patch, "timing": timing, "rows": rows})
            if want_slice:
                s_rows, s_timing = self.score(self.feedback_slice_reqs, f"noop{index}-slice")
                slice_blocks.append(rl.score_block(s_rows))
                pl.json_dump(self.out / "records" / f"noop_rescore_slice_{index:02d}.json",
                             {"index": index, "kind": "noop_rescore_slice", "block": slice_blocks[-1],
                              "timing": s_timing, "rows": s_rows})
            log(f"no-op re-score {index + 1}/{k}: centre acc={block['overall']['acc']:.4f} "
                f"({timing['wall_seconds']}s)")
        self.noise_replicates = blocks
        self.slice_noise_replicates = slice_blocks
        self.feedback_noise = rl.noise_band(blocks, self.dev_item_rows(),
                                            scale=self.args.retention_band_scale,
                                            term=self.args.retention_band_term)
        if slice_blocks:
            self.slice_noise = rl.noise_band(slice_blocks, self.slice_item_rows(),
                                             scale=self.args.retention_band_scale,
                                             term=self.args.retention_band_term)
        self.retention_flags = {
            "retention": bool(self.args.retention),
            "retention_band_scale": self.args.retention_band_scale,
            "retention_band_term": self.args.retention_band_term,
            "noop_rescores": k,
            "feedback_slice": self.args.feedback_slice,
            "feedback_slice_gate": bool(self.args.feedback_slice_gate),
            "band_rule": rl.FEEDBACK_BAND_NOTE,
        }
        pl.json_dump(self.out / "feedback_noise.json", {
            "protocol": rl.PROTOCOL,
            "feedback_channel": {"taskset": self.args.taskset, "band": self.feedback_noise},
            "feedback_slice": ({"taskset": self.args.feedback_slice, "band": self.slice_noise}
                               if slice_blocks else None),
            "flags": self.retention_flags,
        })
        log("feedback noise band (dev): centre band="
            f"{self.feedback_noise['centre']['band']:.5f} "
            f"(harness_se={self.feedback_noise['centre']['harness_se']}, "
            f"floor={self.feedback_noise['centre']['quantisation_floor']:.5f}, "
            f"replicates={self.feedback_noise['replicates']}, "
            f"deterministic={self.feedback_noise['harness']['deterministic']})")
        log("feedback noise band per family: "
            + str({k2: round(v["band"], 5) for k2, v in self.feedback_noise["per_family"].items()}))
        return self.feedback_noise

    def dev_item_rows(self) -> list[dict]:
        """The dev pool's *items* as rows (no scoring): only the sampling term needs them."""
        return [{"id": r["item"]["id"], "family": r["item"]["family"],
                 "episode": r["item"].get("episode", r["item"]["id"])} for r in self.dev_reqs]

    def slice_item_rows(self) -> list[dict]:
        if not self.feedback_slice_reqs:
            return []
        return [{"id": r["item"]["id"], "family": r["item"]["family"],
                 "episode": r["item"].get("episode", r["item"]["id"])}
                for r in self.feedback_slice_reqs]

    def observe_centre_reading(self, sha: str, rows: list[dict], tag: str) -> None:
        """记一次"同一个中心的又一次读数"（代内重复：步前读数 vs 上代步后读数）。

        §4a 允许噪声带来两类证据：启动时空转重测，或**跨代的重复读数**。这里累积后者；
        :meth:`current_feedback_noise` 会在两者都可用时按 writeup 里的确定性规则择一。
        """
        if not (self.args.retention or self.args.feedback_noise):
            return
        self.centre_readings.append({"sha": sha, "block": rl.score_block(rows), "tag": tag})

    def current_feedback_noise(self) -> dict:
        """当前可用的反馈噪声带（启动重测优先，其次跨代重复读数）。"""
        replicates, source = rl.band_replicates(self.noise_replicates, self.centre_readings)
        band = rl.noise_band(replicates, self.dev_item_rows(),
                             scale=self.args.retention_band_scale, source=source,
                             term=self.args.retention_band_term)
        return band

    def current_slice_noise(self) -> dict | None:
        if self.slice_noise:
            return self.slice_noise
        if not self.feedback_slice_reqs:
            return None
        return rl.noise_band([], self.slice_item_rows(), scale=self.args.retention_band_scale,
                             source="none", term=self.args.retention_band_term)

    # -- 未作答健康度（19 票；只记录，不改判据）--------------------------------- #
    def no_answer(self, rows: list[dict]) -> dict:
        """逐 family / 逐 stratum 的未作答率（`score_taskset.no_answer_summary`）。"""
        from score_taskset import no_answer_summary

        return no_answer_summary(rows)

    def attribution(self, rows: list[dict]) -> dict | None:
        """对 q0 基线的机械归因（全题增益 vs 两臂都作答子集的增益 + 未作答率超出量）。"""
        if not self.q0_reference:
            return None
        from score_taskset import attribution_verdict

        return attribution_verdict(rows, self.q0_reference["rows"],
                                   margin=self.args.no_answer_margin)

    def q0_reference_measure(self, live: dict) -> dict:
        """启动时实测一次 q0（无编辑的起点）在**同一 dev 池**上的分数与未作答率。

        之后每一代都与这份本次运行实测的基线比较，而不是与记忆中的常数比较。测量期间把引擎
        恢复到 q0、测完立刻恢复调用前的中心补丁（并断言 live 一致）。行集落盘到
        ``records/q0_reference.json``，归因规则与事后复核都用它。
        """
        before = dict(live)
        self.patch(live, {})                      # 空目标 = 把 live 的坐标写回基码
        rows, timing = self.score(self.dev_reqs, "q0-reference")
        self.patch(live, before)                  # 恢复调用前的中心
        if live != before:
            raise RuntimeError("q0 reference measurement did not restore the centre patch")
        payload = {
            "kind": "q0_reference",
            "taskset": self.args.taskset,
            "fitness": pl.fitness_summary(rows),
            "no_answer": self.no_answer(rows),
            "timing": timing,
            "rows": rows,
        }
        pl.json_dump(self.out / "records" / "q0_reference.json", payload)
        self.q0_reference = payload
        na = payload["no_answer"]["overall"]
        log(f"q0 reference dev: acc={payload['fitness']['acc']:.4f} "
            f"no_answer={na['rate']:.4f} ({na['n_no_answer']}/{na['n']})")
        log("q0 reference per-family no-answer: "
            + str({k: round(v["rate"], 4) for k, v in payload["no_answer"]["per_family"].items()}))
        return payload

    def no_answer_gate(self, centre_na: dict, centre_attr: dict | None,
                       candidate_records: list[dict]) -> dict:
        """把"未作答率是否被推动"做成机械判定，写进当代表。

        规则（`--no-answer-margin`，默认 0.05）：
          * 逐 arm 报整体与逐 family 的未作答率相对 q0 基线的超出量；
          * 只要 **整体超出 > margin**、或 **任一 family 超出 > margin**、或归因判定为
            `artifact_dominated`，该代即 `flagged`。
        """
        margin = self.args.no_answer_margin
        ref = (self.q0_reference["no_answer"] if self.q0_reference else None)

        def excess(na: dict) -> dict:
            if not ref:
                return {}
            out = {"overall": na["overall"]["rate"] - ref["overall"]["rate"]}
            for key, block in na["per_family"].items():
                if key in ref["per_family"]:
                    out[key] = block["rate"] - ref["per_family"][key]["rate"]
            return out

        rates = [c["no_answer"]["overall"]["rate"] for c in candidate_records]
        best = max(candidate_records, key=lambda c: (c["fitness"]["acc"] or 0.0)) if candidate_records else None
        # 每一块都带**完整的** no_answer 汇总（overall + per_family + per_stratum），
        # 这样 excess() 能逐 family 比较，history 行里读 ["no_answer"]["overall"]["rate"]。
        blocks = {"centre": {"no_answer": centre_na, "attribution": centre_attr}}
        if best is not None:
            blocks["best_candidate"] = {"index": best["index"],
                                        "no_answer": best["no_answer"],
                                        "attribution": best["attribution"]}
        flags = []
        for name, block in blocks.items():
            if not block:
                continue
            exc = excess(block["no_answer"])
            block["excess_vs_q0"] = exc
            trip = [k for k, v in exc.items() if v is not None and v > margin]
            if trip:
                flags.append({"arm": name, "kind": "no_answer_excess", "buckets": trip,
                              "worst": max(exc[k] for k in trip)})
            attr = block.get("attribution")
            if attr and attr.get("verdict") == "artifact_dominated":
                flags.append({"arm": name, "kind": "artifact_dominated",
                              "gain_all": attr.get("gain_all"), "gain_both": attr.get("gain_both"),
                              "no_answer_excess": attr.get("no_answer_excess")})
        return {
            "metric": ("no_answer = argmax at the first gold-token position == 130073 (<|im_end|>); "
                       "flag when arm rate exceeds the q0 baseline by more than --no-answer-margin "
                       "overall or in any family, or when a claimed gain does not survive on the "
                       "both-answered subset"),
            "margin": margin,
            "q0_reference": ref,
            "centre": blocks.get("centre"),
            "best_candidate": blocks.get("best_candidate"),
            "candidates_no_answer": ({"n": len(rates), "min": min(rates),
                                      "median": sorted(rates)[len(rates) // 2],
                                      "max": max(rates)} if rates else None),
            "flagged": flags,
            "flagged_arms": sorted({f["arm"] for f in flags}),
        }

    # -- probes ------------------------------------------------------------- #
    def kl_probe(self) -> dict | None:
        if not (self.args.probe_prompts and self.args.bf16_logprobs):
            return None
        from vllm import SamplingParams

        prompts = json.loads(Path(self.args.probe_prompts).read_text())
        reference = json.loads(Path(self.args.bf16_logprobs).read_text())
        requests = []
        spans = []
        for index, prompt_ids in enumerate(prompts["decode_prompts"]):
            continuation = reference["continuations"][str(index)]
            sequence = list(prompt_ids) + list(continuation)
            requests.append({"prompt_token_ids": sequence})
            spans.append((len(prompt_ids), len(sequence)))
        sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0, detokenize=False)
        started = time.perf_counter()
        outputs = self.llm.generate(requests, sampling, use_tqdm=False)
        deltas = []
        for index, output in enumerate(outputs):
            start, end = spans[index]
            for position in range(start, end):
                entry = output.prompt_logprobs[position]
                token_id = output.prompt_token_ids[position]
                if not entry or token_id not in entry:
                    continue
                ours = float(entry[token_id].logprob)
                theirs = reference["logprobs"][str(index)][position - start]
                deltas.append(abs(ours - theirs))
        return {
            "tokens": len(deltas),
            "mean_abs_delta": float(np.mean(deltas)) if deltas else None,
            "generation_kl_error": float(np.mean(np.exp(deltas))) if deltas else None,
            "seconds": round(time.perf_counter() - started, 3),
        }

    def hard_gate(self, center: pl.CentreState, generation: int) -> dict:
        """Two calibres on the same file pair (ticket 18's redefinition).

        * ``raw``   -- code positions, the pre-18 gate (kept for the record);
        * ``value`` -- decoded E2M1 values, **the frozen gate**: the pass
          criterion, plus the ``0 <-> 8`` representation count reported apart.
        """
        args = self.args
        if args.skip_gate:
            return {"skipped": True}
        started = time.perf_counter()
        dump_path = self.out / f"kernel_dump_gen{generation:02d}.safetensors"
        export_path = self.out / f"kernel_export_gen{generation:02d}.safetensors"
        compare_path = self.out / f"kernel_compare_gen{generation:02d}.json"
        values_path = self.out / f"kernel_values_gen{generation:02d}.json"
        dump = self.llm.collective_rpc(pw.search_dump, kwargs={"out_path": str(dump_path)})[0]
        self.export_center(center, export_path)
        subprocess.run([
            sys.executable, args.compare_script,
            "--left", str(dump_path), "--right", str(export_path), "--out", str(compare_path),
        ], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([
            sys.executable, args.compare_values_script,
            "--left", str(dump_path), "--right", str(export_path), "--out", str(values_path),
        ], check=True, stdout=subprocess.DEVNULL)
        report = json.loads(compare_path.read_text())
        values = json.loads(values_path.read_text())
        report["dump"] = dump
        report["seconds"] = round(time.perf_counter() - started, 2)
        report["dump_path"] = str(dump_path)
        report["export_path"] = str(export_path)
        report["values_path"] = str(values_path)
        report["gate"] = {
            "mode": "value",
            "pass": bool(values["pass_value"]),
            "tensors": values["tensors"],
            "tensors_with_value_diffs": values["tensors_with_value_diffs"],
            "value_diffs": values["value_diffs"],
            "canonical_code_diffs": values["canonical_code_diffs"],
            "raw_code_diffs": values["raw_code_diffs"],
            "zero8_diffs": values["zero8_diffs"],
            "structural_mismatches": values["structural_mismatches"],
            "notes": values["notes"][:20],
        }
        if not args.keep_dumps:
            dump_path.unlink(missing_ok=True)
            export_path.unlink(missing_ok=True)
        log(f"hard gate gen{generation}: {report['tensors']} tensors; raw diffs="
            f"{values['raw_code_diffs']} (0<->8 {values['zero8_diffs']}), value diffs="
            f"{values['value_diffs']} on {values['tensors_with_value_diffs']} tensors; "
            f"pass={report['gate']['pass']}, {report['seconds']}s")
        if args.gate_halt and not report["gate"]["pass"]:
            raise RuntimeError(
                f"hard gate gen{generation} failed on the value calibre: "
                f"{values['value_diffs']} value diffs; see {values_path}"
            )
        return report

    # -- §1 每代保留（定向修订循环，默认关）------------------------------------ #
    def retention_check(self, state: pl.CentreState, generation: int, live: dict,
                        centre_rows: list[dict], centre_sha: str, state_sha: str,
                        ledger: dict, step_snapshot: dict | None, step_coordinates) -> tuple:
        """把 HarnessDev 的"读失败 -> 定向修复 -> 复验"循环落成协议规则。

        步骤：① 在**反馈通道**（dev 池）上重读刚刚应用了中心步的中心；② 相对**上一代
        中心**的读数逐 family 分解；③ 任一家族跌幅超过本次运行自己测出的噪声带
        （§4a，而不是记忆里的常数）→ 按票 09 §4 的残差规则把这一步**整步回滚**：坐标与
        残差逐坐标恢复到位前值（未取的步不得借 `gamma*e` 在下一代返回），中心身份逐位
        复原，并把逐 family 证据写成 `retention_rollback`。

        返回 ``(retention_or_None, post_rows_or_None)``。`--retention` 关闭时完全不进
        这个函数（没有任何额外评分）。
        """
        args = self.args
        post_rows, post_timing = self.score(self.dev_reqs, f"centre-post-gen{generation}")
        post_summary = pl.fitness_summary(post_rows)
        band = self.current_feedback_noise()
        decision = rl.retention_decision(centre_rows, post_rows, band)

        slice_block = None
        slice_gate = None
        slice_rows = None
        if self.feedback_slice_reqs is not None:
            slice_rows, slice_timing = self.score(self.feedback_slice_reqs,
                                                 f"centre-slice-gen{generation}")
            slice_block = rl.score_block(slice_rows)
            if self.slice_reading and self.slice_reading.get("sha") == centre_sha:
                slice_gate = rl.retention_decision(self.slice_reading["rows"], slice_rows,
                                                   self.current_slice_noise())
                slice_gate["channel"] = "feedback_slice"
            elif self.args.feedback_slice_gate:
                slice_gate = {"channel": "feedback_slice", "skipped": True,
                              "reason": ("no comparable slice reading of the previous centre "
                                         "(first generation of this process); the slice gate is "
                                         "not evaluated this generation"),
                              "rollback": False, "breaches": []}

        channels = {"feedback": decision}
        if slice_gate:
            channels["feedback_slice"] = slice_gate
        breached = [name for name, block in channels.items()
                    if block.get("rollback") and (name == "feedback" or args.feedback_slice_gate)]
        rollback = bool(breached)

        retention = {
            "protocol": rl.PROTOCOL,
            "kind": "retention_rollback" if rollback else "retention_keep",
            "rule": rl.RETENTION_RULE_NOTE,
            "generation": generation,
            "rollback": rollback,
            "breached_channels": breached,
            "channels": channels,
            "band": {"source": band.get("source"), "centre": band["centre"],
                     "per_family": {f: band["per_family"][f] for f in decision["per_family"]}},
            "post_fitness": post_summary,
            "post_timing": post_timing,
            "step": {"snapshot_layers": (len(step_snapshot["layers"]) if step_snapshot else 0),
                     "step_coordinates": (len(step_coordinates) if step_coordinates else None),
                     "step_ledger_moved": ledger.get("moved"),
                     "step_ledger_touched": ledger.get("touched")},
        }
        if slice_block is not None:
            retention["feedback_slice"] = {
                "block": slice_block,
                "timing": slice_timing,
                "gate": slice_gate,
                "role": ("wired into the retention decision (--feedback-slice-gate)"
                         if args.feedback_slice_gate else
                         "monitoring only (candidate selection and the acceptance gate are "
                         "unchanged; the slice never enters selection)"),
                "band": (self.current_slice_noise() if slice_gate else None),
            }

        if rollback:
            restore = pl.restore_layers(state, step_snapshot)
            if restore["residual_mismatches"]:
                retention["restore"] = restore
                pl.json_dump(self.out / "records" / f"retention_gen{generation:02d}.json", retention)
                raise RuntimeError(
                    f"retention rollback gen{generation} left {restore['residual_mismatches']} "
                    f"residual mismatches -- the re-anchoring is not faithful; see "
                    f"records/retention_gen{generation:02d}.json")
            reanchor = pl.residual_reanchor(ledger, step_coordinates)
            redeploy = self.patch(live, center_delta(state))
            restored_sha = state.sha256(self._base_hash)
            retention["restore"] = restore
            retention["reanchor"] = reanchor
            retention["redeploy"] = redeploy
            retention["centre_sha256_after_step"] = state_sha
            retention["centre_restored_sha256"] = restored_sha
            retention["sha_restored_exactly"] = bool(restored_sha == centre_sha)
            retention["residual_ledger_effective"] = "reverted (the step was not taken)"
            retention["post_reading_discarded"] = True
            log(f"gen{generation} RETENTION ROLLBACK ({'+'.join(breached)}): "
                f"feedback breaches={[b['family'] for b in decision['breaches']]}"
                + (f" slice breaches={[b['family'] for b in (slice_gate or {}).get('breaches', [])]}"
                   if slice_gate else "")
                + f" (worst {decision['worst_family']} {decision['worst_delta']:+.4f}); "
                f"step reverted, residual re-anchored over {reanchor['coordinates']} "
                f"coordinates, centre sha restored exactly="
                f"{retention['sha_restored_exactly']}")
            pl.json_dump(self.out / "records" / f"retention_gen{generation:02d}.json", retention)
            if not retention["sha_restored_exactly"]:
                raise RuntimeError(
                    f"retention rollback gen{generation} did not restore the centre identity: "
                    f"{restored_sha} != {centre_sha}; see "
                    f"records/retention_gen{generation:02d}.json")
            # slice 的"上一代读数"仍描述同一个中心（rollback 之后 sha 与步前相同），
            # 因此 self.slice_reading 保持不变即可。
            return retention, None

        # no rollback: the post-step reading is a real reading of the new centre
        self.observe_centre_reading(state_sha, post_rows, f"post-gen{generation}")
        if slice_rows is not None:
            self.slice_reading = {"sha": state_sha, "rows": slice_rows, "block": slice_block}
        pl.json_dump(self.out / "records" / f"retention_gen{generation:02d}.json", retention)
        log(f"gen{generation} retention keep: post-step centre acc="
            f"{post_summary['acc']:.4f} (d={decision['centre_delta']:+.4f} vs previous, "
            f"band {decision['centre_band']:.5f}); worst family "
            f"{decision['worst_family']} {decision['worst_delta']:+.4f}")
        return retention, post_rows

    def run_generation(self, state: pl.CentreState, generation: int, live: dict,
                       history: list, centre_sha: str) -> dict:
        args = self.args
        started = time.perf_counter()
        group_index = generation % len(self.groups)
        group = self.groups[group_index]
        edits = self.edits_for(group_index)
        pairs = max(1, args.candidates // 2)
        population = pl.propose_generation(
            self.groups, self.masks, generation, group_index, pairs, edits, args.seed
        )
        log(f"gen{generation}: group={group.group_id} candidates={len(population)} "
            f"edits/candidate={edits}")

        centre_rows, centre_timing = self.score(self.dev_reqs, f"centre-gen{generation}")
        centre_summary = pl.fitness_summary(centre_rows)
        centre_episodes = pl.episode_scores(centre_rows)
        centre_na = self.no_answer(centre_rows)
        centre_attr = self.attribution(centre_rows)
        self.observe_centre_reading(centre_sha, centre_rows, f"pre-gen{generation}")
        log(f"gen{generation} centre dev acc={centre_summary['acc']:.4f} "
            f"gold_prob={centre_summary['gold_prob']:.4f} "
            f"na={centre_na['overall']['rate']:.4f}")

        candidate_records = []
        realized_list = []
        for candidate in population:
            candidate_edits = candidate.edits(state, self.reader)
            realized = pl.realized_displacement(candidate_edits)
            realized_list.append(realized)
            target = dict(center_delta(state))
            for edit in candidate_edits:
                target[(edit["layer"], edit["flat"])] = edit["new_code"]
            patch = self.patch(live, target)
            rows, timing = self.score(self.dev_reqs, f"cand{candidate.index}")
            summary = pl.fitness_summary(rows)
            episodes = pl.episode_scores(rows)
            stats = pl.paired_stats(episodes, centre_episodes)
            candidate_records.append({
                "index": candidate.index,
                "pair": candidate.pair,
                "side": candidate.side,
                "sign": candidate.sign,
                "coordinates": [[layer, flat] for layer, flat in candidate.coordinates],
                "edits": candidate_edits,
                "realized": {f"{layer}|{flat}": value for (layer, flat), value in realized.items()},
                "fitness": summary,
                "episodes": episodes,
                "delta_center": stats["mean"],
                "sd": stats["sd"],
                "no_answer": self.no_answer(rows),
                "attribution": self.attribution(rows),
                "patch": patch,
                "timing": timing,
                "wall_seconds": timing["wall_seconds"],
            })
            pl.json_dump(self.out / "records" / f"cand_gen{generation:02d}_{candidate.index:03d}.json",
                         {"generation": generation, "candidate": candidate_records[-1], "rows": rows})
            log(f"  cand{candidate.index:03d} pair{candidate.pair}{'+' if candidate.sign > 0 else '-'} "
                f"acc={summary['acc']:.4f} gp={summary['gold_prob']:.4f} "
                f"na={candidate_records[-1]['no_answer']['overall']['rate']:.4f} "
                f"d={stats['mean']:+.4f} sd={stats['sd']:.4f} "
                f"refit={patch.get('refit_seconds', 0)*1000:.0f}ms "
                f"score={timing['wall_seconds']:.2f}s")

        t_restore = time.perf_counter()
        self.patch(live, center_delta(state))
        restore_seconds = round(time.perf_counter() - t_restore, 3)

        # --- guards ---------------------------------------------------------
        confirm = None
        if self.confirm_reqs is not None:
            confirm_rows, confirm_timing = self.score(self.confirm_reqs, f"confirm-gen{generation}")
            confirm = {"fitness": pl.fitness_summary(confirm_rows), "timing": confirm_timing}
            pl.json_dump(self.out / "records" / f"confirm_gen{generation:02d}.json", confirm)
            log(f"gen{generation} confirm acc={confirm['fitness']['acc']:.4f} "
                f"({confirm_timing['wall_seconds']}s)")
        kl = self.kl_probe()
        if kl:
            log(f"gen{generation} KL monitor: {kl['generation_kl_error']:.4f} "
                f"over {kl['tokens']} tokens")
        gate = self.hard_gate(state, generation)

        # --- aggregation + centre step --------------------------------------
        pairs_payload = []
        for pair_index in range(pairs):
            plus = candidate_records[2 * pair_index]
            minus = candidate_records[2 * pair_index + 1]
            pairs_payload.append({
                "pair": pair_index,
                "plus": {"index": plus["index"], "fitness": plus["fitness"]["acc"],
                         "delta_center": plus["delta_center"]},
                "minus": {"index": minus["index"], "fitness": minus["fitness"]["acc"],
                          "delta_center": minus["delta_center"]},
            })
        aggregation = pl.aggregate(pairs_payload, realized_list)
        gradient = aggregation.pop("gradient")
        # 第 2 轮 §1：步前快照（只有 --retention 才取；快照覆盖 apply_step 会访问的全部坐标）
        step_snapshot = None
        step_coordinates = None
        if args.retention:
            step_coordinates = pl.step_coordinates(state, gradient)
            step_snapshot = pl.snapshot_layers(state, step_coordinates)
        ledger = pl.apply_step(state, self.reader, gradient, args.alpha)
        state.generation = generation + 1
        deploy = self.patch(live, center_delta(state))
        state_sha = state.sha256(self._base_hash)

        # --- §1 定向修订：读回反馈通道 -> 逐 family 分解 -> 越带则回滚 ----------
        # docstring 契约：--retention 关闭时不进 retention_check（无额外评分、
        # 无回滚判定）；快照也只在开时拍，否则 rollback 会拿着 None 快照崩掉。
        if args.retention:
            retention, centre_post_rows = self.retention_check(
                state, generation, live, centre_rows, centre_sha, state_sha,
                ledger, step_snapshot, step_coordinates,
            )
        else:
            retention, centre_post_rows = None, None
        if retention and retention["rollback"]:
            state_sha = retention["centre_restored_sha256"]
            deploy = retention["redeploy"]
        centre_post = (pl.fitness_summary(centre_post_rows) if centre_post_rows
                       else (retention or {}).get("post_fitness"))

        if args.verify_live:
            for candidate in population:
                self.seen_coordinates.update(candidate.coordinates)
            touched = set(self.seen_coordinates)
            live_check = self.verify_live(live, f"gen{generation}", extra=touched)
        else:
            live_check = None
        resolution = resolution_report(candidate_records)
        no_answer_gate = self.no_answer_gate(centre_na, centre_attr, candidate_records)
        if no_answer_gate["flagged"]:
            log(f"gen{generation} NO-ANSWER FLAG: {no_answer_gate['flagged']}")
        record = {
            "generation": generation,
            "group_id": group.group_id,
            "group_index": group_index,
            "pairs": pairs,
            "edits_per_candidate": edits,
            "population": [
                {"index": c.index, "pair": c.pair, "side": c.side, "sign": c.sign,
                 "coordinates": [[layer, flat] for layer, flat in c.coordinates]}
                for c in population
            ],
            "protocol": pl.PROTOCOL,
            "seed": args.seed,
            "alpha": args.alpha,
            "gamma": pl.GAMMA,
            "clamp": pl.CLAMP,
            "tau": args.tau,
            "center_sha256_before": centre_sha,
            "center_sha256_after": state_sha,
            "center_fitness": centre_summary,
            "centre_episodes": centre_episodes,
            "candidates": [
                {key: value for key, value in cand.items()
                 if key not in ("episodes", "realized", "edits", "patch", "timing")}
                for cand in candidate_records
            ],
            "aggregation": {
                "qualified_pairs": aggregation["qualified_pairs"],
                "dropped_pairs": aggregation["dropped_pairs"],
                "pair_diffs": aggregation["pair_diffs"],
                "pair_rewards": aggregation["pair_rewards"],
                "weight_total": aggregation["weight_total"],
                "gradient_support": len(gradient),
                "gradient_norm": float(np.linalg.norm(list(gradient.values()))) if gradient else 0.0,
            },
            "residual_ledger": ledger,
            "retention": retention,
            "centre_post_fitness": centre_post,
            "resolution": resolution,
            "live_readback": live_check,
            "confirm": confirm,
            "no_answer_gate": no_answer_gate,
            "kl": kl,
            "hard_gate": {key: gate[key] for key in gate if key != "rows"},
            "timings": {
                "centre_score": centre_timing["wall_seconds"],
                "candidate_score_total": round(sum(c["wall_seconds"] for c in candidate_records), 3),
                "candidate_patch_total": round(sum(c["patch"].get("refit_seconds", 0)
                                                    for c in candidate_records), 4),
                "restore_seconds": restore_seconds,
                "deploy_seconds": deploy.get("refit_seconds", 0.0),
            },
            "wall_seconds": round(time.perf_counter() - started, 2),
        }
        pl.json_dump(self.out / "records" / f"gen{generation:02d}.json", record)
        pl.json_dump(self.out / "state.json", state.to_dict())
        with (self.out / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "generation": generation,
                "group_id": group.group_id,
                "center_sha256_before": centre_sha,
                "center_sha256_after": state_sha,
                "center_acc": centre_summary["acc"],
                "center_gold_prob": centre_summary["gold_prob"],
                "moved": ledger["moved"],
                "rejected_beyond_grid": ledger["rejected_beyond_grid"],
                "residual_clamped": ledger["residual_clamped"],
                "qualified_pairs": aggregation["qualified_pairs"],
                "dropped_pairs": aggregation["dropped_pairs"],
                "kl": kl,
                "wall_seconds": record["wall_seconds"],
                # 19 票：未作答健康度 + 机械归因（不影响 fitness / 停止判据）
                "centre_no_answer": centre_na["overall"]["rate"],
                "centre_no_answer_per_family": {k: v["rate"]
                                                for k, v in centre_na["per_family"].items()},
                "centre_attribution": centre_attr,
                "best_candidate_index": no_answer_gate["best_candidate"]["index"]
                if no_answer_gate["best_candidate"] else None,
                "best_candidate_no_answer": no_answer_gate["best_candidate"]["no_answer"]["overall"]["rate"]
                if no_answer_gate["best_candidate"] else None,
                "no_answer_flagged": no_answer_gate["flagged"],
                # 第 2 轮 §1/§4：保留裁决与中心轨迹读数（monitoring 通道）
                "retention_rollback": bool(retention and retention["rollback"]),
                "retention_breaches": ([{"family": b["family"], "delta": b["delta"],
                                         "band": b["band"]} for b in
                                        (retention["channels"]["feedback"]["breaches"]
                                         if retention else [])]),
                "retention_worst_family": (retention["channels"]["feedback"]["worst_family"]
                                           if retention else None),
                "retention_worst_delta": (retention["channels"]["feedback"]["worst_delta"]
                                          if retention else None),
                "retention_band_source": (retention["band"]["source"] if retention else None),
                "retention_centre_band": (retention["channels"]["feedback"]["centre_band"]
                                          if retention else None),
                "centre_post_acc": (centre_post or {}).get("acc"),
                "centre_post_gold_prob": (centre_post or {}).get("gold_prob"),
                "feedback_slice_acc": ((retention or {}).get("feedback_slice", {}) or {})
                .get("block", {}).get("overall", {}).get("acc"),
                "feedback_slice_episodes": ((retention or {}).get("feedback_slice", {}) or {})
                .get("block", {}).get("overall", {}).get("episodes"),
            }, ensure_ascii=False) + "\n")
        history.append(record)
        return record

    def confirm_arms(self, live: dict, state: pl.CentreState, tags=("q0", "center"),
                     variant: str = PROMPT_VARIANT_DEFAULT) -> dict:
        """Score the frozen confirm pool on q0 and on the resumed centre.

        Ticket 18 froze the terminal wording as an episode-level paired
        comparison against q0 on the 900-episode confirm pool, so the decision
        needs **both** arms measured on the same pool in the same process; the
        verdict itself (classify_outcome with the frozen delta, the percentile
        bootstrap and the sign-flip null) is computed offline by
        ``confirm_gate.py --measure`` from the rows dumped here.

        Round 2 (§3): the same two arms are additionally measured under the
        harness-swap prompt form, so the terminal record can state whether a gain
        survives the swap.  Opening the pool once per *process* is what the
        multiplicity accounting counts (``--looks``); measuring extra arms on the
        same items in the same process does not open it again.
        """
        reqs = None
        if variant != PROMPT_VARIANT_DEFAULT:
            reqs = self.confirm_reqs_by_variant.get(variant)
            if reqs is None:
                raise RuntimeError(f"no requests prepared for prompt variant {variant!r}")
        arms = {}
        targets = {"q0": pl.CentreState(), "center": state}
        for name in tags:
            target = targets[name]
            self.patch(live, center_delta(target))
            rows, timing = self.score(reqs or self.confirm_reqs, f"confirm-arms-{name}-{variant}")
            arms[name] = {"fitness": pl.fitness_summary(rows), "timing": timing,
                          "no_answer": self.no_answer(rows), "rows": rows}
            log(f"confirm arm {name} [{variant}]: acc={arms[name]['fitness']['acc']:.4f} "
                f"episodes={arms[name]['fitness']['episodes']} "
                f"na={arms[name]['no_answer']['overall']['rate']:.4f} "
                f"({timing['wall_seconds']}s)")
        # 19 票归因：确认池上"中心 vs q0"的增益是否在"两臂都作答"子集上仍然成立
        if "q0" in arms and "center" in arms:
            from score_taskset import attribution_verdict

            arms["attribution_center_vs_q0"] = attribution_verdict(
                arms["center"]["rows"], arms["q0"]["rows"], margin=self.args.no_answer_margin)
            log(f"confirm attribution centre vs q0 [{variant}]: "
                + json.dumps({k: arms["attribution_center_vs_q0"][k] for k in
                              ("verdict", "gain_all", "gain_both", "no_answer_excess",
                               "n_both_answered")}, ensure_ascii=False))
        arms["variant"] = variant
        arms["unstable_prefix"] = sum(1 for r in (reqs or self.confirm_reqs)
                                      if not r.get("stable", True))
        return arms

    def confirm_arms_by_variant(self, live: dict, state: pl.CentreState) -> dict:
        """§3: every arm (q0 and centre) under every terminal prompt form."""
        variants = terminal_variants(self.args)
        arms = {}
        for variant in variants:
            arms[variant] = self.confirm_arms(live, state, variant=variant)
        return arms

    def export_terminal_payload(self, state: pl.CentreState, deliverable: str) -> dict:
        """§2: export the payload the terminal rule actually delivers.

        ``deliverable == "center"`` exports the resumed centre; ``"q0"`` exports
        the **unpatched** q0 payload (an empty centre), so "the deliverable is q0"
        is a real artifact and not just a sentence in a record.
        """
        center = state if deliverable == "center" else pl.CentreState()
        tag = "terminal_center" if deliverable == "center" else "terminal_q0"
        started = time.perf_counter()
        report = self.materialise_and_export(center, tag)
        payload_dir = Path(report["payload_dir"])
        digest = None
        try:
            import hashlib

            handle = hashlib.sha256()
            with (payload_dir / "model.safetensors").open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    handle.update(chunk)
            digest = handle.hexdigest()
        except OSError as error:  # pragma: no cover - disk-side guard
            log(f"payload sha256 failed: {error}")
        out = {
            "deliverable": deliverable,
            "payload_dir": str(payload_dir),
            "model_safetensors_sha256": digest,
            "model_safetensors_bytes": (payload_dir / "model.safetensors").stat().st_size
            if (payload_dir / "model.safetensors").exists() else None,
            "reference_check": {key: report[key] for key in report if key != "rows"},
            "seconds": round(time.perf_counter() - started, 2),
            "centre_generation": state.generation,
        }
        pl.json_dump(self.out / "terminal_payload.json", out)
        log(f"terminal deliverable={deliverable} payload -> {payload_dir} "
            f"sha256={digest}")
        return out

    def export_center(self, center: pl.CentreState, out_path: Path) -> Path:
        """Offline kernel export of the centre (weight nibbles re-derived from codes).

        Scales, reciprocal chain and alphas are q0's (fixed scaling per ticket
        14); only ``weight`` differs, because only E2M1 codes were edited.  The
        full-scale exporter equivalence is checked once by
        :meth:`materialise_and_export`.
        """
        from safetensors import safe_open
        from safetensors.torch import save_file

        base_file = Path(self.args.kernel_base).expanduser()
        touched = {layer for layer, bucket in center.codes.items() if bucket}
        touched_modules = {self.layout[layer][0] for layer in touched}
        tensors = {}
        with safe_open(base_file, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for key in keys:
                module = key.rsplit(".", 1)[0]
                if key.endswith(".weight") and module in touched_modules:
                    continue
                tensors[key] = handle.get_tensor(key)
            for layer in sorted(touched):
                module, offset = self.layout[layer]
                key = f"{module}.weight"
                # Fused modules carry several checkpoint shards (q/k/v,
                # gate/up): patching each shard from a *fresh* file read would
                # silently drop the previous shard's edits, so reuse the tensor
                # already patched in this loop.
                base_tensor = tensors[key] if key in tensors else handle.get_tensor(key).clone()
                in_features = self.shapes[layer][1]
                if self.reader.kind == "int8":
                    # kernel-format int8 weight is [in, out] contiguous; the
                    # payload-order flat index indexes the *transposed* view.
                    # Element writes through .t() alias the storage; a
                    # reshape(-1) here would flatten in the wrong order.
                    view = base_tensor.t()          # [out_fused, in], zero-copy
                    for flat_index, code in center.codes[layer].items():
                        row, col = divmod(flat_index, in_features)
                        view[offset + row, col] = int(code)
                    tensors[f"{module}.weight"] = base_tensor
                    continue
                flat = base_tensor.reshape(-1).clone()
                for flat_index, code in center.codes[layer].items():
                    row, col = divmod(flat_index, in_features)
                    position = (offset + row) * in_features + col
                    byte_index = position // 2
                    shift = 4 * (position % 2)
                    byte = int(flat[byte_index])
                    flat[byte_index] = (byte & ~(0x0F << shift)) | (int(code) << shift)
                tensors[f"{module}.weight"] = flat.reshape(base_tensor.shape)
        save_file(tensors, str(out_path))
        return out_path

    def materialise_and_export(self, center: pl.CentreState, tag: str) -> dict:
        """Full-independence check: materialise the centre payload, run the
        reference exporter, compare against the fast in-memory export."""
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        started = time.perf_counter()
        payload_dir = self.out / f"payload_{tag}"
        payload_dir.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "generation_config.json", "tokenizer.json",
                     "tokenizer_config.json", "chat_template.jinja"):
            source = Path(self.args.model).expanduser() / name
            if source.exists():
                (payload_dir / name).write_bytes(source.read_bytes())
        touched = {layer for layer, bucket in center.codes.items() if bucket}
        source_file = Path(self.args.model).expanduser() / "model.safetensors"
        tensors = {}
        with safe_open(source_file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
        # Patch the *raw* packed bytes of the edits only: the payload carries
        # ~70M non-canonical code-8 nibbles (equal to code 0 in value), and a
        # re-encode from canonical codes would rewrite them and make the
        # byte-level comparison against the engine dump fail spuriously.
        # int8: codes are the bytes themselves; there is nothing to fold.
        for layer in touched:
            if self.reader.kind == "int8":
                weight = tensors[f"{layer}.weight"].clone()
                flat = weight.reshape(-1)
                for flat_index, code in center.codes[layer].items():
                    flat[int(flat_index)] = int(code)
                tensors[f"{layer}.weight"] = weight
                continue
            packed = tensors[f"{layer}.weight_packed"].clone()
            flat = packed.reshape(-1)
            for flat_index, code in center.codes[layer].items():
                position = flat_index // 2
                shift = 4 * (flat_index % 2)
                byte = int(flat[position])
                flat[position] = (byte & ~(0x0F << shift)) | (int(code) << shift)
            tensors[f"{layer}.weight_packed"] = packed
        save_file(tensors, str(payload_dir / "model.safetensors"))
        del tensors
        reference = payload_dir / "kernel_reference.safetensors"
        subprocess.run([
            sys.executable, self.args.kernel_base_script,
            "--payload", str(payload_dir), "--out", str(reference),
        ], check=True, stdout=subprocess.DEVNULL)
        fast = self.out / f"kernel_fast_{tag}.safetensors"
        self.export_center(center, fast)
        compare_path = self.out / f"kernel_fast_vs_reference_{tag}.json"
        subprocess.run([
            sys.executable, self.args.compare_script,
            "--left", str(fast), "--right", str(reference), "--out", str(compare_path),
        ], check=True, stdout=subprocess.DEVNULL)
        report = json.loads(compare_path.read_text())
        report["seconds"] = round(time.perf_counter() - started, 2)
        report["payload_dir"] = str(payload_dir)
        fast.unlink(missing_ok=True)
        reference.unlink(missing_ok=True)
        log(f"fast-vs-reference export ({tag}): mismatches={report['mismatches']} "
            f"of {report['tensors']} in {report['seconds']}s")
        return report


def main() -> int:
    if os.environ.get("PILOT_FAULTHANDLER"):
        import faulthandler

        faulthandler.enable()
        faulthandler.dump_traceback_later(90, exit=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="q0 NVFP4 payload directory")
    parser.add_argument("--taskset", required=True, help="dev taskset jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--confirm", help="confirm taskset jsonl (opened once per generation)")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--edits", type=int, default=16)
    parser.add_argument("--edits-plan", help="json: group_id -> sparse edit count")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.02,
                        help="confirm-time non-inferiority tolerance (accuracy); "
                             "does not gate aggregation (QES-original, ungated)")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--limit", type=int, default=0, help="dev item limit (smoke)")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--kernel-base", default=str(Path("out/pilot/kernel_base.safetensors")))
    parser.add_argument("--kernel-base-script", default=str(Path(__file__).with_name("build_kernel_payload.py")))
    parser.add_argument("--stop-policy", action="store_true",
                        help="enforce ticket 13/18 6.5 stop rules between generations "
                             "(--kl-halt / --no-accept-halt / --budget-seconds) and write "
                             "stop_ledger.jsonl; the frozen loop itself has no stop rule")
    parser.add_argument("--kl-baseline", type=float, default=2.1346,
                        help="q0 Generation KL Error on the frozen probe set; the "
                             "materiality floor for the trend clause is measured "
                             "against it")
    parser.add_argument("--trend-materiality", type=float, default=1.5,
                        help="a two-consecutive-rise trend halts only when BOTH "
                             "rising points exceed --kl-baseline x this. 1.5 is "
                             "grounded in the observed generation-to-generation "
                             "spread (-16%% to +20%% of baseline); 0 reproduces the "
                             "literal 'any two rises' rule")
    parser.add_argument("--kl-halt", type=float, default=6.4,
                        help="halt when the centre KL exceeds this (frozen: 3x baseline 2.1346)")
    parser.add_argument("--no-accept-halt", type=int, default=2,
                        help="halt after this many consecutive generations with moved == 0")
    parser.add_argument("--budget-seconds", type=float, default=0.0,
                        help="halt when the process wall clock exceeds this (0 = off)")
    parser.add_argument("--alpha-ladder", action="store_true",
                        help="ticket 18 3.3 two-sided alpha ladder (records every switch)")
    parser.add_argument("--ladder-low-cross", type=float, default=0.40)
    parser.add_argument("--ladder-high-reject", type=float, default=0.05)
    parser.add_argument("--ladder-alpha-high", type=float, default=16.0)
    parser.add_argument("--ladder-alpha-low", type=float, default=8.0)
    parser.add_argument("--compare-script", default=str(Path(__file__).with_name("compare_kernel_files.py")))
    parser.add_argument("--compare-values-script",
                        default=str(Path(__file__).with_name("compare_kernel_values.py")),
                        help="value-semantics gate (tickets 05/18)")
    parser.add_argument("--gate-halt", action="store_true",
                        help="abort the generation when the value gate fails")
    parser.add_argument("--confirm-only", action="store_true",
                        help="deploy the resumed centre, score the confirm pool on q0 and "
                             "on the centre, dump the arms, exit (terminal wording input)")
    parser.add_argument("--rebuild-kernel", action="store_true")
    parser.add_argument("--resume", help="state.json to resume from")
    parser.add_argument("--start-generation", type=int, default=0)
    parser.add_argument("--stop-after-generation", type=int, default=-1)
    parser.add_argument("--replay-alpha", help="comma list of alphas to re-account offline")
    parser.add_argument("--replay-records", action="store_true", help="re-derive populations only")
    parser.add_argument("--probe-prompts")
    parser.add_argument("--bf16-logprobs")
    parser.add_argument("--reference-check", action="store_true",
                        help="run the materialise+reference-exporter equivalence once")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--verify-live", action="store_true",
                        help="read every live coordinate back from the engine each generation")
    parser.add_argument("--keep-dumps", action="store_true")
    parser.add_argument("--no-answer-margin", type=float, default=0.05,
                        help="19 票健康度阈值：逐 arm 的未作答率（相对启动时实测的 q0 基线）"
                             "整体或任一 family 超出这个量即在该代打 flag；归因判定为 "
                             "artifact_dominated 时同样打 flag。只记录，不影响 fitness/停止")
    parser.add_argument("--skip-q0-reference", action="store_true",
                        help="跳过启动时的 q0 未作答基线测量（默认关闭；关掉后归因规则不可用）")
    # ---- 第 2 轮：保留（retention）/ 反馈噪声带 / harness swap。默认关或纯追加 ----
    parser.add_argument("--retention", action="store_true",
                        help="第 2 轮 §1（默认关）：每代中心步之后再读一次反馈通道，逐 family "
                             "分解；任一家族跌幅超过本次运行实测的噪声带就把该步整步回滚"
                             "（残差按票 09 §4 重锚），该代记为 retention_rollback")
    parser.add_argument("--retention-band-scale", type=float, default=1.0,
                        help="噪声带的倍数（预注册值 1.0 = 逐字按实测带判定；改这个值属于"
                             "偏离预注册规则，会写进台账）")
    parser.add_argument("--retention-band-term", default="harness",
                        choices=list(rl.BAND_TERMS),
                        help="阈值挂在哪个实测标准误上（预注册值 harness = 同一中心在完整"
                             "管道上重测的标准误；sampling = 本池 episode 级（簇校正）标准误；"
                             "max = 两者较大者）。实测提醒：打分器逐位确定时 harness 项的 SE "
                             "为 0、带 = 量化下限（1 题），而 dev 池上中心逐代逐 family 的自然"
                             "摆动是 2–9 题 —— 用 harness 项会几乎每代回滚；改判据必须显式选 "
                             "term 并记进台账，不要静默改数")
    parser.add_argument("--noop-rescores", type=int, default=3,
                        help="启动时对**同一个中心**在完整管道上重测的次数（噪声带来源；"
                             "< 2 表示不测，带退回到量化下限 1/n）")
    parser.add_argument("--feedback-noise", action="store_true",
                        help="只测并记录噪声带，不启用 §1 回滚（诊断用）")
    parser.add_argument("--feedback-slice",
                        help="第 2 轮 §4b（默认关）：中心专用的反馈切片 jsonl，每代评一次，"
                             "与 dev 池、封存池均不相交；默认只做监控，不进候选选择")
    parser.add_argument("--feedback-slice-gate", action="store_true",
                        help="把反馈切片接进**保留判定**（切片越带也触发回滚）；候选选择与"
                             "验收门不变")
    parser.add_argument("--prompt-variant", default=PROMPT_VARIANT_DEFAULT,
                        choices=list(_prompt_variants()),
                        help="终端头条形态（默认 frozen = 冻结的 'Answer:' 后缀形态）；"
                             "另一个形态作为 harness swap 臂一并测量")
    parser.add_argument("--terminal-swap", action=argparse.BooleanOptionalAction, default=True,
                        help="第 2 轮 §3（默认开，仅终端 --confirm-only 路径）：q0 与中心在"
                             "**两种 prompt 形态**下都测一遍封存池。代价：封存池每臂一次"
                             "评分（v3 confirm 32,040 题 ≈99 s/臂），4 臂 ≈6-7 min + 载入；"
                             "同一进程内多测几臂不算多次开箱（--looks 语义不变）")
    parser.add_argument("--retention-terminal", action=argparse.BooleanOptionalAction, default=True,
                        help="第 2 轮 §2（默认开，仅终端路径）：交付物 = 中心**当且仅当**它在"
                             "封存池上严格优于 q0，否则交付物就是 q0，记录写 "
                             "no_gain_delivered_baseline")
    parser.add_argument("--terminal-export", action=argparse.BooleanOptionalAction, default=True,
                        help="按 §2 的裁决导出交付载荷（中心或 q0）；关掉只写记录不落盘")
    parser.add_argument("--quantization", default="compressed-tensors")
    parser.add_argument("--linear-backend", default="cutlass")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.30)
    args = parser.parse_args()

    started = time.time()
    runner = Runner(args)
    pl.json_dump(runner.out / "census.json", runner.census)
    log(f"groups={len(runner.groups)} layers={len(runner.layers)} "
        f"coordinates={runner.census['totals']['coordinates']} "
        f"movable={runner.census['totals']['movable']}")

    state_path = Path(args.resume).expanduser() if args.resume else runner.out / "state.json"
    if state_path.exists() and args.resume:
        state = pl.CentreState.from_dict(json.loads(state_path.read_text()))
        log(f"resumed centre generation={state.generation} "
            f"delta_coords={sum(len(b) for b in state.codes.values())} "
            f"residual_coords={sum(len(b) for b in state.residual.values())}")
    else:
        state = pl.CentreState()
    center_sha = state.sha256(runner._base_hash)
    log(f"centre sha256={center_sha}")

    if args.replay_records:
        return replay_records(runner, args, state)

    runner.kernel_base = runner.ensure_kernel_base()
    layout_check = runner.verify_layout()
    # The engine core is forked from this process; release the decoded payload
    # cache first (a multi-GB resident map in the parent wedges the child's
    # pinned/UVA allocation during init).
    runner.reader.release_codes()
    runner.base_codes.clear()
    runner.start_engine()
    runner.prepare_requests()

    history_path = runner.out / "history.jsonl"
    history = [json.loads(line) for line in history_path.open()] if history_path.exists() else []
    final_generation = args.generations
    if args.stop_after_generation >= 0:
        final_generation = min(final_generation, args.stop_after_generation + 1)

    live: dict[tuple[str, int], int] = {}
    if state.codes:
        patch = runner.patch(live, center_delta(state))
        log(f"deployed resumed centre: {patch}")

    # 19 票：启动时实测一次 q0 在同一 dev 池上的分数与未作答率，作为本次运行的基线
    if not args.skip_q0_reference:
        runner.q0_reference_measure(live)
    else:
        log("q0 reference measurement skipped (--skip-q0-reference): "
            "no-answer attribution against q0 is unavailable")
        runner.q0_reference = None

    # 第 2 轮 §4a：本次运行自己的反馈噪声带（只有打开 retention/噪声带/切片时才测）
    if args.retention or args.feedback_noise or args.feedback_slice:
        runner.measure_feedback_noise(live, state)
    else:
        log("feedback noise band not measured (--retention / --feedback-noise / "
            "--feedback-slice are all off): a retention threshold would fall back to the "
            "quantisation floor")

    if args.reference_check:
        check = runner.materialise_and_export(state, f"gen{state.generation:02d}")
        pl.json_dump(runner.out / "reference_check.json", check)

    if not history:
        rows, timing = runner.score(runner.dev_reqs, "centre")
        summary = pl.fitness_summary(rows)
        log(f"centre gen{state.generation} dev: acc={summary['acc']:.4f} "
            f"gp={summary['gold_prob']:.4f} episodes={summary['episodes']} "
            f"({timing['wall_seconds']}s)")
        baseline = {
            "generation": state.generation,
            "centre_sha256": center_sha,
            "kind": "centre",
            "fitness": summary,
            "no_answer": runner.no_answer(rows),
            "attribution": runner.attribution(rows),
            "timing": timing,
            "rows": rows,
        }
        pl.json_dump(runner.out / "records" / f"centre_gen{state.generation:02d}.json", baseline)
        with (runner.out / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "generation": state.generation, "kind": "centre_baseline",
                "center_sha256_before": center_sha, "center_sha256_after": center_sha,
                "center_acc": summary["acc"], "center_gold_prob": summary["gold_prob"],
                "moved": 0, "rejected_beyond_grid": 0, "residual_clamped": 0,
                "qualified_pairs": 0, "dropped_pairs": [], "kl": runner.kl_probe(),
                # 19 票：基线行也带未作答率与归因（baseline 的 attribution 应与 q0 自比，
                # 即近似全 0；它是后续每代比较的锚）
                "centre_no_answer": baseline["no_answer"]["overall"]["rate"],
                "centre_no_answer_per_family": {k: v["rate"]
                                                for k, v in baseline["no_answer"]["per_family"].items()},
                "centre_attribution": baseline["attribution"],
                "no_answer_flagged": [],
            }, ensure_ascii=False) + "\n")

    if args.confirm_only:
        if runner.confirm_reqs is None:
            raise RuntimeError("--confirm-only requires --confirm")
        # §3：封存池上把 q0 与中心在**每一种** prompt 形态下都测一遍（同一进程 = 同一次开箱）
        variants = terminal_variants(args)
        log(f"terminal arms: variants={variants} (headline={args.prompt_variant}), "
            f"both arms (q0, centre) measured per variant; this is ONE open of the pool "
            f"regardless of how many arms are measured in this process")
        arms_by_variant = runner.confirm_arms_by_variant(live, state)
        headline = args.prompt_variant
        # 向后兼容：`arms` 仍是头条形态的两臂（confirm_gate.py --measure 逐字不变）
        payload = {
            "protocol": pl.PROTOCOL,
            "delta": args.tau,
            "center_sha256": center_sha,
            "center_generation": state.generation,
            "confirm_taskset": args.confirm,
            "prompt_variant": headline,
            "variants": variants,
            "arms": arms_by_variant[headline],
            "arms_by_variant": arms_by_variant,
        }
        # §2 交付物裁决（默认开）+ §3 swap 稳健性门
        terminal = None
        if args.retention_terminal:
            terminal = rl.terminal_retention(arms_by_variant, headline=headline, delta=args.tau)
            terminal["sealed_pool"] = {
                "taskset": args.confirm,
                "opens_in_this_process": 1,
                "note": ("every variant's q0 and centre arms are measured on the same items in "
                         "the same process: one open, several arms; confirm_gate.py --measure "
                         "still consumes payload['arms'] (the headline variant) unchanged"),
            }
            if args.terminal_export and terminal.get("ok"):
                terminal["payload"] = runner.export_terminal_payload(state, terminal["deliverable"])
            pl.json_dump(runner.out / "terminal_retention.json", terminal)
            pl.json_dump(runner.out / "prompt_swap.json", terminal["swap"])
            payload["terminal_retention"] = terminal
            log(f"terminal retention: deliverable={terminal['deliverable']} "
                f"record={terminal['record']} "
                f"headline gain={terminal['headline']['gain_mean']:+.4f} "
                f"swap_gate={terminal['swap']['gate_passes']} "
                f"claimable={terminal['claim']['gain_claimable']}")
        else:
            log("terminal retention DISABLED (--no-retention-terminal): the centre is exported "
                "as the deliverable regardless of the sealed-pool comparison")
        pl.json_dump(runner.out / "records" / "confirm_arms.json", payload)
        log("confirm arms written; compute the wording with "
            "confirm_gate.py --measure records/confirm_arms.json")
        return 0

    stop_verdict = None
    for generation in range(state.generation, final_generation):
        runner.run_generation(state, generation, live, history, center_sha)
        center_sha = state.sha256(runner._base_hash)
        if args.stop_policy:
            stop_verdict = enforce_stop_policy(args, runner, history[-1], history, started)
            if stop_verdict["halt"]:
                pl.json_dump(runner.out / "stop.json", {
                    "protocol": pl.PROTOCOL,
                    "rule": stop_verdict["halt"]["rule"],
                    "detail": stop_verdict["halt"]["detail"],
                    "generation": stop_verdict["generation"],
                    "center_sha256": center_sha,
                    "alpha": args.alpha,
                    "ledger": stop_verdict,
                })
                log(f"STOP after gen{stop_verdict['generation']}: {stop_verdict['halt']}")
                break

    elapsed = round(time.time() - started, 1)
    pl.json_dump(runner.out / "summary.json", {
        "protocol": pl.PROTOCOL,
        "group_protocol": pl.GROUP_PROTOCOL,
        "model": args.model,
        "taskset": args.taskset,
        "generations": final_generation,
        "candidates_per_generation": args.candidates,
        "alpha": args.alpha,
        "tau": args.tau,
        "seed": args.seed,
        "engine_load_seconds": runner.engine_load_seconds,
        "layout_check": {key: value for key, value in layout_check.items() if key != "errors"},
        "timings": runner.timings,
        "wall_seconds": elapsed,
        "final_centre_sha256": center_sha,
        "stop": stop_verdict,
        # 19 票：q0 的未作答基线（启动时实测，归因规则的参考臂）
        "q0_reference": ({"taskset": runner.q0_reference["taskset"],
                          "fitness": runner.q0_reference["fitness"],
                          "no_answer": runner.q0_reference["no_answer"]}
                         if runner.q0_reference else None),
        "no_answer_margin": args.no_answer_margin,
        # 第 2 轮：反馈噪声带 + 保留配置（§4a 要求这个数写进 summary，而不是记忆里的常数）
        "round2": {
            "protocol": rl.PROTOCOL,
            "feedback_noise": runner.feedback_noise,
            "feedback_slice_noise": runner.slice_noise,
            "flags": runner.retention_flags or {
                "retention": bool(args.retention),
                "retention_band_scale": args.retention_band_scale,
                "feedback_slice": args.feedback_slice,
                "feedback_slice_gate": bool(args.feedback_slice_gate),
            },
            "prompt_variant": args.prompt_variant,
            "retention_rollbacks": [
                {"generation": record["generation"],
                 "breaches": [{"family": b["family"], "delta": b["delta"], "band": b["band"]}
                              for b in record["retention"]["channels"]["feedback"]["breaches"]],
                 "worst_family": record["retention"]["channels"]["feedback"]["worst_family"],
                 "worst_delta": record["retention"]["channels"]["feedback"]["worst_delta"]}
                for record in history
                if record.get("retention", {}) and record["retention"].get("rollback")
            ],
            "centre_readings": len(runner.centre_readings),
        },
        "state": state.to_dict(),
    })
    log(f"done in {elapsed}s; final centre sha256={center_sha}")
    return 0


def resolution_report(candidates: list[dict]) -> dict:
    """Episode-level pairing noise: the basis of the confirmation gate.

    * ``candidate_sd`` -- paired-diff SD of each candidate against the centre;
    * ``mirror_pairs`` -- SD inside a mirror pair (same coordinates, opposite
      sign: a pure-noise floor if the response is locally linear);
    * ``all_pairs``   -- every candidate-vs-candidate paired-diff SD.
    """
    from itertools import combinations

    values = []
    for candidate in candidates:
        if candidate["sd"] is not None:
            values.append(candidate["sd"])
    mirror = []
    by_pairs: dict[int, list[dict]] = {}
    for candidate in candidates:
        by_pairs.setdefault(candidate["pair"], []).append(candidate)
    for pair, members in sorted(by_pairs.items()):
        if len(members) != 2:
            continue
        stats = pl.paired_stats(members[0]["episodes"], members[1]["episodes"])
        mirror.append({"pair": pair, "sd": stats["sd"], "mean": stats["mean"]})
    cross = []
    for left, right in combinations(candidates, 2):
        stats = pl.paired_stats(left["episodes"], right["episodes"])
        if stats["sd"] is not None:
            cross.append(stats["sd"])
    episode_accs = {}
    for candidate in candidates:
        episode_accs[candidate["index"]] = [
            float(np.mean(values)) for values in candidate["episodes"].values()
        ]
    return {
        "episodes": len(candidates[0]["episodes"]) if candidates else 0,
        "candidate_sd_mean": float(np.mean(values)) if values else None,
        "candidate_sd_min": float(np.min(values)) if values else None,
        "candidate_sd_max": float(np.max(values)) if values else None,
        "mirror_pairs": mirror,
        "mirror_sd_mean": float(np.mean([item["sd"] for item in mirror])) if mirror else None,
        "mirror_mean_abs": float(np.mean([abs(item["mean"]) for item in mirror])) if mirror else None,
        "all_pairs_n": len(cross),
        "all_pairs_sd_mean": float(np.mean(cross)) if cross else None,
        "all_pairs_sd_median": float(np.median(cross)) if cross else None,
        "all_pairs_sd_min": float(np.min(cross)) if cross else None,
        "all_pairs_sd_max": float(np.max(cross)) if cross else None,
        "candidate_episode_acc": episode_accs,
    }


def record_budget_interval(out: Path, started: float, ended: float) -> None:
    """Append this process's ``[start, ended]`` span to ``budget.jsonl``.

    ``--budget-seconds`` caps the *campaign*, not one process: the long run
    resumes from ``state.json`` whenever it is restarted, so a counter that
    restarts with it would hand every restart a fresh budget.  The interval is
    rewritten at each generation, so a SIGKILLed process still leaves a bounded
    record of the time it spent.
    """
    with (out / "budget.jsonl").open("a") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "start": started,
                                 "end": ended}) + "\n")


def ledger_trailing_seconds(out: Path) -> float:
    """Wall clock reached by the process that wrote the ledger's last entry.

    ``stop_ledger.jsonl`` records the *process* wall clock at each generation,
    so the value grows inside one process and drops when the run resumes in a
    new one.  The last monotone stretch's maximum is therefore what the
    previous process spent -- the fallback for a process that predates
    ``budget.jsonl``.
    """
    peak = 0.0
    previous = None
    ledger = out / "stop_ledger.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            wall = json.loads(line).get("wall_seconds")
            if wall is None:
                continue
            wall = float(wall)
            if previous is not None and wall < previous:
                peak = 0.0  # a restart: only the trailing stretch is the last process
            peak = max(peak, wall)
            previous = wall
    return peak


def cumulative_seconds(out: Path, started: float) -> float:
    """Device seconds spent by earlier processes plus this process's elapsed."""
    spans: dict[float, float] = {}
    recorded = False
    log_path = out / "budget.jsonl"
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            key = round(float(entry["start"]), 3)
            spans[key] = max(spans.get(key, 0.0), float(entry["end"]))
            recorded = recorded or abs(key - round(started, 3)) < 1.0
    prior = sum(end - start for start, end in spans.items() if start < started - 1.0)
    if not recorded:
        # This process's first generation: the interval log says nothing about
        # us yet, so the ledger's trailing stretch is the only evidence of what
        # the previous process spent.  Once we have written our own interval the
        # log is authoritative and the fallback must not fire, or this process's
        # own wall clock would be counted twice.  Pin the handoff as an interval
        # of its own so later generations inherit it instead of re-deriving it.
        carried = max(prior, ledger_trailing_seconds(out))
        if carried > 0.0:
            record_budget_interval(out, started - carried, started)
        prior = carried
    return prior + (time.time() - started)


def enforce_stop_policy(args, runner: Runner, record: dict, history: list,
                        started: float) -> dict:
    """Ticket 13/18 §6.5 stop rules + §3.3 alpha ladder, per generation.

    The frozen loop above runs ``--generations`` to exhaustion: ticket 13's stop
    rules (two consecutive generations without an accepted update, budget
    exhaustion, KL over threshold) and ticket 18 §3.3's two-sided alpha ladder
    are protocol, not scoring, so they are applied here between generations and
    every decision is written to ``stop_ledger.jsonl``.  Nothing in the
    population / aggregation / centre-step math is touched; the only mutation is
    ``args.alpha`` when the ladder fires, which the next generation picks up.

    Guards (a breach halts, per §6.3/§6.5):
      * hard gate  -- enforced in-process by ``--gate-halt`` (raises);
      * readback   -- ``live_readback.value_mismatches != 0``;
      * KL         -- ``generation_kl_error > --kl-halt`` (frozen: 3x the q0
                      baseline 2.1346 -> 6.4), **or** the 05 trend criterion --
                      two consecutive rises (``trend_breach``) that are also
                      materially above the baseline
                      (``trend_material``, floor = ``--kl-baseline`` x
                      ``--trend-materiality``).  Both clauses halt; the
                      materiality band is what keeps the trend clause from
                      firing on wobble inside the baseline's own noise band.
      * no accept  -- ``moved == 0`` for ``--no-accept-halt`` generations
                      (acceptance is defined as ``moved >= 1``);
      * budget     -- cumulative device seconds (this process plus every earlier
                      one, reconstructed from ``budget.jsonl``) exceed
                      ``--budget-seconds``, so a restart cannot reset the cap.
    """
    ledger = record.get("residual_ledger") or {}
    aggregation = record.get("aggregation") or {}
    moved = int(ledger.get("moved", 0))
    # 第 2 轮 §1：被回滚的整步等于"没有发生的更新"，与 moved == 0 同类记账
    retention = record.get("retention") or {}
    rolled_back = bool(retention.get("rollback"))
    if rolled_back:
        moved = 0
    support = int(aggregation.get("gradient_support", 0))
    qualified = int(aggregation.get("qualified_pairs", 0))
    rejected = int(ledger.get("rejected_beyond_grid", 0))
    readback = record.get("live_readback") or {}
    kl = record.get("kl") or {}
    kl_error = kl.get("generation_kl_error")
    cross_grid = (moved / support) if support else None
    reject_rate = (rejected / support) if support else None
    wall = cumulative_seconds(runner.out, started)
    prior_seconds = wall - (time.time() - started)
    record_budget_interval(runner.out, started, time.time())

    # -- trend criterion (05): two consecutive rises ------------------------ #
    # 05 §7 defines the trend on the series' *upper bound* and explicitly "not a
    # single point over threshold"; the literal two-consecutive-rises reading
    # fires on KL wobbling inside the baseline's own band, which is what cut the
    # 2026-09-10 formal run from its intended ~300 generations to 16.  The
    # materiality band restores the documented intent: a rise only counts once
    # the series is clearly above the q0 baseline.  --trend-materiality 0
    # reproduces the literal rule.
    kl_series = [((entry.get("kl") or {}).get("generation_kl_error"))
                 for entry in history if entry.get("kind") != "centre_baseline"]
    if not history or history[-1].get("generation") != record.get("generation"):
        kl_series.append(kl_error)  # tolerate being called with a stale history
    trend_breach = bool(
        len(kl_series) >= 3
        and None not in kl_series[-3:]
        and kl_series[-1] > kl_series[-2] > kl_series[-3]
    )
    material_floor = args.kl_baseline * args.trend_materiality
    trend_material = bool(
        trend_breach and kl_series[-1] > material_floor and kl_series[-2] > material_floor
    )

    state = getattr(enforce_stop_policy, "_state", None)
    if state is None:
        state = {"no_accept": 0, "low_cross": 0, "high_reject": 0, "switches": []}
        enforce_stop_policy._state = state

    halt = None
    if readback and readback.get("value_mismatches"):
        halt = {"rule": "readback_value_mismatch",
                "detail": readback.get("value_mismatches")}
    if halt is None and kl_error is not None and kl_error > args.kl_halt:
        halt = {"rule": "kl_over_threshold",
                "detail": {"kl": kl_error, "threshold": args.kl_halt}}
    if halt is None and trend_material:
        halt = {"rule": "kl_trend_breach",
                "detail": {"series": kl_series[-3:],
                           "materiality_floor": material_floor,
                           "materiality": args.trend_materiality,
                           "baseline": args.kl_baseline}}
    if halt is None and moved == 0:
        state["no_accept"] += 1
        if state["no_accept"] >= args.no_accept_halt:
            halt = {"rule": "no_accept",
                    "detail": {"consecutive": state["no_accept"],
                               "qualified_pairs": qualified}}
    else:
        state["no_accept"] = 0
    if halt is None and args.budget_seconds > 0 and wall > args.budget_seconds:
        halt = {"rule": "budget_exhausted",
                "detail": {"cumulative_seconds": round(wall, 1),
                           "prior_processes_seconds": round(prior_seconds, 1),
                           "budget_seconds": args.budget_seconds}}

    # -- alpha ladder (§3.3, only ever changes alpha) ----------------------- #
    switch = None
    if not halt and args.alpha_ladder:
        if cross_grid is not None and cross_grid < args.ladder_low_cross and qualified >= 8:
            state["low_cross"] += 1
        else:
            state["low_cross"] = 0
        if reject_rate is not None and reject_rate > args.ladder_high_reject and qualified <= 8:
            state["high_reject"] += 1
        else:
            state["high_reject"] = 0
        if state["low_cross"] >= 2 and args.alpha != args.ladder_alpha_high:
            switch = {"from": args.alpha, "to": args.ladder_alpha_high,
                      "trigger": "cross_grid < %.2f x2 and M >= 8" % args.ladder_low_cross,
                      "evidence": {"cross_grid": cross_grid, "qualified_pairs": qualified,
                                   "generations": [entry["generation"] for entry in history[-2:]]}}
            args.alpha = args.ladder_alpha_high
        elif state["high_reject"] >= 2 and args.alpha != args.ladder_alpha_low:
            switch = {"from": args.alpha, "to": args.ladder_alpha_low,
                      "trigger": "reject > %.2f x2 and M <= 8" % args.ladder_high_reject,
                      "evidence": {"reject_rate": reject_rate, "qualified_pairs": qualified}}
            args.alpha = args.ladder_alpha_low
        if switch:
            state["low_cross"] = 0
            state["high_reject"] = 0
            state["switches"].append({"generation": record["generation"], **switch})

    verdict = {
        "generation": record["generation"],
        "moved": moved,
        "gradient_support": support,
        "qualified_pairs": qualified,
        "rejected_beyond_grid": rejected,
        "cross_grid": cross_grid,
        "reject_rate": reject_rate,
        "kl": kl_error,
        "trend_breach": trend_breach,
        "trend_material": trend_material,
        "materiality_floor": round(material_floor, 4),
        "readback_value_mismatches": readback.get("value_mismatches"),
        "wall_seconds": round(wall, 1),
        "alpha": args.alpha,
        "switch": switch,
        "halt": halt,
    }
    pl.json_dump(runner.out / "records" / f"stop_policy_gen{record['generation']:02d}.json", verdict)
    with (runner.out / "stop_ledger.jsonl").open("a") as handle:
        handle.write(json.dumps(verdict, ensure_ascii=False) + "\n")
    log(f"guards gen{record['generation']}: moved={moved}/{support} "
        f"cross={cross_grid if cross_grid is None else round(cross_grid, 3)} "
        f"reject={rejected} M={qualified} kl={kl_error} "
        f"alpha={args.alpha}{' TREND-BREACH' if trend_breach else ''}"
        f"{' SWITCH' if switch else ''}{' HALT:' + halt['rule'] if halt else ''}")
    return verdict


def replay_records(runner: Runner, args, state: pl.CentreState) -> int:
    """Re-derive every generation's population from the records alone."""
    records = sorted((runner.out / "records").glob("gen*.json"))
    report = {"populations": [], "all_match": True}
    for path in records:
        record = json.loads(path.read_text())
        population = pl.propose_generation(
            runner.groups, runner.masks,
            generation=record["generation"], group_index=record["group_index"],
            pairs=record["pairs"], edits_per_candidate=record["edits_per_candidate"],
            seed=args.seed,
        )
        mine = [
            {"pair": c.pair, "side": c.side, "sign": c.sign,
             "coordinates": [[layer, flat] for layer, flat in c.coordinates]}
            for c in population
        ]
        theirs = [
            {"pair": c["pair"], "side": c["side"], "sign": c["sign"], "coordinates": c["coordinates"]}
            for c in record["population"]
        ]
        match = mine == theirs
        report["all_match"] &= match
        report["populations"].append({"generation": record["generation"], "match": match})
    pl.json_dump(runner.out / "replay_report.json", report)
    print(json.dumps(report, indent=1))
    return 0 if report["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
