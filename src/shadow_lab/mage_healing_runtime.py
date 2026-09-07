"""GB10 execution adapter: real Mage generation, native materialization and scoring."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch

from .es_quant import FrozenLinearWeights, materialize_es_candidate
from .image_eval import RewardInput, score_reward_batch
from .mage_qad import MageTrajectorySource, trajectory_loss
from .mage_qad_campaign import atomic_json
from .mage_qad_checkpoint import content_hash
from .nvfp4_runtime import materialize_native_nvfp4


def require_gb10() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("Mage healing execution requires GB10 CUDA capability 12.1")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class MageHealingRuntime:
    def __init__(self, pipeline, teacher, master, scorers, *, height: int, width: int):
        require_gb10()
        self.pipeline, self.teacher, self.master, self.scorers = pipeline, teacher, master, scorers
        self.height, self.width = height, width
        self.transformer = pipeline.model.transformer
        pipeline.model.requires_grad_(False).eval()
        teacher.requires_grad_(False).eval()
        self.original = FrozenLinearWeights.capture(teacher, master.layout)
        self.source = MageTrajectorySource(pipeline, height=height, width=width)

    def materialize(self, arm: str, clips: torch.Tensor) -> dict:
        if arm == "qad":
            result = materialize_es_candidate(
                self.transformer, self.master, self.master.layout, clips, device="cuda"
            )
            return result["packed_weight_hashes"]
        self.original.restore(self.transformer, "cuda")
        if arm != "bf16":
            materialize_native_nvfp4(self.transformer, scheme=arm)
        return {"arm": arm}

    def evaluate(
        self,
        clips: torch.Tensor,
        samples,
        out: Path,
        *,
        candidate_sha256: str,
        arm: str = "qad",
        replay: bool = True,
    ) -> dict:
        """Two independent materializations/generations, with images saved each batch.

        BF16/static/RTN are used only as paired/blind baselines. Only the QAD arm
        can pass the native healing quality gate. Every refusal counts as failure.
        ``replay=False`` is the single-pass exploration mode for ES candidates: it
        records no replay evidence, so such a report can never pass ``quality_gate``.
        """
        out.mkdir(parents=True, exist_ok=False)
        report: dict = {
            "candidate_sha256": candidate_sha256,
            "rows": [],
            "failures": 0,
            "compute_capability": list(torch.cuda.get_device_capability()),
            "kernel_names": [],
            "sample_ids": [s.sample_id for s in samples],
            "replay": replay,
        }
        first_rows = {}
        for replaying in (False, True) if replay else (False,):
            packed = self.materialize(arm, clips)
            report["replay_weight_sha256" if replaying else "weight_sha256"] = content_hash(packed)
            for index, sample in enumerate(samples):
                try:
                    trajectory = self.source({"prompt": sample.prompt, "seed": sample.seed})
                    loss = trajectory_loss(self.transformer, self.teacher, trajectory)
                    profile = not replaying and index == 0 and arm == "qad"
                    if profile:
                        from torch.profiler import ProfilerActivity

                        with torch.profiler.profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                        ) as prof:
                            image = self.generate(sample)
                        report["kernel_names"] = sorted(
                            {
                                event.name
                                for event in prof.events()
                                if event.device_type == torch.autograd.DeviceType.CUDA
                            }
                        )
                        prof.export_chrome_trace(str(out / "native-trace.json"))
                    else:
                        image = self.generate(sample)
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    payload = buffer.getvalue()
                    path = out / f"{index:03d}{'-replay' if replaying else ''}.png"
                    path.write_bytes(payload)
                    row = score_reward_batch([RewardInput(payload, sample)], self.scorers)[0]
                    if replaying:
                        first_rows[sample.sample_id].update(
                            replay_sha256=hashlib.sha256(payload).hexdigest(),
                            replay_scores=row["scores"],
                        )
                    else:
                        row.update(image_path=str(path.resolve()), teacher_loss=loss)
                        first_rows[sample.sample_id] = row
                except Exception as error:
                    report["failures"] += 1
                    report.setdefault("errors", []).append(
                        {"sample_id": sample.sample_id, "replay": replaying, "error": str(error)}
                    )
                report["rows"] = list(first_rows.values())
                atomic_json(out / "evaluation.json", report)
        return report

    def generate(self, sample):
        return self.pipeline.generate(
            [sample.prompt],
            seeds=[sample.seed],
            heights=[self.height],
            widths=[self.width],
            steps=4,
            cfg=1.0,
        )[0]
