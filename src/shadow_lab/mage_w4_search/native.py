"""Independent packed NVFP4 handoff and eager Mage execution.

Real model imports are lazy. An installation report is representation evidence,
never native-compute evidence; only a real CUDA profile can supply the latter.
The caller owns request authorization, data purpose and the device budget.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class NativeContractError(ValueError):
    """An independent quantized request violates its frozen contract."""


def _hash(value):
    from ..mage_qad_checkpoint import content_hash

    return content_hash(value)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(path, allowed):
    resolved = Path(path).expanduser().resolve(strict=True)
    frozen = {str(Path(p).expanduser().resolve()): digest for p, digest in allowed.items()}
    if str(resolved) not in frozen:
        raise NativeContractError(f"asset-not-allowed: {resolved}")
    digest = _file_hash(resolved)
    if digest != frozen[str(resolved)]:
        raise NativeContractError(f"asset-file-hash-mismatch: {resolved}")
    return resolved, digest


@dataclass(frozen=True)
class PackedLayer:
    qdata: Any
    block_scale: Any
    tensor_scale: Any
    shape: tuple[int, int]

    def record(self):
        return {
            "weight_packed": self.qdata,
            "weight_scale": self.block_scale,
            "tensor_scale": self.tensor_scale,
        }


@dataclass(frozen=True)
class QuantizedAssets:
    layers: Mapping[str, PackedLayer]
    remainder: Mapping[str, Any]
    source_files: Mapping[str, str]
    sha256: str
    fixed_sha256: str

    def record(self):
        return {
            "packed": {n: v.record() for n, v in self.layers.items()},
            "remainder": dict(self.remainder),
        }

    def verify(self):
        if _hash(self.record()) != self.sha256:
            raise NativeContractError("center-hash: tensors changed without a new center identity")
        fixed = {
            "scales": {n: [v.block_scale, v.tensor_scale] for n, v in self.layers.items()},
            "remainder": dict(self.remainder),
        }
        if _hash(fixed) != self.fixed_sha256:
            raise NativeContractError("fixed-state-hash: frozen scales or remainder changed")


def load_quantized_assets(
    packed_path,
    remainder_path,
    *,
    allowed_assets: Mapping[str, str],
    expected_layer_count: int = 174,
    expected_shapes: Mapping[str, tuple[int, int]] | None = None,
) -> QuantizedAssets:
    """Read only the two allowlisted independent payloads; hash names mean file SHA256.

    Legacy clip/config metadata in a QAD export is not used by the runtime.
    The embedded payload hash is independently checked, then discarded.
    """
    import torch

    from ..mage_qad_checkpoint import read_checkpoint

    packed_file, packed_sha = _verified_file(packed_path, allowed_assets)
    rest_file, rest_sha = _verified_file(remainder_path, allowed_assets)
    try:
        payload = read_checkpoint(packed_file)
        rest = read_checkpoint(rest_file)
    except (ValueError, KeyError, TypeError) as error:
        raise NativeContractError(f"payload-integrity: {error}") from error
    allowed_fields = {
        "sha256",
        "schema_version",
        "identity",
        "packed",
        "clips",
        "layout",
        "quasar_config",
    }
    if set(payload) - allowed_fields or payload.get("schema_version") != 1:
        raise NativeContractError("independent-packed-schema: unexpected training state or version")
    if (
        not isinstance(payload.get("packed"), dict)
        or len(payload["packed"]) != expected_layer_count
    ):
        raise NativeContractError("layer-count: packed census differs from the frozen census")
    if expected_shapes is not None and set(expected_shapes) != set(payload["packed"]):
        raise NativeContractError("layer-shape-census: names differ")
    rest.pop("sha256")
    for name, value in rest.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise NativeContractError("remainder-schema: tensor state only")
        if not torch.isfinite(value.float()).all():
            raise NativeContractError(f"remainder-nonfinite: {name}")
    layers = {}
    for name, entry in payload["packed"].items():
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise NativeContractError("packed-schema: invalid layer")
        required = {"weight_packed", "weight_scale", "tensor_scale"}
        if not required.issubset(entry) or set(entry) - required - {"input_tensor_scale"}:
            raise NativeContractError(f"packed-schema: {name}")
        qdata, scale, tensor_scale = (
            entry[k] for k in ("weight_packed", "weight_scale", "tensor_scale")
        )
        if not all(isinstance(x, torch.Tensor) for x in (qdata, scale, tensor_scale)):
            raise NativeContractError(f"packed-tensor-type: {name}")
        if qdata.dtype != torch.uint8 or qdata.ndim != 2 or not qdata.is_contiguous():
            raise NativeContractError(f"packed-layout: {name}")
        n, width = qdata.shape
        if not n or not width or n % 16 or (width * 2) % 16:
            raise NativeContractError(f"packed-shape-block16: {name}")
        if scale.dtype != torch.float8_e4m3fn or tuple(scale.shape) != (n, width // 8):
            raise NativeContractError(f"block-scale-layout: {name}")
        if (
            not scale.is_contiguous()
            or not torch.isfinite(scale.float()).all()
            or (scale.float() < 0).any()
        ):
            raise NativeContractError(f"block-scale-value: {name}")
        if (
            tensor_scale.dtype != torch.float32
            or tensor_scale.numel() != 1
            or not torch.isfinite(tensor_scale).all()
            or tensor_scale.item() <= 0
        ):
            raise NativeContractError(f"tensor-scale-value: {name}")
        if expected_shapes is not None and tuple(expected_shapes[name]) != (n, width * 2):
            raise NativeContractError(f"layer-shape: {name}")
        layers[name] = PackedLayer(qdata, scale, tensor_scale.reshape(()), (n, width * 2))
    record = {"packed": {n: v.record() for n, v in layers.items()}, "remainder": rest}
    fixed = {
        "scales": {n: [v.block_scale, v.tensor_scale] for n, v in layers.items()},
        "remainder": rest,
    }
    return QuantizedAssets(
        layers,
        rest,
        {str(packed_file): packed_sha, str(rest_file): rest_sha},
        _hash(record),
        _hash(fixed),
    )


def _native_tensor(layer, *, device, dtype):
    """Direct packed constructor, equivalent to nvfp4_runtime's native installer.

    Only the E4M3 block scales are swizzled. Qdata is already the E2M1 packed
    matrix. The stored tensor scale is a dequantization multiplier, not its
    reciprocal. Dynamic A4 deliberately does not consume QAD calibration.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        QuantizeTensorToNVFP4Kwargs,
    )
    from torchao.prototype.mx_formats.utils import (
        hp_data_dims_to_swizzled_scale_dims_nvfp4,
        to_blocked,
    )

    n, k = layer.shape
    scales = to_blocked(layer.block_scale.to(device).contiguous()).flatten()
    scales = scales.view(*hp_data_dims_to_swizzled_scale_dims_nvfp4(n, k))
    return NVFP4Tensor(
        layer.qdata.to(device).contiguous(),
        scales,
        16,
        dtype,
        per_tensor_scale=layer.tensor_scale.to(device).reshape(()),
        act_per_tensor_scale=None,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        act_quant_kwargs=QuantizeTensorToNVFP4Kwargs(
            block_size=16,
            is_swizzled_scales=True,
            use_triton_kernel=False,
            use_dynamic_per_tensor_scale=True,
        ),
    )


def install_quantized_assets(
    transformer, assets: QuantizedAssets, *, device="cuda", tensor_factory=None
) -> dict:
    """Replace all Linear weights directly, restoring every other model state.

    ``tensor_factory`` is an external backend seam for small CPU fixtures; using
    it always produces fixture evidence. It cannot qualify native execution.
    """
    import torch

    started = time.monotonic()
    assets.verify()
    modules = {n: m for n, m in transformer.named_modules() if isinstance(m, torch.nn.Linear)}
    if set(modules) != set(assets.layers):
        raise NativeContractError("linear-coverage: packed layers must cover every Linear exactly")
    weight_keys = {f"{name}.weight" for name in modules}
    state = transformer.state_dict()
    ignored = {f"{name}.input_tensor_scale" for name in modules} & set(assets.remainder)
    required = set(state) - weight_keys
    actual = set(assets.remainder) - ignored
    if required != actual:
        raise NativeContractError(
            f"remainder-coverage: missing={sorted(required - actual)}, "
            f"unexpected={sorted(actual - required)}"
        )
    for name, module in modules.items():
        if tuple(module.weight.shape) != assets.layers[name].shape:
            raise NativeContractError(f"linear-shape: {name}")
        if module.weight.dtype != torch.bfloat16:
            raise NativeContractError(f"linear-dtype: expected bfloat16 output for {name}")
    for name in required:
        source, target = assets.remainder[name], state[name]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise NativeContractError(
                f"remainder-layout: {name}: {source.shape}/{source.dtype} "
                f"!= {target.shape}/{target.dtype}"
            )
    for name in ignored:
        value = assets.remainder[name]
        if value.numel() != 1 or not torch.isfinite(value).all() or value.item() <= 0:
            raise NativeContractError(f"calibration-buffer: {name}")
    if tensor_factory is None:
        target = torch.device(device)
        if target.type != "cuda" or torch.cuda.get_device_capability(target) != (12, 1):
            raise NativeContractError("native-device: this campaign requires GB10 CC 12.1")
    factory = tensor_factory or _native_tensor
    # Validate coverage before changing any model state. A factory failure may
    # leave a partial installation; callers must discard that failed runtime.
    with torch.no_grad():
        for name, module in modules.items():
            module.weight = torch.nn.Parameter(
                factory(assets.layers[name], device=device, dtype=torch.bfloat16),
                requires_grad=False,
            )
        for name in required:
            owner_name, _, attr = name.rpartition(".")
            owner = transformer.get_submodule(owner_name) if owner_name else transformer
            value = assets.remainder[name].to(device)
            if attr in owner._parameters:
                setattr(owner, attr, torch.nn.Parameter(value, requires_grad=False))
            else:
                setattr(owner, attr, value)
        transformer.requires_grad_(False)
    return {
        "center_sha256": assets.sha256,
        "fixed_sha256": assets.fixed_sha256,
        "source_file_sha256": dict(assets.source_files),
        "installed_layers": sorted(modules),
        "activation_quantization": "dynamic_per_tensor_and_block16",
        "ignored_calibration_buffers": sorted(ignored),
        "native_compute": False,
        "backend": "fixture" if tensor_factory else "torchao_nvfp4",
        "install_seconds": time.monotonic() - started,
    }


def save_quantized_assets(assets: QuantizedAssets, directory) -> dict:
    """Save an independent center; the new directory is committed by its receipt.

    Packed tensor bytes and canonical content identities survive serialization;
    the outer torch archive is not required to have the same file SHA256.
    """
    from ..mage_qad_checkpoint import atomic_save

    assets.verify()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    packed, rest = directory / "packed.pt", directory / "remainder.pt"
    payload = {"schema_version": 1, "packed": assets.record()["packed"]}
    remainder = dict(assets.remainder)
    atomic_save({**payload, "sha256": _hash(payload)}, packed)
    atomic_save({**remainder, "sha256": _hash(remainder)}, rest)
    receipt = {
        "packed_path": str(packed),
        "remainder_path": str(rest),
        "file_sha256": {str(p): _file_hash(p) for p in (packed, rest)},
        "center_sha256": assets.sha256,
        "fixed_sha256": assets.fixed_sha256,
    }
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


@dataclass(frozen=True)
class GenerationConfig:
    # Explicit key avoids pipeline fallback reading ~/.mageflow or environment.
    watermark_key: int = field(repr=False)
    height: int = 512
    width: int = 512
    steps: int = 4
    cfg: float = 1.0
    negative_prompt: str = " "
    prompt_template: str = "mage-flow"
    attention_backend: str = "flash2"
    replay_max_abs_pixel_error: int = 0
    require_identical_png: bool = True
    deterministic_algorithms: bool = True
    allow_tf32: bool = False

    def __post_init__(self):
        if (self.height, self.width, self.steps, self.cfg) != (512, 512, 4, 1.0):
            raise NativeContractError("generation-load: expected frozen 512x512/4/CFG1")
        if not isinstance(self.watermark_key, int) or not 0 <= self.watermark_key < 2**64:
            raise NativeContractError("watermark-key: explicit unsigned 64-bit integer required")
        if (
            not isinstance(self.replay_max_abs_pixel_error, int)
            or not 0 <= self.replay_max_abs_pixel_error <= 255
        ):
            raise NativeContractError("replay-comparator: expected integer pixel tolerance 0..255")
        if self.attention_backend not in {"flash2", "flash4"}:
            raise NativeContractError("attention-backend: unsupported")

    def identity(self):
        identity = asdict(self)
        identity.pop("watermark_key")
        identity["watermark_key_sha256"] = hashlib.sha256(
            str(self.watermark_key).encode()
        ).hexdigest()
        return identity


def build_native_pipeline(
    assets: QuantizedAssets,
    *,
    auxiliary_root,
    allowed_assets,
    generation_config: GenerationConfig,
    device="cuda",
    expected_runtime: Mapping[str, str],
    before_stage=None,
):
    """Build Mage from structure + independent tensors + allowlisted auxiliaries.

    ``expected_runtime`` maps installed distribution names to exact versions.
    Include torch, torchao, diffusers and transformers; no packages are upgraded.
    All auxiliary files and Mage Python sources must appear in ``allowed_assets``.
    The hook lets the caller account for cold start/install against its lease.
    This entry point never calls the stock pipeline/transformer loader.
    """
    import importlib.metadata
    import importlib.util

    import torch

    started = time.monotonic()
    if before_stage:
        before_stage("native_cold_start")
    assets.verify()
    required_versions = {"torch", "torchao", "diffusers", "transformers"}
    if not required_versions.issubset(expected_runtime):
        raise NativeContractError("runtime-identity: missing frozen versions")
    versions = {name: importlib.metadata.version(name) for name in expected_runtime}
    if versions != dict(expected_runtime):
        raise NativeContractError(f"runtime-version-mismatch: {versions}")
    target = torch.device(device)
    if target.type != "cuda" or torch.cuda.get_device_capability(target) != (12, 1):
        raise NativeContractError("native-device: this campaign requires GB10 CC 12.1")
    root = Path(auxiliary_root).resolve(strict=True)
    reads = dict(assets.source_files)

    def verified(path):
        resolved, digest = _verified_file(path, allowed_assets)
        reads[str(resolved)] = digest
        return resolved

    def read_json(path):
        return json.loads(verified(path).read_text())

    model_index = read_json(root / "model_index.json")
    transformer_config = read_json(root / "transformer/config.json")
    scheduler_path = verified(root / "scheduler/scheduler_config.json")

    def auxiliary(key):
        value = model_index.get(key)
        if not isinstance(value, str) or not value:
            raise NativeContractError(f"auxiliary-identity: missing {key}")
        return (root / value).resolve(strict=True)

    vae = verified(auxiliary("_vae_source"))
    text_encoder = auxiliary("_text_encoder_path")
    if not text_encoder.is_dir():
        raise NativeContractError("text-encoder: local directory required")
    encoder_files = sorted(p for p in text_encoder.rglob("*") if p.is_file())
    if not encoder_files or not (text_encoder / "config.json").is_file():
        raise NativeContractError("text-encoder: incomplete local snapshot")
    for path in encoder_files:
        verified(path)
    # Verify the source tree before importing Mage, including its sampler and
    # text/VAE definitions. Graph evidence was inspected against these APIs.
    spec = importlib.util.find_spec("mage_flow")
    if spec is None or not spec.submodule_search_locations:
        raise NativeContractError("mage-package: local Mage installation required")
    source_root = Path(next(iter(spec.submodule_search_locations)))
    for path in sorted(source_root.rglob("*.py")):
        if any(
            part.startswith(".") or part == "__pycache__"
            for part in path.relative_to(source_root).parts
        ):
            continue
        verified(path)

    from diffusers import FlowMatchEulerDiscreteScheduler
    from mage_flow.models.mage_flow import MageFlowModel, ModelConfig
    from mage_flow.pipeline import MageFlowPipeline

    metadata = {
        "_class_name",
        "txt_max_length",
        "max_sequence_length",
        "param_dtype",
        "packing",
        "schedule_mode",
        "static_shift",
        "use_time_shift",
        "rope_type",
        "apply_text_rotary_emb",
        "mlp_ratio",
        "depth_single_blocks",
        "theta",
        "qkv_bias",
        "guidance_embed",
        "vec_in_dim",
        "vec_type",
        "time_type",
        "double_block_type",
    }
    structure = {k: v for k, v in transformer_config.items() if k not in metadata}
    config = ModelConfig(
        vae_path=str(vae),
        txt_enc_path=str(text_encoder),
        model_structure=structure,
        txt_max_length=transformer_config.get("txt_max_length", 2048),
        packing=transformer_config.get("packing", True),
        static_shift=transformer_config.get("static_shift", 6.0),
        attn_type=generation_config.attention_backend,
        pretrained_model_name_or_path=None,
        pretrained_full_model_path=None,
    )
    # The constructor allocates structure randomly but reads no MMDiT weights.
    # Exhaustive replacement below is required before any model invocation.
    model = MageFlowModel(config)
    model.transformer.to(dtype=torch.bfloat16)
    if before_stage:
        before_stage("native_install")
    installation = install_quantized_assets(model.transformer, assets, device=device)
    model.transformer.to(device=device)
    model.txt_enc.to(device=device, dtype=torch.bfloat16)
    model.vae.to(device=device, dtype=torch.bfloat16)
    model.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        json.loads(scheduler_path.read_text())
    )
    model.eval()
    runtime_identity = {
        "versions": versions,
        "source_file_sha256": reads,
        "model_structure": structure,
        "text_encoder_path": str(text_encoder),
        "vae_path": str(vae),
        "scheduler_config": json.loads(scheduler_path.read_text()),
        "generation_config": generation_config.identity(),
        "device_name": torch.cuda.get_device_name(target),
        "compute_capability": [12, 1],
        "torch_cuda": torch.version.cuda,
        "cold_start_seconds": time.monotonic() - started,
        "asset_evidence": "verified explicit inputs; external I/O isolation is caller responsibility",
    }
    return NativePipeline(
        MageFlowPipeline(model, device),
        assets,
        generation_config,
        installation,
        runtime_identity,
        device=device,
        before_stage=before_stage,
    )


class NativePipeline:
    """Serial eager generation and predeclared same-device save/reload replay.

    Requests are nonempty sequences of ``sample_id/prompt/seed`` dictionaries.
    The upstream contract must authorize their purpose and budget. This runtime
    neither opens prompt manifests nor implements reward/acceptance decisions.
    """

    def __init__(
        self,
        pipeline,
        assets,
        generation_config,
        installation,
        runtime_identity,
        *,
        device="cuda",
        before_stage=None,
        tensor_factory=None,
    ):
        self.pipeline = pipeline
        self.assets = assets
        self.config = generation_config
        self.installation = installation
        self.runtime_identity = runtime_identity
        self.device = device
        self.before_stage = before_stage
        self.tensor_factory = tensor_factory

    def generate(self, samples, directory) -> dict:
        import torch

        samples = list(samples)
        ids = [item.get("sample_id") for item in samples]
        if (
            not samples
            or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise NativeContractError("samples: unique nonempty sample_id values required")
        for item in samples:
            if (
                not isinstance(item.get("prompt"), str)
                or not item["prompt"].strip()
                or not isinstance(item.get("seed"), int)
                or not 0 <= item["seed"] < 2**32
            ):
                raise NativeContractError("samples: explicit prompt and uint32 seed required")
        self.assets.verify()
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        config = self.config
        real_cuda = (
            self.installation["backend"] == "torchao_nvfp4"
            and torch.device(self.device).type == "cuda"
            and self.tensor_factory is None
        )
        report = {
            "schema_version": 1,
            "center_sha256": self.assets.sha256,
            "fixed_sha256": self.assets.fixed_sha256,
            "installation": self.installation,
            "runtime": self.runtime_identity,
            "generation": config.identity(),
            "request_sha256": _hash(samples),
            "rows": [],
            "native_compute": False,
        }
        previous = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        started = time.monotonic()
        try:
            torch.use_deterministic_algorithms(config.deterministic_algorithms)
            torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
            torch.backends.cudnn.allow_tf32 = config.allow_tf32
            for index, item in enumerate(samples):
                row = {
                    "sample_id": item["sample_id"],
                    "seed": item["seed"],
                    "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                }
                row_start = time.monotonic()
                try:
                    if self.before_stage:
                        self.before_stage("native_generate")
                    if real_cuda:
                        torch.cuda.synchronize(self.device)
                        torch.cuda.reset_peak_memory_stats(self.device)
                        # Mage returns a placeholder on refusal. Classify this
                        # before generation, so it cannot enter image rewards.
                        verdict = self.pipeline.model.txt_enc.screen_text(item["prompt"])
                        if verdict.violates:
                            raise NativeContractError("prompt-refused: no scored image produced")
                    kwargs = dict(
                        seeds=[item["seed"]],
                        heights=[config.height],
                        widths=[config.width],
                        steps=config.steps,
                        cfg=config.cfg,
                        neg_prompts=[config.negative_prompt],
                        prompt_template=config.prompt_template,
                        gs_key=config.watermark_key,
                        static_shift=None,
                        renormalization=False,
                        batch_cfg=True,
                    )
                    kernel_names = []
                    with torch.inference_mode():
                        if real_cuda:
                            with torch.profiler.profile(
                                activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ]
                            ) as prof:
                                images = self.pipeline.generate([item["prompt"]], **kwargs)
                                torch.cuda.synchronize(self.device)
                            kernel_names = [
                                e.name for e in prof.events() if str(e.device_type).endswith("CUDA")
                            ]
                            trace_path = directory / f"{index:04d}.trace.json"
                            prof.export_chrome_trace(str(trace_path))
                            row["trace_path"] = str(trace_path)
                            row["trace_file_sha256"] = _file_hash(trace_path)
                            from ..nvfp4_runtime import native_kernel_evidence

                            evidence = native_kernel_evidence(kernel_names)
                            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                                self.device
                            )
                            row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
                        else:
                            images = self.pipeline.generate([item["prompt"]], **kwargs)
                            evidence = {
                                "native_compute": False,
                                "compute_kernel": [],
                                "fallback_reason": "fixture execution",
                            }
                    if len(images) != 1 or images[0].size != (config.width, config.height):
                        raise NativeContractError(
                            "generation-output: missing image or wrong resolution"
                        )
                    image = images[0].convert("RGB")
                    path = directory / f"{index:04d}.png"
                    image.save(path, format="PNG")
                    row.update(
                        status="ok",
                        image_path=str(path),
                        image_file_sha256=_file_hash(path),
                        pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        native_evidence=evidence,
                        cuda_kernel_names=sorted(set(kernel_names)),
                    )
                except Exception as error:
                    row.update(status="failed", error_type=type(error).__name__, error=str(error))
                row["end_to_end_seconds"] = time.monotonic() - row_start
                report["rows"].append(row)
                if row["status"] != "ok":
                    break  # Includes exhausted budgets; never proceed to another image.
        finally:
            torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[2:]
            report["end_to_end_seconds"] = time.monotonic() - started
            rows = report["rows"]
            report["status"] = (
                "ok"
                if len(rows) == len(samples) and all(r["status"] == "ok" for r in rows)
                else "failed"
            )
            report["native_compute"] = (
                bool(rows)
                and report["status"] == "ok"
                and all(r.get("native_evidence", {}).get("native_compute", False) for r in rows)
            )
            report["qualified"] = report["status"] == "ok" and report["native_compute"]
            (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    def replay(self, samples, directory) -> dict:
        import numpy as np
        from PIL import Image

        samples = list(samples)
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        first = self.generate(samples, directory / "before")
        if self.before_stage:
            self.before_stage("native_save_reload")
        started = time.monotonic()
        saved = save_quantized_assets(self.assets, directory / "center")
        restored = load_quantized_assets(
            saved["packed_path"],
            saved["remainder_path"],
            allowed_assets=saved["file_sha256"],
            expected_layer_count=len(self.assets.layers),
            expected_shapes={n: v.shape for n, v in self.assets.layers.items()},
        )
        packed_equal = (
            restored.sha256 == self.assets.sha256
            and restored.fixed_sha256 == self.assets.fixed_sha256
        )
        if not packed_equal:
            raise NativeContractError("replay-reload: packed content identity changed")
        self.installation = install_quantized_assets(
            self.pipeline.model.transformer,
            restored,
            device=self.device,
            tensor_factory=self.tensor_factory,
        )
        self.assets = restored
        reload_seconds = time.monotonic() - started
        second = self.generate(samples, directory / "after")
        comparisons = []
        for before, after in zip(first["rows"], second["rows"], strict=False):
            if before["status"] != "ok" or after["status"] != "ok":
                comparisons.append({"sample_id": before["sample_id"], "passed": False})
                continue
            with (
                Image.open(before["image_path"]) as image_a,
                Image.open(after["image_path"]) as image_b,
            ):
                error = int(
                    np.abs(
                        np.asarray(image_a, dtype=np.int16) - np.asarray(image_b, dtype=np.int16)
                    ).max()
                )
            png_equal = before["image_file_sha256"] == after["image_file_sha256"]
            comparisons.append(
                {
                    "sample_id": before["sample_id"],
                    "max_abs_pixel_error": error,
                    "png_equal": png_equal,
                    "passed": error <= self.config.replay_max_abs_pixel_error
                    and (png_equal or not self.config.require_identical_png),
                }
            )
        image_match = len(comparisons) == len(samples) and all(c["passed"] for c in comparisons)
        native = first["native_compute"] and second["native_compute"]
        report = {
            "scope": "same-runtime-zero-update-save-reload",
            "packed_reload_equal": packed_equal,
            "comparisons": comparisons,
            "images_match": image_match,
            "native_compute": native,
            "qualified": packed_equal and image_match and native,
            "generation": self.config.identity(),
            "save_reload_seconds": reload_seconds,
            "before_report": str(directory / "before/report.json"),
            "after_report": str(directory / "after/report.json"),
            "saved_center": saved,
        }
        (directory / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
