"""Tests for experiment/analyze_bonsai_vs_unpacked.py.

Uses synthetic packed + unpacked safetensors pairs (no real 54 GB download) to
verify: name normalisation, layer alignment, streaming dequant-vs-ref compare,
and codeword brute-force search.
"""

import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

# The module under test lives under experiment/ which is not a package; add it
# to sys.path the same way experiment scripts are run.
from safetensors.torch import save_file  # noqa: E402

_EXP = os.path.join(os.path.dirname(__file__), "..", "experiment")
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

import analyze_bonsai_vs_unpacked as ab  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers: build synthetic packed + unpacked safetensors on disk              #
# --------------------------------------------------------------------------- #
def _write_safetensors(path: str, tensors: dict[str, torch.Tensor]) -> None:
    save_file(tensors, path)


def _write_index(path: str, weight_map: dict[str, str]) -> None:
    with open(path, "w") as fh:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, fh)


def _make_synthetic_pair(tmp_path, group_size=128, out_features=4, in_features=256):
    """Create one packed layer + matching unpacked (fp16 reference) layer.

    Uses the repo's own packer so the round-trip is exact, then simulates the
    Bonsai naming divergence:
      packed:   language_model.model.layers.0.linear_attn.in_proj_a
      unpacked: model.language_model.layers.0.linear_attn.in_proj_a
    """
    from shadow_peft.quantization import dequantize_2bit_ternary, quantize_2bit_ternary

    torch.manual_seed(0)
    w_fp16 = torch.randn(out_features, in_features, dtype=torch.float16)
    packed, scales, biases = quantize_2bit_ternary(w_fp16, group_size=group_size)
    # The unpacked reference is the *dequantized* form (what a real unpacked
    # FP16 checkpoint stores): the lossy approximation, not the raw weights.
    w_ref = dequantize_2bit_ternary(packed, scales, biases, group_size=group_size)

    packed_name = "language_model.model.layers.0.linear_attn.in_proj_a"
    unpacked_name = "model.language_model.layers.0.linear_attn.in_proj_a"

    packed_dir = tmp_path / "packed"
    unpacked_dir = tmp_path / "unpacked"
    packed_dir.mkdir()
    unpacked_dir.mkdir()

    _write_safetensors(
        str(packed_dir / "model.safetensors"),
        {
            f"{packed_name}.weight": packed,
            f"{packed_name}.scales": scales,
            f"{packed_name}.biases": biases,
        },
    )
    _write_index(
        str(packed_dir / "model.safetensors.index.json"),
        {
            f"{packed_name}.weight": "model.safetensors",
            f"{packed_name}.scales": "model.safetensors",
            f"{packed_name}.biases": "model.safetensors",
        },
    )
    _write_safetensors(
        str(unpacked_dir / "model-00001-of-00001.safetensors"),
        {f"{unpacked_name}.weight": w_ref},
    )
    _write_index(
        str(unpacked_dir / "model.safetensors.index.json"),
        {f"{unpacked_name}.weight": "model-00001-of-00001.safetensors"},
    )
    return {
        "packed_dir": str(packed_dir),
        "unpacked_dir": str(unpacked_dir),
        "packed_name": packed_name,
        "unpacked_name": unpacked_name,
        "w_ref": w_ref,
    }


# --------------------------------------------------------------------------- #
# name normalisation                                                          #
# --------------------------------------------------------------------------- #
class TestNormalizeLayerName:
    def test_packed_prefix_stripped(self):
        assert (
            ab.normalize_layer_name("language_model.model.layers.0.in_proj_a")
            == "layers.0.in_proj_a"
        )

    def test_unpacked_prefix_stripped(self):
        assert (
            ab.normalize_layer_name("model.language_model.layers.0.in_proj_a")
            == "layers.0.in_proj_a"
        )

    def test_both_collapse_to_same_key(self):
        packed = ab.normalize_layer_name("language_model.model.layers.5.mlp.down_proj")
        unpacked = ab.normalize_layer_name("model.language_model.layers.5.mlp.down_proj")
        assert packed == unpacked == "layers.5.mlp.down_proj"

    def test_lm_head_and_embed(self):
        assert ab.normalize_layer_name("language_model.lm_head") == "lm_head"
        assert ab.normalize_layer_name("lm_head") == "lm_head"
        assert ab.normalize_layer_name("language_model.model.embed_tokens") == "embed_tokens"
        assert ab.normalize_layer_name("model.language_model.embed_tokens") == "embed_tokens"

    def test_non_prefixed_passthrough(self):
        assert ab.normalize_layer_name("layers.0.norm.weight") == "layers.0.norm.weight"


# --------------------------------------------------------------------------- #
# layer alignment                                                             #
# --------------------------------------------------------------------------- #
class TestLayerAlignment:
    def test_align_all_triplets_matched(self, tmp_path):
        fixture = _make_synthetic_pair(tmp_path)
        triplets, unpacked_index = ab.load_and_align(fixture["packed_dir"], fixture["unpacked_dir"])
        assert len(triplets) == 1
        assert triplets[0]["packed_base"] == fixture["packed_name"]
        assert fixture["unpacked_name"] + ".weight" in unpacked_index

    def test_missing_unpacked_reports_unmatched(self, tmp_path):
        fixture = _make_synthetic_pair(tmp_path)
        idx_path = tmp_path / "unpacked" / "model.safetensors.index.json"
        with open(idx_path, "w") as fh:
            json.dump({"weight_map": {"something.else.weight": "x.safetensors"}}, fh)
        triplets, _ = ab.load_and_align(fixture["packed_dir"], fixture["unpacked_dir"])
        assert triplets[0]["unpacked_key"] is None


# --------------------------------------------------------------------------- #
# streaming layer compare                                                     #
# --------------------------------------------------------------------------- #
class TestLayerCompare:
    def test_dequant_matches_reference(self, tmp_path):
        """Our own packer is a perfect round-trip, so cosine == 1."""
        fixture = _make_synthetic_pair(tmp_path)
        triplets, unpacked_index = ab.load_and_align(fixture["packed_dir"], fixture["unpacked_dir"])
        row = ab.compare_layer(
            triplets[0],
            fixture["packed_dir"],
            fixture["unpacked_dir"],
            unpacked_index,
        )
        assert row["cosine"] == pytest.approx(1.0, abs=1e-3)
        assert row["mae"] == pytest.approx(0.0, abs=1e-3)
        assert row["max_abs"] == pytest.approx(0.0, abs=1e-3)

    def test_metrics_keys_present(self, tmp_path):
        fixture = _make_synthetic_pair(tmp_path)
        triplets, unpacked_index = ab.load_and_align(fixture["packed_dir"], fixture["unpacked_dir"])
        row = ab.compare_layer(
            triplets[0],
            fixture["packed_dir"],
            fixture["unpacked_dir"],
            unpacked_index,
        )
        for key in ("mse", "mae", "max_abs", "cosine", "rel_l2"):
            assert key in row


# --------------------------------------------------------------------------- #
# codeword brute-force                                                        #
# --------------------------------------------------------------------------- #
class TestBruteForceCodebook:
    def _fixture_codes(self, tmp_path):
        """A tiny layer whose unpacked FP16 values are exactly +-s_g / 0 so that
        the optimal mapping is unambiguous."""
        group_size = 128
        out_features, in_features = 2, 256
        groups_per_row = in_features // group_size  # = 2
        s_g = torch.full((out_features, groups_per_row), 0.5, dtype=torch.float16)
        ternary_vals = torch.tensor([1, 0, -1], dtype=torch.float16)
        t = ternary_vals.repeat(in_features // 3 + 1)[:in_features]
        t = t.unsqueeze(0).repeat(out_features, 1)
        s_g_expanded = s_g.repeat_interleave(group_size, dim=1)
        w_ref = (t * s_g_expanded).to(torch.float16)

        codes = (t + 1).to(torch.int64)  # values in {0,1,2}
        n_words = in_features // 16
        codes = codes.reshape(out_features, n_words, 16)
        shifts = 2 ** (2 * torch.arange(16, dtype=torch.int64))
        packed = (codes * shifts).sum(dim=-1)
        packed = (packed & 0xFFFFFFFF).to(torch.int32)
        biases = (-s_g).to(torch.float16)

        packed_name = "language_model.model.layers.0.test_layer"
        unpacked_name = "model.language_model.layers.0.test_layer"
        packed_dir = tmp_path / "packed"
        unpacked_dir = tmp_path / "unpacked"
        packed_dir.mkdir()
        unpacked_dir.mkdir()
        _write_safetensors(
            str(packed_dir / "model.safetensors"),
            {
                f"{packed_name}.weight": packed,
                f"{packed_name}.scales": s_g,
                f"{packed_name}.biases": biases,
            },
        )
        _write_index(
            str(packed_dir / "model.safetensors.index.json"),
            {
                f"{packed_name}.weight": "model.safetensors",
                f"{packed_name}.scales": "model.safetensors",
                f"{packed_name}.biases": "model.safetensors",
            },
        )
        _write_safetensors(
            str(unpacked_dir / "model.s00001.safetensors"),
            {f"{unpacked_name}.weight": w_ref},
        )
        _write_index(
            str(unpacked_dir / "model.safetensors.index.json"),
            {f"{unpacked_name}.weight": "model.s00001.safetensors"},
        )
        return {
            "packed_dir": str(packed_dir),
            "unpacked_dir": str(unpacked_dir),
            "packed_name": packed_name,
        }

    def test_enumerates_24_mappings(self):
        mappings = ab.enumerate_code_mappings()
        # 4 codes choose 3 positions * 3! permutations = 4 * 6 = 24
        assert len(mappings) == 24

    def test_canonical_mapping_in_set(self):
        mappings = ab.enumerate_code_mappings()
        found = any(m[0] == -1 and m[1] == 0 and m[2] == 1 for m in mappings)
        assert found

    def test_optimal_mapping_is_canonical(self, tmp_path):
        fx = self._fixture_codes(tmp_path)
        triplets, unpacked_index = ab.load_and_align(fx["packed_dir"], fx["unpacked_dir"])
        result = ab.bruteforce_layer(
            triplets[0],
            fx["packed_dir"],
            fx["unpacked_dir"],
            unpacked_index,
        )
        best = result["best_mapping"]
        assert best[0] == -1
        assert best[1] == 0
        assert best[2] == 1
        assert result["best_mse"] == pytest.approx(0.0, abs=1e-8)
