from __future__ import annotations

import pytest

from edge_cloud.config import EdgeConfig


def test_legacy_config_defaults_to_legacy_handoff() -> None:
    config = EdgeConfig.from_dict({"edge_id": "edge-02", "escalation_threshold": 0.5})

    assert config.edge_id == "edge-02"
    assert config.escalation_threshold == pytest.approx(0.5)
    assert config.handoff_mode == "legacy"
    assert config.handoff_threshold == pytest.approx(0.5)
    assert config.handoff_threshold_by_task == {}
    assert config.max_boundary_wait_tokens == 12
    assert config.allow_prefix_rewrite is False
    assert config.probe_path == ""
    assert config.handoff_policy_version == "v1"
    assert config.cost_per_edge_token == 0.0
    assert config.cost_per_cloud_input_token == 0.0
    assert config.cost_per_cloud_output_token == 0.0
    assert config.cost_per_cloud_call == 0.0
    assert config.cloud_rtt_ms == 0.0
    assert config.max_cost_per_min == 0.0


def test_config_loads_handoff_settings_and_ignores_unknown_fields() -> None:
    config = EdgeConfig.from_dict(
        {
            "handoff_mode": "probe",
            "handoff_threshold": 0.7,
            "handoff_threshold_by_task": {"process_calc": 0.8},
            "max_boundary_wait_tokens": 3,
            "allow_prefix_rewrite": True,
            "probe_path": "runs/handoff/probe.json",
            "handoff_policy_version": "v2",
            "cost_per_edge_token": 0.0001,
            "cost_per_cloud_input_token": 0.0002,
            "cost_per_cloud_output_token": 0.0003,
            "cost_per_cloud_call": 0.01,
            "cloud_rtt_ms": 45.0,
            "max_cost_per_min": 1.5,
            "field_from_the_future": {"ignored": True},
        }
    )

    assert config.handoff_mode == "probe"
    assert config.handoff_threshold == pytest.approx(0.7)
    assert config.handoff_threshold_by_task == {"process_calc": pytest.approx(0.8)}
    assert config.max_boundary_wait_tokens == 3
    assert config.allow_prefix_rewrite is True
    assert config.probe_path == "runs/handoff/probe.json"
    assert config.handoff_policy_version == "v2"
    assert config.cost_per_edge_token == pytest.approx(0.0001)
    assert config.cost_per_cloud_input_token == pytest.approx(0.0002)
    assert config.cost_per_cloud_output_token == pytest.approx(0.0003)
    assert config.cost_per_cloud_call == pytest.approx(0.01)
    assert config.cloud_rtt_ms == pytest.approx(45.0)
    assert config.max_cost_per_min == pytest.approx(1.5)
    assert "field_from_the_future" not in config.to_dict()


def test_config_accepts_every_handoff_mode() -> None:
    for mode in ("off", "legacy", "probe", "oracle"):
        assert EdgeConfig(handoff_mode=mode).handoff_mode == mode


def test_config_rejects_unknown_handoff_mode() -> None:
    with pytest.raises(ValueError):
        EdgeConfig(handoff_mode="auto")
    with pytest.raises(ValueError):
        EdgeConfig.from_dict({"handoff_mode": "auto"})


def test_config_rejects_out_of_range_handoff_values() -> None:
    for bad_threshold in (-0.01, 1.01):
        with pytest.raises(ValueError):
            EdgeConfig(handoff_threshold=bad_threshold)
    with pytest.raises(ValueError):
        EdgeConfig(handoff_threshold_by_task={"process_calc": 1.5})
    with pytest.raises(ValueError):
        EdgeConfig(max_boundary_wait_tokens=-1)
    with pytest.raises(ValueError):
        EdgeConfig(cost_per_edge_token=-0.1)
    with pytest.raises(ValueError):
        EdgeConfig(cost_per_cloud_input_token=-0.1)
    with pytest.raises(ValueError):
        EdgeConfig(cost_per_cloud_output_token=-0.1)
    with pytest.raises(ValueError):
        EdgeConfig(cost_per_cloud_call=-0.1)
    with pytest.raises(ValueError):
        EdgeConfig(cloud_rtt_ms=-1.0)
    with pytest.raises(ValueError):
        EdgeConfig(max_cost_per_min=-1.0)


def test_default_config_roundtrips_without_changing_legacy_fields() -> None:
    config = EdgeConfig()
    restored = EdgeConfig.from_dict(config.to_dict())

    assert restored == config
    assert restored.escalation_threshold == pytest.approx(0.62)
    assert restored.handoff_mode == "legacy"


def test_escalation_threshold_by_task_defaults_empty_and_roundtrips() -> None:
    assert EdgeConfig().escalation_threshold_by_task == {}

    configured = EdgeConfig.from_dict(
        {
            "escalation_threshold": 0.5,
            "escalation_threshold_by_task": {"fault_diag": 0.35, "process_qa": 0.7},
        }
    )
    assert configured.escalation_threshold_by_task == {
        "fault_diag": pytest.approx(0.35),
        "process_qa": pytest.approx(0.7),
    }

    restored = EdgeConfig.from_dict(configured.to_dict())
    assert restored == configured
    assert restored.escalation_threshold_by_task == configured.escalation_threshold_by_task


def test_escalation_threshold_by_task_stringifies_keys_and_numeric_values() -> None:
    config = EdgeConfig(escalation_threshold_by_task={7: 0, "fault_diag": "0.25"})

    assert config.escalation_threshold_by_task == {"7": 0.0, "fault_diag": 0.25}


def test_config_rejects_invalid_escalation_threshold_by_task() -> None:
    for bad_mapping in ([0.2], "process_calc", None, 0.3):
        with pytest.raises(ValueError):
            EdgeConfig(escalation_threshold_by_task=bad_mapping)
    for bad_value in (-0.01, 1.01, "high", None, [0.5]):
        with pytest.raises(ValueError):
            EdgeConfig(escalation_threshold_by_task={"process_calc": bad_value})
