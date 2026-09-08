"""JSON boundaries for authenticated coordinator/worker delivery."""

from dataclasses import replace

from .contracts import (
    CandidateRequest,
    DataManifest,
    preflight,
)


def read_manifest(value):
    return DataManifest.from_dict(value)


def read_request(value):
    return CandidateRequest.from_dict(value)


def validate_worker_grant(grant, lease, manifest, *, device_id, now, cleanup_seconds=30):
    """Revalidate metadata and the authenticated capability; do not mint a lease."""
    request = read_request(grant["request"])
    receipt = preflight(
        replace(request, lease_id=None), manifest, expected_center_version=request.center_version
    )
    if receipt.status != "ready_for_reservation":
        raise ValueError(receipt.to_dict())
    if (
        grant["status"] != "authorized"
        or grant["request_digest"] != request.request_digest
        or request.lease_id != lease.lease_id
        or request.device_id != lease.device_id
        or request.run_id != lease.run_id
        or request.ticket_id != lease.ticket_id
        or lease.device_id != device_id
    ):
        raise ValueError("coordinator grant/lease mismatch")
    if not lease.started_at <= now < lease.hard_deadline - cleanup_seconds:
        raise ValueError("worker lease expired or not started")
    return request
