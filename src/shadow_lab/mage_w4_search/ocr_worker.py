"""One frozen OCR engine over registered development images under a worker lease.

This worker emits observations only. The coordinator must verify its stop receipt,
meter the full lease, and compare both engines with external human transcription
before calling compare_ocr/freeze_scorer. It cannot mint a scoring lock itself.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from .budget import DeviceLease
from .contracts import digest_json
from .execution import read_manifest, validate_worker_grant
from .ocr import CalibrationPlan, recognize
from .scorers import SubprocessOCRAdapter, decode_ocr_config


def validate_comparison_job(job, request):
    """Bind the ordered image bytes and generation receipts before any OCR call."""
    if request.ticket_id != "04" or request.purpose != "ocr_selection":
        raise ValueError("only ticket 04 OCR development requests are allowed")
    if request.runtime_identity.get("ocr_comparison_job_sha256") != digest_json(job):
        raise ValueError("OCR job identity changed after coordinator authorization")
    if job.get("schema") != "mage-w4-ocr-worker-v1" or job.get("config_index") not in (0, 1):
        raise ValueError("unknown OCR worker job or engine index")
    raw = dict(job["plan"])
    raw["configs"] = tuple(decode_ocr_config(value) for value in raw["configs"])
    raw["image_sha256"] = tuple(raw["image_sha256"])
    raw["generation_receipt_sha256"] = tuple(raw["generation_receipt_sha256"])
    plan = CalibrationPlan(**raw)
    if plan.sha256 != job["plan_sha256"]:
        raise ValueError("comparison plan identity mismatch")
    images = job["images"]
    if (
        tuple(row["sha256"] for row in images) != plan.image_sha256
        or tuple(row["generation_report_sha256"] for row in images)
        != plan.generation_receipt_sha256
        or tuple(row["sample_id"] for row in images)
        != tuple(row.sample_id for row in request.samples)
    ):
        raise ValueError("OCR images must exactly match the registered request and plan")
    verified = []
    for row in images:
        data = Path(row["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("OCR image SHA mismatch")
        report_bytes = Path(row["generation_report_path"]).read_bytes()
        if hashlib.sha256(report_bytes).hexdigest() != row["generation_report_sha256"]:
            raise ValueError("generation report SHA mismatch")
        report = json.loads(report_bytes)
        matching = [item for item in report["rows"] if item["sample_id"] == row["sample_id"]]
        if (
            report.get("qualified") is not True
            or report.get("center_sha256") != request.q0_digest
            or len(matching) != 1
            or matching[0].get("image_file_sha256") != row["sha256"]
        ):
            raise ValueError("OCR selection requires the registered native q0 image")
        verified.append(data)
    return plan, tuple(verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device-id", choices=("gooo", "goooinext"), required=True)
    args = parser.parse_args()
    directory = args.run_root / "jobs" / args.job_id
    output = directory / "ocr-observations.json"
    if output.exists() or any(directory.glob("ocr-observation-*.json")):
        raise ValueError(
            "OCR observations already exist; recover them instead of recognizing again"
        )
    manifest = read_manifest(json.loads((args.run_root / "data-v1/manifest.json").read_text()))
    lease = DeviceLease(**json.loads((directory / "lease.json").read_text()))
    grant = json.loads((directory / "grant.json").read_text())
    request = validate_worker_grant(
        grant, lease, manifest, device_id=args.device_id, now=time.time()
    )
    job = json.loads((directory / "ocr-job.json").read_text())
    plan, images = validate_comparison_job(job, request)
    config = plan.configs[job["config_index"]]
    adapter = SubprocessOCRAdapter(
        config, job["model_root"], python_executable=job["python_executable"]
    )
    rows = []
    # Materialize all image bytes before loading a model, then preserve every
    # observation (including engine failures). Partial files remain auditable.
    for index, data in enumerate(images):
        receipt = recognize(data, config, adapter)
        row = {"observation": asdict(receipt), "child": adapter.last_receipt}
        (directory / f"ocr-observation-{index:04d}.json").write_text(
            json.dumps(row, indent=2) + "\n"
        )
        rows.append(row)
    result = {
        "schema": "mage-w4-ocr-observations-v1",
        "request_digest": request.request_digest,
        "lease_id": lease.lease_id,
        "device_id": args.device_id,
        "plan_sha256": plan.sha256,
        "config_sha256": config.sha256,
        "job_sha256": digest_json(job),
        "rows": rows,
        "cost_basis": "pending_coordinator_verified_lease_stop",
        "scorer_frozen": False,
    }
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "observations": len(rows),
                "failures": sum(row["observation"]["status"] != "ok" for row in rows),
                "scorer_frozen": False,
            }
        )
    )


if __name__ == "__main__":
    main()
