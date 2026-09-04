"""File-format adapters for canonical AnnotationStore exports."""

from __future__ import annotations

import csv
import io
import json


def to_jsonl(export: dict) -> str:
    header = {
        "record_type": "experiment",
        **export["experiment"],
        "schema_version": export["schema_version"],
        "export_mode": export["export_mode"],
    }
    rows = [json.dumps(header, ensure_ascii=False, sort_keys=True)]
    rows.extend(
        json.dumps(
            {"record_type": "answer_event", **answer},
            ensure_ascii=False,
            sort_keys=True,
        )
        for answer in export["answers"]
    )
    rows.extend(
        json.dumps(
            {"record_type": "review_event", **review},
            ensure_ascii=False,
            sort_keys=True,
        )
        for review in export.get("reviews", [])
    )
    return "\n".join(rows) + "\n"


def to_csv(export: dict) -> str:
    fields = (
        "record_type",
        "event_id",
        "assignment_id",
        "review_assignment_id",
        "comparison_id",
        "prompt_id",
        "seed",
        "annotator_id",
        "reviewer_id",
        "event_type",
        "presentation_decision",
        "canonical_decision",
        "reason_codes",
        "reason_target",
        "flagged",
        "received_at",
        "supersedes_event_id",
        "is_superseded",
        "qc_kind",
        "qc_group_id",
        "comment",
        "left_arm_id",
        "right_arm_id",
        "arm_decision",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for answer in export["answers"]:
        row = dict(answer)
        row["record_type"] = "answer_event"
        row["reason_codes"] = "|".join(answer["reason_codes"])
        row.update(answer.get("reveal", {}))
        writer.writerow(row)
    for review in export.get("reviews", []):
        row = dict(review)
        row["record_type"] = "review_event"
        row.update(review.get("reveal", {}))
        writer.writerow(row)
    return buffer.getvalue()
