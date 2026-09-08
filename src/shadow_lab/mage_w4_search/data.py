"""Freeze pilot metadata without opening confirmation or final image evidence."""

from __future__ import annotations

import unicodedata
from collections import Counter

from .contracts import DataManifest, PromptRecord, digest_json

# Curated inventories are made exhaustive by an explicit development-only suffix.
# They are desired strings, NEVER a transcription of pixels or OCR calibration truth.
TARGETS = {
    "aa5fb8be14f900e588a2740c": ("00:45",),
    "a06326840e773f186ce82bf2": ("Delivery Pick-up 取餐处",),
    "973c59359e2ea9e075d8262f": ("健康咨询 Health Desk",),
    "15772ad619891aa655941ec6": ("CHOICE / 选择",),
    "ef15f6590fe6cc185a0a0304": ("Medium Roast：中度烘焙",),
    "c6382032931b1b1feab6d19e": ("归还处 Returns",),
    "663301e5042fe8b3b65b572c": ("请节约用水 Save Water",),
    "db05caf0c1903d56c3ceeb00": ("Eau de Nuit", "Limited Edition"),
    "b4e8de2afe09e1595751c3c8": ("Linen 亚麻",),
    "cf87cddbc291d78f48ccc7cf": ("Bamboo / 竹编",),
    "b8c483cadbded829c862c635": ("Mercury 汞",),
    "ddb8c00a606cb86de999314c": ("猫毛",),
    "5f2928db3e7e662d5deaca2a": ("晚安",),
    "fc8ef2b3eb599f3625a93285": ("Exploring Egypt",),
}
USES = ("ocr_selection", "scale_calibration", "candidate_development")

RESERVE_FAMILIES = {
    "confirmation": (
        (
            "botanical-cabinet",
            "a botanical cabinet with pressed ferns and brass specimen drawers",
            "FERN 27",
            "蕨叶",
        ),
        (
            "mountain-lift",
            "an empty cable car above a snowy mountain pass at dawn",
            "SUMMIT",
            "山顶",
        ),
        (
            "museum-orrery",
            "a brass mechanical orrery inside a quiet museum gallery",
            "ORBIT",
            "轨道",
        ),
        (
            "tea-ceremony",
            "a tea ceremony with a stone basin, bamboo ladle and green teacups",
            "TEA HOUSE",
            "茶室",
        ),
        (
            "dockyard-crane",
            "a dockyard crane lifting a yellow fishing boat at low tide",
            "DOCK 4",
            "船坞",
        ),
        (
            "jellyfish-sub",
            "a research submersible surrounded by translucent jellyfish",
            "DIVE 12",
            "深潜",
        ),
        (
            "ceramic-workshop",
            "a ceramic workshop with a potter shaping a wide red clay bowl",
            "KILN",
            "陶窑",
        ),
        (
            "lunar-greenhouse",
            "a lunar greenhouse with tomato vines under a star-filled window",
            "LUNA",
            "月球",
        ),
    ),
    "final": (
        (
            "basket-market",
            "a woven basket market shaded by bright fabric awnings",
            "BASKETS",
            "竹篮",
        ),
        (
            "arctic-station",
            "an arctic weather station with frost-covered antennas at blue hour",
            "POLAR 8",
            "极地",
        ),
        (
            "jazz-cafe",
            "a street jazz cafe with a saxophonist beside an orange window",
            "JAZZ NIGHT",
            "爵士",
        ),
        (
            "rainforest-bridge",
            "a rope bridge crossing a misty rainforest canyon",
            "TRAIL 9",
            "步道",
        ),
        (
            "glass-studio",
            "a glassblowing studio with a glowing vessel on a steel pipe",
            "GLASS",
            "琉璃",
        ),
        (
            "old-observatory",
            "a medieval observatory with a wooden telescope and celestial globe",
            "NORTH STAR",
            "北极星",
        ),
        (
            "solar-desert",
            "a desert solar farm with rows of mirrored dishes under a copper sky",
            "SOLAR",
            "日光",
        ),
        (
            "underwater-archive",
            "an underwater archive with sealed bookshelves behind thick glass",
            "ARCHIVE",
            "档案",
        ),
    ),
}


def _canonical(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _shingles(text):
    value = _canonical(text)
    return {value[i : i + 5] for i in range(max(1, len(value) - 4))}


def freeze_pilot_manifest(rows: list[dict]) -> tuple[DataManifest, dict]:
    """Group known provenance and high lexical overlap before assigning dev uses.

    The 64 newly authored reserve prompts are a metadata reserve, not a powered
    formal sample-size choice. This narrow curated population is reported as such.
    """
    if not rows or any(not row.get("sample_id") or not row.get("prompt") for row in rows):
        raise ValueError("development rows require sample_id and prompt")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate development sample_id")
    parent = list(range(len(rows)))

    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    shingles = [_shingles(row["prompt"]) for row in rows]
    links = []
    for i, row in enumerate(rows):
        for j in range(i):
            other = rows[j]
            same_source = bool(row.get("source_id")) and (row.get("source"), row["source_id"]) == (
                other.get("source"),
                other.get("source_id"),
            )
            overlap = len(shingles[i] & shingles[j]) / max(1, len(shingles[i] | shingles[j]))
            if same_source or overlap >= 0.8:
                parent[find(i)] = find(j)
                links.append(
                    [
                        row["sample_id"],
                        other["sample_id"],
                        "source" if same_source else "lexical>=0.8",
                    ]
                )
    members = {}
    for i, row in enumerate(rows):
        members.setdefault(find(i), []).append(row["sample_id"])
    group_ids = {key: "old-dev/" + digest_json(sorted(ids)) for key, ids in members.items()}
    # Spread applicable OCR groups round-robin, then nontext groups independently.
    target_groups = sorted(key for key, ids in members.items() if set(ids) & TARGETS.keys())
    other_groups = sorted(set(members) - set(target_groups))
    assignments = {
        group: USES[i % len(USES)]
        for sequence in (target_groups, other_groups)
        for i, group in enumerate(sequence)
    }
    records = []
    derived = []
    for i, row in enumerate(rows):
        targets = TARGETS.get(row["sample_id"], ())
        prompt = row["prompt"]
        if targets:
            prompt += "\nAll text in the image must consist of exactly these text segments: "
            prompt += " | ".join(repr(t) for t in targets) + ". Include no other text."
            derived.append(row["sample_id"])
        records.append(
            PromptRecord(
                row["sample_id"],
                prompt,
                group_ids[find(i)],
                "development",
                (assignments[find(i)],),
                targets,
                bool(targets),
            )
        )
    for pool, families in RESERVE_FAMILIES.items():
        for family, scene, english, chinese in families:
            for variant, composition in enumerate(
                (
                    "A wide documentary photograph",
                    "A close-up painted illustration",
                    "An editorial poster",
                    "A miniature diorama photograph",
                )
            ):
                targets = (english,) if variant == 2 else (chinese,) if variant == 3 else ()
                prompt = (
                    f"{composition} of {scene}. Clear spatial relationships and careful lighting."
                )
                if targets:
                    prompt += f" One clearly visible label must read exactly {targets[0]!r}. No other text anywhere."
                else:
                    prompt += " No lettering or numbers anywhere."
                records.append(
                    PromptRecord(
                        f"reserve-{pool}-{family}-{variant}",
                        prompt,
                        f"authored-20260909/{family}",
                        pool,
                        (),
                        targets,
                        bool(targets),
                    )
                )
    manifest = DataManifest(tuple(records))
    errors = manifest.validate()
    if errors:
        raise ValueError(errors)
    return manifest, {
        "schema": "mage-pilot-data-freeze/v1",
        "manifest_digest": manifest.digest,
        "old_development_rows": len(rows),
        "derived_exhaustive_target_ids": derived,
        "desired_targets_are_pixel_transcriptions": False,
        "development_uses": dict(
            Counter(r.development_uses[0] for r in records if r.pool == "development")
        ),
        "applicable_ocr_by_use": dict(
            Counter(
                r.development_uses[0] for r in records if r.pool == "development" and r.target_text
            )
        ),
        "source_links": links,
        "near_duplicate_rule": "same upstream source identifier or NFC-casefold five-character-shingle Jaccard >= 0.8; transitive components",
        "limits": [
            "Lexical grouping cannot guarantee absence of semantic near-duplicates.",
            "Reserve is model-assisted authored metadata: 8 independent design families per pool, 4 dependent variants each.",
            "Formal sample sizes, attempt count and power are not frozen; narrow curated population only.",
        ],
        "reserve_groups_per_pool": 8,
        "reserve_prompts_per_pool": 32,
        "sealed_images_generated": 0,
        "sealed_scores_read": 0,
        "formal_sample_sizes_frozen": False,
    }
