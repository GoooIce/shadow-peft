"""Opaque, independently randomized four-way image packets and strict ballots."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

from .mage_qad_campaign import atomic_json
from .mage_qad_checkpoint import content_hash


def build_blind_packet(
    directory: Path,
    arms: dict[str, dict[str, Path]],
    *,
    candidate_sha256: str,
    prompts: dict[str, str] | None = None,
) -> dict:
    from PIL import Image

    if set(arms) != {"bf16", "rtn", "quasar-grid", "qad"}:
        raise ValueError("blind evaluation requires all four arms")
    sample_ids = set(arms["qad"])
    if len(sample_ids) != 30 or any(set(images) != sample_ids for images in arms.values()):
        raise ValueError("blind evaluation requires exactly 30 matching samples in every arm")
    public_dir = directory / "public"
    public_dir.mkdir(parents=True, exist_ok=False)
    trials, public = [], []
    rng = secrets.SystemRandom()
    shuffled_ids = sorted(sample_ids)
    rng.shuffle(shuffled_ids)
    for sid in shuffled_ids:
        trial_id = secrets.token_hex(12)
        names = list(arms)
        rng.shuffle(names)
        labels, images = {}, []
        for label, arm in zip("ABCD", names, strict=True):
            labels[label] = arm
            filename = f"{trial_id}-{label}.png"
            # Re-encode pixels to strip arm/source identifiers in PNG metadata.
            with Image.open(arms[arm][sid]) as image:
                image.convert("RGB").save(public_dir / filename)
            images.append(
                {
                    "label": label,
                    "file": filename,
                    "sha256": hashlib.sha256((public_dir / filename).read_bytes()).hexdigest(),
                }
            )
        trials.append({"trial_id": trial_id, "sample_id": sid, "labels": labels})
        public.append(
            {"trial_id": trial_id, "prompt": prompts[sid] if prompts else "", "images": images}
        )
    atomic_json(public_dir / "packet.json", {"trials": public})
    # A static viewer has no key or candidate name in its served directory.
    (public_dir / "index.html").write_text("""<!doctype html><meta charset="utf-8">
<title>Image comparison</title><style>body{font:16px sans-serif;margin:2em}section{margin-bottom:3em}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1em}img{width:100%}</style>
<h1>Image comparison</h1><p>Rank A–D for each trial and record all visibly broken images.</p>
<div id="trials"></div><script>
fetch('packet.json').then(r=>r.json()).then(p=>p.trials.forEach(t=>{
const s=document.createElement('section');const h=document.createElement('h2');h.textContent=t.trial_id;const p=document.createElement('p');p.textContent=t.prompt;
s.append(h,p);const g=document.createElement('div');g.className='grid';
t.images.forEach(i=>{const f=document.createElement('figure');const c=document.createElement('figcaption');
c.textContent=i.label;const img=document.createElement('img');img.src=i.file;f.append(c,img);g.append(f)});
s.append(g);document.getElementById('trials').append(s)}));</script>""")
    private = {
        "candidate_sha256": candidate_sha256,
        "trials": trials,
        "public_sha256": content_hash({"trials": public}),
    }
    key = directory / "private-key.json"
    atomic_json(key, private)
    key.chmod(0o600)
    return {
        "passed": False,
        "candidate_sha256": candidate_sha256,
        "private_key": str(key.resolve()),
        "private_sha256": content_hash(private),
        "public_directory": str(public_dir.resolve()),
        "public_sha256": private["public_sha256"],
    }


def judge_ballots(private_key: Path, ballots: list[dict], *, final: bool = False) -> dict:
    private = json.loads(private_key.read_text())
    trials = {t["trial_id"]: t for t in private["trials"]}
    if len(ballots) != 30 or len(trials) != 30 or {b["trial_id"] for b in ballots} != set(trials):
        raise ValueError("requires one ballot for each of 30 blinded trials")
    wins = broken = 0
    for ballot in ballots:
        ranking, failures = ballot["ranking"], ballot["broken"]
        if len(ranking) != 4 or set(ranking) != set("ABCD") or not set(failures) <= set("ABCD"):
            raise ValueError("each ballot must rank A-D exactly once and list broken labels")
        labels = trials[ballot["trial_id"]]["labels"]
        qad = next(label for label, arm in labels.items() if arm == "qad")
        static = next(label for label, arm in labels.items() if arm == "quasar-grid")
        broken += qad in failures
        wins += ranking.index(qad) < ranking.index(static)
    return {
        "passed": broken == 0 and (final or wins >= 24),
        "wins": wins,
        "visibly_broken": broken,
        "candidate_sha256": private["candidate_sha256"],
    }
