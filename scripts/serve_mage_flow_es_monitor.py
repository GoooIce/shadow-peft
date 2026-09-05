#!/usr/bin/env python3
"""Serve a read-only LAN dashboard for a Mage Flow ES experiment directory."""

from __future__ import annotations

import argparse
import json
import mimetypes
import posixpath
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


EXPECTED_GRADIENT_EPISODES = 24
EXPECTED_CANDIDATES_PER_EPISODE = 80
IMAGES_PER_EVALUATION = 8
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    # The writer may be appending its last line while we poll it.
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        pass
    return rows


def _episode_status(
    root: Path, episode_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode = root / f"gradient-episode-{episode_index}"
    candidate_paths = sorted(episode.glob("generation-*-candidates.jsonl"))
    rows = [row for path in candidate_paths for row in _read_jsonl(path)]
    run = _read_json(episode / "run.json")
    summary_paths = sorted(episode.glob("generation-*-summary.json"))
    complete = bool(run and int(run.get("completed_generations", 0)) >= 1)
    if not complete and summary_paths:
        complete = True
    state = "done" if complete else "running" if episode.exists() else "pending"
    failures = sum(row.get("failure_reason") is not None for row in rows)
    replays = sum(isinstance(row.get("replay_audit"), dict) for row in rows)
    exact_replays = sum(
        row.get("replay_audit", {}).get("exact") is True
        for row in rows
        if isinstance(row.get("replay_audit"), dict)
    )
    return (
        {
            "index": episode_index,
            "state": state,
            "candidate_rows": len(rows),
            "failures": failures,
            "replay_audits": replays,
            "exact_replay_audits": exact_replays,
        },
        rows,
    )


def _candidate_record(episode_index: int, row: dict[str, Any]) -> dict[str, Any]:
    axes = row.get("axis_means") or {}
    spec = row.get("spec") or {}
    return {
        "episode": episode_index,
        "population_index": spec.get("population_index"),
        "reward": row.get("reward"),
        "hps_v2_1": axes.get("hps_v2_1"),
        "clip_h14": axes.get("clip_h14"),
        "failure_reason": row.get("failure_reason"),
        "exact_replay": (row.get("replay_audit") or {}).get("exact"),
    }


def _center_evaluations(root: Path) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    paths = sorted(root.glob("selection-episode-*.json")) + sorted(
        root.glob("confirmation-episode-*.json")
    )
    for path in paths:
        report = _read_json(path)
        if report is None:
            continue
        episode = report.get("episode") or {}
        records = []
        for record in report.get("records", []):
            axes = record.get("axis_means") or {}
            records.append(
                {
                    "label": record.get("label"),
                    "reward": record.get("reward"),
                    "hps_v2_1": axes.get("hps_v2_1"),
                    "clip_h14": axes.get("clip_h14"),
                }
            )
        evaluations.append(
            {
                "stage": (
                    "selection" if path.name.startswith("selection") else "confirmation"
                ),
                "episode": episode.get("index"),
                "records": records,
            }
        )
    return evaluations


def _gallery_status(gallery_root: Path) -> dict[str, Any]:
    images = []
    if gallery_root.is_dir():
        for path in sorted(gallery_root.rglob("*"), reverse=True):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(gallery_root).as_posix()
            images.append(
                {
                    "name": path.name,
                    "group": path.parent.relative_to(gallery_root).as_posix(),
                    "url": "/gallery/" + urllib.parse.quote(relative, safe="/"),
                }
            )
    return {"image_count": len(images), "images": images[:240]}


def build_status(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    episodes = []
    all_rows: list[tuple[int, dict[str, Any]]] = []
    for index in range(EXPECTED_GRADIENT_EPISODES):
        episode, rows = _episode_status(root, index)
        episodes.append(episode)
        all_rows.extend((index, row) for row in rows)

    replays = sum(episode["replay_audits"] for episode in episodes)
    valid = [item for item in all_rows if item[1].get("reward") is not None]
    latest = [
        _candidate_record(episode, row) for episode, row in reversed(valid[-100:])
    ]
    gradient_report = _read_json(root / "gradient-report.json")
    selection = _read_json(root / "selection-decision.json")
    result = _read_json(root / "result.json")
    completed = sum(episode["state"] == "done" for episode in episodes)
    return {
        "schema_version": 1,
        "stage": (
            "complete"
            if result
            else (
                "confirmation"
                if list(root.glob("confirmation-episode-*.json"))
                else "selection" if gradient_report else "gradient"
            )
        ),
        "gradient": {
            "expected_episodes": EXPECTED_GRADIENT_EPISODES,
            "completed_episodes": completed,
            "expected_candidate_rows": EXPECTED_GRADIENT_EPISODES
            * EXPECTED_CANDIDATES_PER_EPISODE,
            "candidate_rows": len(all_rows),
            "failures": sum(episode["failures"] for episode in episodes),
            "replay_audits": replays,
            "exact_replay_audits": sum(
                episode["exact_replay_audits"] for episode in episodes
            ),
            "generated_images": (len(all_rows) + replays) * IMAGES_PER_EVALUATION,
            "episodes": episodes,
        },
        "latest_candidates": latest,
        "center_evaluations": _center_evaluations(root),
        "selection_decision": selection,
        "result": result,
        "gallery": _gallery_status(root / "gallery"),
    }


def resolve_gallery_path(gallery_root: Path, raw_relative: str) -> Path:
    gallery_root = gallery_root.resolve()
    decoded = urllib.parse.unquote(raw_relative)
    normalized = posixpath.normpath(decoded).lstrip("/")
    candidate = (gallery_root / normalized).resolve()
    if candidate != gallery_root and gallery_root not in candidate.parents:
        raise ValueError("gallery path is outside the controlled root")
    if candidate.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("gallery path does not name a supported image")
    return candidate


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mage Flow ES · 20× 实验监控</title>
<style>
:root{color-scheme:dark;font-family:ui-sans-serif,system-ui;background:#0b1020;color:#e8eefc}body{margin:0;padding:24px;max-width:1440px;margin:auto}h1{font-size:25px;margin:0 0 6px}.muted{color:#9aa8c7}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}.card{background:#151c31;border:1px solid #2a3555;border-radius:12px;padding:14px}.big{font-size:25px;font-weight:700;margin-top:5px}.bar{height:10px;background:#27304b;border-radius:8px;overflow:hidden}.bar i{display:block;height:100%;background:#6aa6ff}.episodes{display:grid;grid-template-columns:repeat(12,1fr);gap:5px}.ep{padding:7px 2px;text-align:center;border-radius:6px;background:#27304b;font-size:12px}.ep.done{background:#17633e}.ep.running{background:#83570a}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:7px;border-bottom:1px solid #27304b;font-size:13px}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}.gallery figure{margin:0;background:#151c31;padding:7px;border-radius:10px}.gallery img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px}.gallery figcaption{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;color:#9aa8c7;margin-top:5px}section{margin-top:26px}@media(max-width:700px){.episodes{grid-template-columns:repeat(6,1fr)}body{padding:14px}}
</style></head><body><h1>Mage Flow ES · 20× 实验监控</h1><div class="muted" id="stamp">连接中…</div>
<div class="grid" id="cards"></div><div class="bar"><i id="progress"></i></div><section><h2>观测 episodes</h2><div class="episodes" id="episodes"></div></section>
<section><h2>最新候选分数</h2><div style="overflow:auto"><table><thead><tr><th>episode</th><th>candidate</th><th>reward</th><th>HPS v2.1</th><th>CLIP H/14</th><th>审计</th></tr></thead><tbody id="scores"></tbody></table></div></section>
<section><h2>选择 / 确认分数</h2><div style="overflow:auto"><table><thead><tr><th>阶段</th><th>episode</th><th>候选</th><th>reward</th><th>HPS v2.1</th><th>CLIP H/14</th></tr></thead><tbody id="centers"></tbody></table></div></section>
<section><h2>实际生成图 <span class="muted" id="galleryCount"></span></h2><div class="gallery" id="gallery"></div></section>
<script>
const fmt=x=>x==null?'—':typeof x==='number'?x.toFixed(6):String(x);const td=(tr,x)=>{const e=document.createElement('td');e.textContent=x;tr.appendChild(e)};
async function update(){try{const s=await(await fetch('/api/status',{cache:'no-store'})).json(),g=s.gradient,p=g.candidate_rows/g.expected_candidate_rows*100;document.getElementById('stamp').textContent=`阶段：${s.stage} · ${new Date().toLocaleString()} · 每 5 秒刷新`;document.getElementById('cards').innerHTML=`<div class=card><div class=muted>候选进度</div><div class=big>${g.candidate_rows} / ${g.expected_candidate_rows}</div></div><div class=card><div class=muted>完成 episodes</div><div class=big>${g.completed_episodes} / ${g.expected_episodes}</div></div><div class=card><div class=muted>已生成图（含审计）</div><div class=big>${g.generated_images}</div></div><div class=card><div class=muted>失败 / 精确审计</div><div class=big>${g.failures} / ${g.exact_replay_audits}</div></div>`;document.getElementById('progress').style.width=`${Math.min(100,p)}%`;const eps=document.getElementById('episodes');eps.replaceChildren();for(const e of g.episodes){const x=document.createElement('div');x.className='ep '+e.state;x.textContent=`${e.index}: ${e.candidate_rows}`;x.title=`${e.state} · failures ${e.failures}`;eps.appendChild(x)}const scores=document.getElementById('scores');scores.replaceChildren();for(const r of s.latest_candidates){const tr=document.createElement('tr');[r.episode,r.population_index,fmt(r.reward),fmt(r.hps_v2_1),fmt(r.clip_h14),r.exact_replay===true?'exact':'—'].forEach(x=>td(tr,x));scores.appendChild(tr)}const centers=document.getElementById('centers');centers.replaceChildren();for(const e of s.center_evaluations)for(const r of e.records){const tr=document.createElement('tr');[e.stage,e.episode,r.label,fmt(r.reward),fmt(r.hps_v2_1),fmt(r.clip_h14)].forEach(x=>td(tr,x));centers.appendChild(tr)}document.getElementById('galleryCount').textContent=`(${s.gallery.image_count} 张)`;const gallery=document.getElementById('gallery');gallery.replaceChildren();for(const r of s.gallery.images){const f=document.createElement('figure'),a=document.createElement('a'),i=document.createElement('img'),c=document.createElement('figcaption');a.href=r.url;a.target='_blank';i.src=r.url;i.loading='lazy';c.textContent=`${r.group} / ${r.name}`;a.appendChild(i);f.append(a,c);gallery.appendChild(f)}}catch(e){document.getElementById('stamp').textContent='读取失败：'+e}}
update();setInterval(update,5000);
</script></body></html>"""


def make_handler(root: Path):
    gallery_root = root / "gallery"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path == "/":
                self._send(HTML.encode(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/status":
                payload = json.dumps(build_status(root), ensure_ascii=False).encode()
                self._send(payload, "application/json; charset=utf-8")
                return
            if parsed.path.startswith("/gallery/"):
                try:
                    path = resolve_gallery_path(
                        gallery_root, parsed.path.removeprefix("/gallery/")
                    )
                except ValueError:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                if not path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send(
                    path.read_bytes(),
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _send(self, payload: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"{self.address_string()} {fmt % args}", flush=True)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8770, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.run_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {root}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(root))
    print(f"serving {root} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
