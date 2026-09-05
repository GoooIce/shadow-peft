#!/usr/bin/env python3
"""Publish verified original-model PNGs as a standalone reference gallery."""

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pilot-data", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text())
    expected = json.loads(args.pilot_data.read_text())["validation"]
    samples = run["samples"]
    if len(samples) != len(expected) or len(samples) != 8:
        raise ValueError("expected all eight pilot validation prompts")
    if any(
        run["generation"][k] != v for k, v in dict(height=512, width=512, steps=4, cfg=1).items()
    ):
        raise ValueError("generation settings differ from pilot")
    for row, reference in zip(samples, expected, strict=True):
        if (row["prompt"], row["seed"]) != (reference["prompt"], reference["seed"]):
            raise ValueError("prompt or noise seed differs from pilot")
        if Path(row["image"]).name != row["image"]:
            raise ValueError("image must be a local filename")
        if hashlib.sha256((args.run / row["image"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("image checksum mismatch")
    public = args.run / "public"
    public.mkdir(exist_ok=False)
    cards = []
    for index, row in enumerate(samples):
        filename = f"{index:02d}.png"
        shutil.copyfile(args.run / row["image"], public / filename)
        cards.append(
            f'<article><div class="meta">参考图 {index + 1} · seed {row["seed"]}</div>'
            f'<p>{html.escape(row["prompt"])}</p><img src="{filename}" '
            f'alt="原模型参考图 {index + 1}" tabindex="0"></article>'
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EXP-008 · 原模型参考</title>
<style>:root{color-scheme:dark;font-family:system-ui,"PingFang SC",sans-serif;background:#11151a;color:#edf2f7}
*{box-sizing:border-box}body{margin:0}main{max-width:1160px;margin:auto;padding:36px 24px}h1{font-size:30px;font-weight:550}
.meta{color:#a6c8bd;font-size:13px}header{padding-bottom:24px;border-bottom:1px solid #35404a;margin-bottom:28px}
header p{color:#aab6c1;line-height:1.7}a{color:#b4e4d5}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
article{background:#1a2028;border:1px solid #35404a;border-radius:14px;padding:20px}article p{line-height:1.65;min-height:90px}
article img{display:block;width:100%;border-radius:8px;cursor:zoom-in}dialog{border:0;background:transparent;padding:0;max-width:96vw;max-height:96vh}
dialog::backdrop{background:#000d}dialog img{display:block;max-height:90vh;max-width:94vw}button{position:absolute;right:12px;top:12px;padding:9px 14px;background:#1a2028;color:white;border:1px solid #647080;border-radius:8px;cursor:pointer}
@media(max-width:650px){.grid{grid-template-columns:1fr}main{padding:24px 16px}article p{min-height:0}}</style>
<main><header><div class="meta">EXP-008 / ORIGINAL MODEL</div><h1>原始模型，同一组提示词</h1>
<p>Mage-Flow-Turbo · 原始 BF16 权重 · 4 步去噪 · 512 × 512 · CFG 1<br>
这 8 张图与盲投使用相同提示词和随机种子。点击图片可放大查看。</p>
<a href="http://192.168.1.177:8778">返回匿名盲投 →</a></header><div class="grid">"""
    page += (
        "".join(cards)
        + """</div></main><dialog><img alt="放大查看原模型图片"><button>关闭</button></dialog>
<script>const d=document.querySelector('dialog');for(const img of document.querySelectorAll('article img')){
const open=()=>{d.querySelector('img').src=img.src;d.showModal()};img.onclick=open;img.onkeydown=e=>{if(e.key==='Enter')open()}}
d.querySelector('button').onclick=()=>d.close();d.onclick=e=>{if(e.target===d)d.close()};</script></html>"""
    )
    (public / "index.html").write_text(page)
    print(
        json.dumps(
            dict(images=len(samples), public=str(public), prompt_seed_identity_verified=True)
        )
    )


if __name__ == "__main__":
    main()
