"""4-bit generation demo for the exp1-native4b 30M shadows (six columns).

    --preset base (port 8769, raw continuation prompts, gooo)
        full    — full-precision LFM2.5-2.6B-Base
        quant   — w4a16 damaged base only
        b0_10m  — bf16 shadow @10M  (grace_B0 anchor, 10.7816)
        q0_10m  — native-nvfp4 shadow @10M (grace_Q0, 10.7137 all-time champion)
        b0_30m  — bf16 shadow @30M  (n4b_B0_30m, 10.7605)
        q0_30m  — native-nvfp4 shadow @30M (n4b_Q0_30m, 10.7564)

    --preset rl (port 8768, chat-template prompts, dc4a)
        full    — full-precision LFM2.5-2.6B (post-trained)
        quant   — w4a16 damaged base only
        q0_10m  — native-nvfp4 @10M (axis2 4.2385, all-time rl champion)
        b0_30m  — bf16 @30M (axis2 4.4821)
        q0_30m  — native-nvfp4 @30M (axis2 4.2835)
        ptq_30m — the b0_30m checkpoint PTQ-swapped to nvfp4 (axis2 4.4706)

    --preset rl1b (port 6006, Pro6000 box, chat-template prompts)
        full    — full-precision LFM2.5-2.6B (post-trained)
        b0_1b   — bf16 shadow @1B (axis2 3.8997, all-time rl champion)
        ptq_1b  — the b0_1b checkpoint PTQ-swapped to nvfp4 (axis2 3.8990)
        b0_100m — bf16 @100M lr3e-5 (axis2 3.9307, gate100 winner)
        quant   — w4a16 damaged base only

rl-preset card numbers read the AXIS2 verdicts (the wiki axis is distorted on
this model); prompts go through the chat template (the model opens a <think>
block — reasoning traces in the output are on-distribution). Column ppl and
params are read from verdict.json / train/summary.json at startup.

    ssh <host> 'cd ~/Code/shadow-peft && nohup .venv/bin/python \
      scripts/serve_generation_demo_n4b.py --preset rl > runs/logs/gen-demo-n4b.log 2>&1 < /dev/null &'
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab import tiers
from shadow_peft import ShadowPeftModel

MAX_NEW = 1024
MAX_PROMPT_CHARS = 600

PRESETS = {
    "base": {
        "model": str(Path.home() / "models" / "LFM2.5-2.6B-Base"),
        "port": 8769,
        "chat": False,
        "title": "2.6B-Base 4-bit 生成测试：30M 影子效果对比（exp1-native4b）",
        "presets": [
            "The capital of France is",
            "In 1969, the first humans landed on",
            "Water boils at a temperature of",
            "Once upon a time in a small village,",
        ],
        # (id, title, ckpt dir or None, verdict path for the card, css)
        "columns": [
            ("full", "全精度基座", None, None, "bf"),
            ("q0_10m", "Q0 原生 nvfp4 @10M（全历史冠军）", "runs/exp1/grace_Q0",
             "runs/exp1/grace_Q0/verdict.json", "win"),
            ("b0_30m", "B0 bf16 影子 @30M", "runs/exp1/n4b_B0_30m",
             "runs/exp1/n4b_B0_30m/verdict.json", ""),
            ("q0_30m", "Q0 原生 nvfp4 @30M", "runs/exp1/n4b_Q0_30m",
             "runs/exp1/n4b_Q0_30m/verdict.json", "win"),
            ("quant", "4-bit 不修补", None, None, "q"),
        ],
        "axis_note": "despaced wikitext ppl",
    },
    "rl": {
        "model": str(Path.home() / "models" / "LFM2.5-2.6B"),
        "port": 8768,
        "chat": True,
        "title": "LFM2.5-2.6B(RL 后训练) 4-bit 生成测试：30M 影子效果对比（exp1-native4b）",
        "presets": [
            "用一句话解释什么是模型量化",
            "Write a haiku about mountains.",
            "9.11 和 9.9 哪个大？为什么？",
            "给一个 Python 函数：判断回文字符串",
        ],
        "columns": [
            ("full", "全精度基座(RL)", None, None, "bf"),
            ("q0_10m", "Q0 原生 nvfp4 @10M（axis2 冠军）", "runs/exp1/grace-rl_Q0",
             "runs/exp1/grace-rl-axis2_Q0/verdict.json", "win"),
            ("b0_30m", "B0 bf16 影子 @30M", "runs/exp1/n4b-rl_B0_30m",
             "runs/exp1/n4b-rl-axis2_B0_30m/verdict.json", ""),
            ("q0_30m", "Q0 原生 nvfp4 @30M", "runs/exp1/n4b-rl_Q0_30m",
             "runs/exp1/n4b-rl-axis2_Q0_30m/verdict.json", "win"),
            ("ptq_30m", "B0_30m 训后换 nvfp4(PTQ-swap)", "runs/exp1/n4b-rl_B0_30m",
             "runs/exp1/n4b-rl-axis2_B0_30m_ptq/verdict.json", ""),
            ("s_only_10m", "影子单独推理 Q0@10M（仅 345M）", "runs/exp1/grace-rl_Q0",
             "runs/exp1/grace-rl-axis2_Q0/verdict.json", "solo"),
            ("s_only_30m", "影子单独推理 Q0@30M（仅 345M）", "runs/exp1/n4b-rl_Q0_30m",
             "runs/exp1/n4b-rl-axis2_Q0_30m/verdict.json", "solo"),
            ("quant", "4-bit 不修补", None, None, "q"),
        ],
        "axis_note": "留出自蒸馏池(axis2) ppl — wiki 轴在此模型上失真，仅作过程记录",
    },
    "rl1b": {
        "model": str(Path.home() / "models" / "LFM2.5-2.6B"),
        "port": 6006,
        "chat": True,
        "title": "LFM2.5-2.6B(RL) 4-bit 生成测试：100M → 1B 剂量对比（exp1-rl1b）",
        "presets": [
            "用一句话解释什么是模型量化",
            "Write a haiku about mountains.",
            "9.11 和 9.9 哪个大？为什么？",
            "给一个 Python 函数：判断回文字符串",
        ],
        "columns": [
            ("full", "全精度基座(RL)", None, None, "bf"),
            ("b0_1b", "B0 bf16 影子 @1B（新王）", "runs/exp1/rl1b_B0_1b",
             "runs/exp1/rl1b-axis2_B0_1b/verdict.json", "win"),
            ("ptq_1b", "B0_1b 训后换 nvfp4(部署形态)", "runs/exp1/rl1b_B0_1b",
             "runs/exp1/rl1b-axis2_B0_1b_ptq/verdict.json", ""),
            ("b0_100m", "B0 bf16 影子 @100M(gate100 胜者)", "runs/exp1/rl1b_B0_100m_lr3",
             "runs/exp1/rl1b-axis2-100m_B0_100m_lr3/verdict.json", ""),
            ("quant", "4-bit 不修补", None, None, "q"),
        ],
        "axis_note": "留出自蒸馏池(axis2) ppl；1B 影子已越过教师本身(recovery≈1.9，轴原点以下)，比较的是绝对 ppl",
    },
}

PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,"PingFang SC",sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.6}
.wrap{max-width:1500px;margin:0 auto}
h1{font-size:21px;margin-bottom:4px}
.sub{color:#94a3b8;font-size:13px;margin-bottom:16px}
textarea{width:100%;height:74px;background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:10px;padding:10px 12px;font-size:14px;resize:vertical}
.row{display:flex;gap:12px;margin:10px 0;flex-wrap:wrap;align-items:center}
label{font-size:13px;color:#94a3b8}
input[type=range]{width:140px}
.val{color:#60a5fa;font-size:13px;min-width:52px}
button{background:#2563eb;color:#fff;border:none;border-radius:10px;padding:10px 26px;font-size:15px;font-weight:600;cursor:pointer}
button:disabled{background:#475569;cursor:wait}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-top:16px}
.card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px;min-height:200px}
.card h3{font-size:13.5px;margin-bottom:2px}
.card .meta{font-size:12px;color:#94a3b8;margin-bottom:8px}
.card .out{font-size:13.5px;white-space:pre-wrap;word-break:break-word;color:#f1f5f9}
.card.bf h3{color:#4ade80}.card.win h3{color:#4ade80}.card.q h3{color:#fbbf24}.card.solo h3{color:#a78bfa}.card.solo{border-color:#7c3aed}
.t{color:#64748b;font-size:12px;margin-top:8px}
.presets{display:flex;gap:8px;flex-wrap:wrap}
.preset{background:#334155;color:#cbd5e1;border-radius:8px;padding:4px 10px;font-size:12px;cursor:pointer}
details.method{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:12px 16px;margin-top:14px;font-size:13px;color:#cbd5e1}
details.method summary{cursor:pointer;color:#60a5fa;font-size:14px;font-weight:600}
details.method ul{padding-left:18px;display:grid;gap:4px;margin-top:6px}
details.method b{color:#e2e8f0}
</style></head><body><div class="wrap">
<h1>__TITLE__</h1>
<div class="sub">__SUB__ · 逐词生成、无缓存全前缀重算 · 温度 0 为贪心 · 耗时随生成长度超线性增长（1024 词全列约需半小时以上），建议先用 64 词__CHATNOTE__</div>
<textarea id="prompt">__DEFAULT__</textarea>
<div class="presets">__PRESETS__</div>
<div class="row">
<label>生成长度 <input type="range" id="n" min="16" max="1024" value="64" oninput="show('n')"><span class="val" id="nv">64</span></label>
<label>温度 <input type="range" id="t" min="0" max="12" value="0" oninput="show('t')"><span class="val" id="tv">0(贪心)</span></label>
<label>前 k 采样 <input type="range" id="k" min="0" max="100" value="0" oninput="show('k')"><span class="val" id="kv">0(全词表)</span></label>
<button id="go" onclick="gen()">生成对比</button>
</div>
<div class="cols">__CARDS__</div>
<details class="method"><summary>方法说明</summary>
<ul>
<li><b>量化(w4a16)</b>：仅权重 4-bit（每 128 输入通道一组 scale/zero），激活 bf16；本台为 fake-quant 形态，部署可切原生 int4 kernel。</li>
<li><b>影子恢复</b>：量化主干冻结，旁挂 345M 影子（L1_5x）自蒸馏恢复；Q0 = 影子权重原生 nvfp4（训练全程在部署格点上，存训一致）。</li>
<li><b>@30M vs @10M</b>：30M 臂出自 exp1-native4b（1B 池生成期间的并跑测试台）；__AXISNOTE__。</li>
</ul></details>
</div></div>
<script>
const IDS=[__IDS__];
function show(id){const v=document.getElementById(id).value;let s=v;if(id==='t'){s=v==0?'0(贪心)':(v/10)}if(id==='k'){s=v==0?'0(全词表)':v}if(id==='n'){s=v}document.getElementById(id+'v').textContent=s}
function setP(e){document.getElementById('prompt').value=e.textContent}
async function gen(){
 const p=document.getElementById('prompt').value.slice(0,600);
 const body={prompt:p,max_new_tokens:+document.getElementById('n').value,
   temperature:+document.getElementById('t').value/10,top_k:+document.getElementById('k').value};
 const b=document.getElementById('go');b.disabled=true;b.textContent='生成中…';
 for(const m of IDS){document.getElementById('t_'+m).textContent='…'}
 try{
  const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json();
  for(const m of IDS){
   document.getElementById('o_'+m).textContent=j[m].text||'(空)';
   document.getElementById('t_'+m).textContent=j[m].seconds.toFixed(1)+' 秒 · '+j[m].tokens+' 词片';
  }
 }catch(e){document.getElementById('o_'+IDS[1]).textContent='请求失败:'+e}
 b.disabled=false;b.textContent='生成对比';
}
</script></body></html>"""


def _read_json(path: str | None) -> dict:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _meta(cid: str, verdict_path: str | None) -> str:
    v = _read_json(verdict_path)
    ppl = v.get("ppl_shadow")
    rec = v.get("recovery")
    parts = [f"{ppl:.4f}" if isinstance(ppl, (int, float)) else "—"]
    if isinstance(rec, (int, float)):
        parts.append(f"recovery {rec:.3f}")
    return " · ".join(parts)


def build_page(cfg) -> str:
    cards, ids, ppls = [], [], []
    for cid, title, _ckpt, verdict_path, css in cfg["columns"]:
        ids.append(cid)
        if cid == "full":
            any_v = next((_read_json(vp) for _c, _t, _k, vp, _s in cfg["columns"] if vp), {})
            meta = f"参照系 · ppl_bf16 {any_v.get('ppl_bf16', '—')}"
        elif cid == "quant":
            any_v = next((_read_json(vp) for _c, _t, _k, vp, _s in cfg["columns"] if vp), {})
            meta = f"受损基座 · ppl_quant {any_v.get('ppl_quant', '—')}"
        else:
            meta = _meta(cid, verdict_path)
            if cid.startswith("s_only"):
                meta = "shadow_only（卡上 ppl 为双路系统的 axis2 值，仅作参照） · " + meta
        ppls.append(f"{cid} {meta}")
        cards.append(
            f'<div class="card {css}"><h3>{title}</h3><div class="meta">{meta}</div>'
            f'<div class="out" id="o_{cid}">—</div><div class="t" id="t_{cid}"></div></div>')
    page = (
        PAGE.replace("__TITLE__", cfg["title"])
        .replace("__SUB__", " · ".join(ppls))
        .replace("__DEFAULT__", cfg["presets"][0])
        .replace("__PRESETS__", "".join(
            f'<span class="preset" onclick="setP(this)">{p}</span>' for p in cfg["presets"]))
        .replace("__CARDS__", "\n".join(cards))
        .replace("__IDS__", ",".join(f"'{i}'" for i in ids))
        .replace("__AXISNOTE__", cfg["axis_note"])
        .replace("__CHATNOTE__", " · chat 模板输入，<think> 推理痕迹为该模型分布内行为" if cfg["chat"] else "")
    )
    return page


def quantized_base(model_path: str):
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    tiers.apply_training_conversion(base, "w4a16", torch.device("cuda"), fake_quant=True)
    return base.to("cuda").eval()


def build_models(cfg):
    from shadow_lab import shadow_quant

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    full = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16).to("cuda").eval()

    def plain(model):
        def f(ids):
            with torch.no_grad():
                return model(input_ids=ids).logits
        return f

    def adapter(peft):
        def f(ids):
            with torch.no_grad():
                return peft(input_ids=ids, use_cache=False, past_key_values=None).logits
        return f

    models = {"full": plain(full), "quant": plain(quantized_base(cfg["model"]))}
    for cid, _t, ckpt, _v, _c in cfg["columns"]:
        if ckpt is None or cid in models:
            continue
        peft = ShadowPeftModel.from_pretrained(quantized_base(cfg["model"]), ckpt + "/train/final").eval()
        if cid.startswith("ptq"):
            n = shadow_quant.project_linears(shadow_quant.collect_shadow_linears(peft), "nvfp4")
            print(f"PTQ-swap column {cid}: {n} shadow linears projected to nvfp4", flush=True)
        if cid.startswith("s_only"):
            # shadow-only inference: the shadow trunk scores with its frozen
            # lm_head copy — the base path never runs
            from shadow_peft import ShadowForCausalLM

            wrapper = ShadowForCausalLM(peft, inference_mode="shadow_only").eval()
            models[cid] = adapter(wrapper)
        else:
            models[cid] = adapter(peft)
    return tok, models


TOK, MODELS, CHAT = None, None, False
PAGE_RENDERED = ""


def encode_prompt(prompt: str) -> torch.Tensor:
    if CHAT:
        # two-step (string then encode): apply_chat_template's tokenized
        # return type differs across transformers versions
        text = TOK.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        ids = TOK(text, add_special_tokens=False).input_ids
    else:
        ids = TOK(prompt, add_special_tokens=False).input_ids
        if TOK.bos_token_id is not None:
            ids = [TOK.bos_token_id] + ids
    return torch.tensor([ids], device="cuda")


def generate(logits_fn, prompt: str, n: int, temp: float, top_k: int) -> tuple[str, int]:
    ids = encode_prompt(prompt)
    eos = TOK.eos_token_id
    produced = []
    for _ in range(n):
        lg = logits_fn(ids)[0, -1].float()
        if temp <= 0:
            nxt = int(lg.argmax())
        elif top_k and top_k > 0:
            k = min(top_k, lg.size(-1))
            vals, idx = lg.topk(k)
            probs = torch.softmax(vals / temp, dim=-1)
            nxt = int(idx[torch.multinomial(probs, 1)])
        else:
            nxt = int(torch.multinomial(torch.softmax(lg / temp, -1), 1))
        if eos is not None and nxt == eos:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        produced.append(nxt)
    return TOK.decode(produced), len(produced)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE_RENDERED.encode())

    def do_POST(self):
        if self.path != "/generate":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = str(body.get("prompt", ""))[:MAX_PROMPT_CHARS]
        n = max(8, min(int(body.get("max_new_tokens", 64)), MAX_NEW))
        temp = max(0.0, min(float(body.get("temperature", 0.0)), 2.0))
        top_k = max(0, min(int(body.get("top_k", 0)), 1000))
        result = {}
        for name, fn in MODELS.items():
            t0 = time.perf_counter()
            text, count = generate(fn, prompt, n, temp, top_k)
            result[name] = {"text": text, "tokens": count,
                            "seconds": round(time.perf_counter() - t0, 2)}
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def main() -> None:
    global TOK, MODELS, CHAT, PAGE_RENDERED
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    port = args.port or cfg["port"]
    CHAT = cfg["chat"]

    PAGE_RENDERED = build_page(cfg)
    print("loading models…", flush=True)
    TOK, MODELS = build_models(cfg)
    for fn in MODELS.values():
        fn(torch.tensor([[TOK.bos_token_id or 1]], device="cuda"))
    print(f"ready on http://0.0.0.0:{port}", flush=True)

    class V6Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6

    try:  # dual-stack bind: platform gateways may reach the container over v6
        httpd = V6Server(("::", port), Handler)
        print(f"dual-stack bound on [::]:{port}", flush=True)
    except OSError:
        httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
