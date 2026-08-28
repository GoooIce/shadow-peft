"""Interactive generation demo server: bf16 base vs 2-bit damaged vs 2-bit + 4-round refinement.

Serves one HTML page + POST /generate on 0.0.0.0:8765. Loads three models:
  full   — full-precision base
  quant  — 2-bit quantized base only (the damaged model)
  shadow — 2-bit base + main shadow adapter + 4-round iterative refinement head
           (checkpoint: floor2_C1r4_lr12e5, ppl 265.3)

Generation is full-prefix recompute (no cache) — simplest correct path for the
refinement head; a 350M model at demo lengths is fast enough.

    ssh goooinext 'cd ~/Code/shadow-peft && nohup .venv/bin/python \
      scripts/serve_generation_demo.py > runs/logs/0827-gen-demo.log 2>&1 < /dev/null &'
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.trainer import TrainConfig, _quantize_base
from shadow_peft import ShadowPeftModel
from shadow_peft.model_utils import build_implicit_shadow_model

MODEL_PATH = str(Path.home() / "models" / "LFM2.5-350M-Base")
CKPT = "runs/exp1/floor2_C1r4_lr12e5/train"
ROUNDS = 4
MAX_NEW = 128
MAX_PROMPT_CHARS = 600

PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>真实生成测试:两比特受损 + 迭代精化影子</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,"PingFang SC",sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.6}
.wrap{max-width:1240px;margin:0 auto}
h1{font-size:21px;margin-bottom:4px}
.sub{color:#94a3b8;font-size:13px;margin-bottom:16px}
textarea{width:100%;height:74px;background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:10px;padding:10px 12px;font-size:14px;resize:vertical}
.row{display:flex;gap:12px;margin:10px 0;flex-wrap:wrap;align-items:center}
label{font-size:13px;color:#94a3b8}
input[type=range]{width:140px}
.val{color:#60a5fa;font-size:13px;min-width:52px}
button{background:#2563eb;color:#fff;border:none;border-radius:10px;padding:10px 26px;font-size:15px;font-weight:600;cursor:pointer}
button:disabled{background:#475569;cursor:wait}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-top:16px}
.card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px;min-height:220px}
.card h3{font-size:14px;margin-bottom:2px}
.card .meta{font-size:12px;color:#94a3b8;margin-bottom:8px}
.card .out{font-size:14px;white-space:pre-wrap;word-break:break-word;color:#f1f5f9}
.card.dmg h3{color:#f87171}.card.bf h3{color:#4ade80}.card.sh h3{color:#60a5fa}
.t{color:#64748b;font-size:12px;margin-top:8px}
.presets{display:flex;gap:8px;flex-wrap:wrap}
.preset{background:#334155;color:#cbd5e1;border-radius:8px;padding:4px 10px;font-size:12px;cursor:pointer}
</style></head><body><div class="wrap">
<h1>真实生成测试:两比特受损基座 × 四轮迭代精化影子修补器</h1>
<div class="sub">三列并排:全精度基座(困惑度 53.5)/ 两比特受损基座(118 万)/ 两比特 + 主影子 + 迭代精化头四轮(265.3,检查点 floor2_C1r4_lr12e5)· 逐词生成、无缓存全前缀重算 · 温度 0 为贪心</div>
<textarea id="prompt">The capital of France is</textarea>
<div class="presets">
<span class="preset" onclick="setP(this)">The capital of France is</span>
<span class="preset" onclick="setP(this)">In 1969, the first humans landed on</span>
<span class="preset" onclick="setP(this)">Water boils at a temperature of</span>
<span class="preset" onclick="setP(this)">Once upon a time in a small village,</span>
</div>
<div class="row">
<label>生成长度 <input type="range" id="n" min="16" max="128" value="64" oninput="show('n')"><span class="val" id="nv">64</span></label>
<label>温度 <input type="range" id="t" min="0" max="12" value="0" oninput="show('t')"><span class="val" id="tv">0(贪心)</span></label>
<label>前 k 采样 <input type="range" id="k" min="0" max="100" value="0" oninput="show('k')"><span class="val" id="kv">0(全词表)</span></label>
<button id="go" onclick="gen()">生成对比</button>
</div>
<div class="cols">
<div class="card bf"><h3>全精度基座</h3><div class="meta">参照系 · 困惑度 53.5</div><div class="out" id="o_full">—</div><div class="t" id="t_full"></div></div>
<div class="card sh"><h3>两比特 + 迭代精化影子 ×4</h3><div class="meta">主角 · 困惑度 265.3 · 参数 2864 万</div><div class="out" id="o_shadow">—</div><div class="t" id="t_shadow"></div></div>
<div class="card dmg"><h3>两比特受损基座(不修补)</h3><div class="meta">损伤现场 · 困惑度 118 万</div><div class="out" id="o_quant">—</div><div class="t" id="t_quant"></div></div>
</div></div>
<script>
function show(id){const v=document.getElementById(id).value;let s=v;if(id==='t'){s=v==0?'0(贪心)':(v/10)}if(id==='k'){s=v==0?'0(全词表)':v}if(id==='n'){s=v}document.getElementById(id+'v').textContent=s}
function setP(e){document.getElementById('prompt').value=e.textContent}
async function gen(){
 const p=document.getElementById('prompt').value.slice(0,600);
 const body={prompt:p,max_new_tokens:+document.getElementById('n').value,
   temperature:+document.getElementById('t').value/10,top_k:+document.getElementById('k').value};
 const b=document.getElementById('go');b.disabled=true;b.textContent='生成中…';
 for(const m of ['full','shadow','quant']){document.getElementById('t_'+m).textContent='…'}
 try{
  const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json();
  for(const m of ['full','shadow','quant']){
   document.getElementById('o_'+m).textContent=j[m].text||'(空)';
   document.getElementById('t_'+m).textContent=j[m].seconds.toFixed(1)+' 秒 · '+j[m].tokens+' 词片';
  }
 }catch(e){document.getElementById('o_shadow').textContent='请求失败:'+e}
 b.disabled=false;b.textContent='生成对比';
}
</script></body></html>"""


def build_models():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    bos = tok.bos_token_id

    full = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda").eval()

    cfg = TrainConfig(bits=2, loss_mode="ce_kl", model=MODEL_PATH, data="unused")
    quant_copy = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=cfg.profile.dtype)
    _quantize_base(quant_copy, cfg, len(tok))
    quant_copy = quant_copy.to("cuda").eval()

    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=cfg.profile.dtype)
    _quantize_base(base, cfg, len(tok))
    base = base.to("cuda").eval()
    peft = ShadowPeftModel.from_pretrained(base, CKPT + "/final").eval()
    refiner = build_implicit_shadow_model(
        base, num_shadow_layers=1,
        shadow_intermediate_size=1664, shadow_num_attention_heads=4,
        shadow_num_key_value_heads=2)
    refiner.load_state_dict(torch.load(CKPT + "/refiner.pt", map_location="cpu"))
    refiner.to(next(base.parameters()).dtype).to("cuda").eval()
    lm_head = base.get_output_embeddings()

    def shadow_logits(ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = peft(input_ids=ids, use_cache=False, past_key_values=None,
                       output_hidden_states=True)
            h = out.hidden_states[-1]
            for _ in range(ROUNDS):
                h = refiner(inputs_embeds=h, attention_mask=None).last_hidden_state
            return lm_head(h)

    def plain_logits(model):
        def f(ids: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return model(input_ids=ids).logits
        return f

    return tok, bos, {
        "full": plain_logits(full),
        "quant": plain_logits(quant_copy),
        "shadow": shadow_logits,
    }


TOK, BOS, MODELS = None, None, None


def encode_prompt(prompt: str) -> torch.Tensor:
    ids = TOK(prompt, add_special_tokens=False).input_ids
    if BOS is not None:
        ids = [BOS] + ids
    return torch.tensor([ids], device="cuda")


def generate(logits_fn, prompt: str, n: int, temp: float, top_k: int) -> tuple[str, int]:
    ids = encode_prompt(prompt)
    eos = TOK.eos_token_id
    produced = []
    for _ in range(n):
        lg = logits_fn(ids)[0, -1].float()
        if temp <= 0:
            nxt = int(lg.argmax())
        else:
            if top_k and top_k > 0:
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
        self.wfile.write(PAGE.encode())

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
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def main() -> None:
    global TOK, BOS, MODELS
    print("loading models…", flush=True)
    TOK, BOS, MODELS = build_models()
    # Warm-up forward for each path (first call pays kernel compile/alloc).
    for fn in MODELS.values():
        fn(torch.tensor([[TOK.bos_token_id or 1]], device="cuda"))
    print("ready on http://0.0.0.0:8765", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8765), Handler).serve_forever()


if __name__ == "__main__":
    main()
