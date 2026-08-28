"""4-bit generation demo server for LFM2.5-1.2B-Base (four columns, port 8767).

    full   — full-precision 1.2B base
    l2q    — w4a16 + 2-layer quarter-wide shadow @10M (intermediate 12288/4=3072, heads 8/2)
    l1dw   — w4a16 + 1-layer double-wide shadow @10M (intermediate 12288*2=24576)
    quant  — w4a16 damaged base only

    ppl numbers and trainable params are read from each arm's verdict.json /
    train/summary.json at startup, so the page always shows the judged values.

    ssh <host> 'cd ~/Code/shadow-peft && nohup .venv/bin/python \
      scripts/serve_generation_demo_w4a16_1p2b.py > runs/logs/gen-demo-w4a16-1p2b.log 2>&1 < /dev/null &'
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab import tiers
from shadow_peft import ShadowPeftModel

MODEL_PATH = str(Path.home() / "models" / "LFM2.5-1.2B-Base")
CKPT_L2Q = "runs/exp1/floor2_w4a16_1p2b_L2q"
CKPT_L1DW = "runs/exp1/floor2_w4a16_1p2b_L1dw"
MAX_NEW = 128
MAX_PROMPT_CHARS = 600

PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>1.2B 4-bit 真实生成测试:同构两臂</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,"PingFang SC",sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.6}
.wrap{max-width:1400px;margin:0 auto}
h1{font-size:21px;margin-bottom:4px}
.sub{color:#94a3b8;font-size:13px;margin-bottom:16px}
textarea{width:100%;height:74px;background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:10px;padding:10px 12px;font-size:14px;resize:vertical}
.row{display:flex;gap:12px;margin:10px 0;flex-wrap:wrap;align-items:center}
label{font-size:13px;color:#94a3b8}
input[type=range]{width:140px}
.val{color:#60a5fa;font-size:13px;min-width:52px}
button{background:#2563eb;color:#fff;border:none;border-radius:10px;padding:10px 26px;font-size:15px;font-weight:600;cursor:pointer}
button:disabled{background:#475569;cursor:wait}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:16px}
.card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px;min-height:200px}
.card h3{font-size:13.5px;margin-bottom:2px}
.card .meta{font-size:12px;color:#94a3b8;margin-bottom:8px}
.card .out{font-size:13.5px;white-space:pre-wrap;word-break:break-word;color:#f1f5f9}
.card.bf h3{color:#4ade80}.card.win h3{color:#4ade80}.card.q h3{color:#fbbf24}
.t{color:#64748b;font-size:12px;margin-top:8px}
.presets{display:flex;gap:8px;flex-wrap:wrap}
.preset{background:#334155;color:#cbd5e1;border-radius:8px;padding:4px 10px;font-size:12px;cursor:pointer}
details.method{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:12px 16px;margin-top:14px;font-size:13px;color:#cbd5e1}
details.method summary{cursor:pointer;color:#60a5fa;font-size:14px;font-weight:600}
.mcols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-top:10px}
.mblock h4{font-size:13.5px;color:#e2e8f0;margin-bottom:6px}
.mblock ul{padding-left:18px;display:grid;gap:4px}
.mblock b{color:#e2e8f0}
</style></head><body><div class="wrap">
<h1>1.2B 4-bit 档真实生成测试:同构两臂(新实验点)</h1>
<div class="sub">四列并排 · 困惑度:全精度 __PPL_BF16__ / 4-bit+两层窄影子 __PPL_L2Q__ / 4-bit+单层宽影子 __PPL_L1DW__ / 4-bit 不修补 __PPL_QUANT__ · 结构换算自 350M(12288÷4 与 ×2) · 逐词生成、无缓存全前缀重算 · 温度 0 为贪心</div>
<details class="method"><summary>方法说明:4-bit 量化怎么做,精度怎么恢复</summary>
<div class="mcols">
<div class="mblock"><h4>① 量化方法(w4a16,权重 4-bit / 激活 bf16)</h4>
<ul>
<li>仅量化<b>权重</b>:每个 Linear 权重做 4-bit 非对称仿射量化,激活与矩阵乘仍走 bf16。</li>
<li><b>每 128 个输入通道一组</b>:组内共享一对 fp32 的 scale / zero-point,16 级电平(0–15)。</li>
<li>量化所有输入维可被 128 整除的 Linear 权重(注意力投影与 MLP 全在其中;embedding 查表不量化),权重体积约压缩 4×。</li>
<li>实现为 torchao Int4WeightOnly;训练/本测试台跑 <b>fake-quant</b> 形态(权重反量化回 bf16 数值但保留量化误差),部署时可切 tinygemm tile-packed 原生 int4 kernel。</li>
</ul></div>
<div class="mblock"><h4>② 精度恢复(影子蒸馏,唯一可训部分是影子)</h4>
<ul>
<li>量化主干<b>全程冻结</b>;在旁挂一个随机初始化的小型"影子"变换器堆(两层窄影子 __PARAMS_L2Q__ / 单层宽影子 __PARAMS_L1DW__ 参数,宽度按 1.2B 自身中间层 12288 做÷4 与×2),经门控注入参与前向。</li>
<li>教师 = 同款 bf16 全精度模型(冻结)。损失 = CE(对真词)+ KL(对教师 logits,蒸馏权重 1.0)+ 影子辅助项(0.05)。</li>
<li>1.2B 自采样数据池单遍 <b>10M tokens</b>(seq 512 × 8,AdamW lr 1e-5 常数)。</li>
<li>链路:bf16 __PPL_BF16__ → 量化受损 <b>__PPL_QUANT__</b>(+__DAMAGE__)→ 影子恢复 <b>__PPL_L2Q__ / __PPL_L1DW__</b>,吃回 __REC_L2Q__% / __REC_L1DW__% 的损伤,距全精度仅 +__GAP_L2Q__ / +__GAP_L1DW__。</li>
</ul></div>
</div></details>
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
<div class="card bf"><h3>全精度基座(1.2B)</h3><div class="meta">参照系 · __PPL_BF16__</div><div class="out" id="o_full">—</div><div class="t" id="t_full"></div></div>
<div class="card win"><h3>4-bit + 两层窄影子 @10M</h3><div class="meta">__PPL_L2Q__ · __PARAMS_L2Q__ 参数 · 恢复 __REC_L2Q__%</div><div class="out" id="o_l2q">—</div><div class="t" id="t_l2q"></div></div>
<div class="card win"><h3>4-bit + 单层宽影子 @10M</h3><div class="meta">__PPL_L1DW__ · __PARAMS_L1DW__ 参数 · 恢复 __REC_L1DW__%</div><div class="out" id="o_l1dw">—</div><div class="t" id="t_l1dw"></div></div>
<div class="card q"><h3>4-bit 不修补</h3><div class="meta">受损基座 · __PPL_QUANT__</div><div class="out" id="o_quant">—</div><div class="t" id="t_quant"></div></div>
</div></div>
<script>
function show(id){const v=document.getElementById(id).value;let s=v;if(id==='t'){s=v==0?'0(贪心)':(v/10)}if(id==='k'){s=v==0?'0(全词表)':v}if(id==='n'){s=v}document.getElementById(id+'v').textContent=s}
function setP(e){document.getElementById('prompt').value=e.textContent}
async function gen(){
 const p=document.getElementById('prompt').value.slice(0,600);
 const body={prompt:p,max_new_tokens:+document.getElementById('n').value,
   temperature:+document.getElementById('t').value/10,top_k:+document.getElementById('k').value};
 const b=document.getElementById('go');b.disabled=true;b.textContent='生成中…';
 for(const m of ['full','l2q','l1dw','quant']){document.getElementById('t_'+m).textContent='…'}
 try{
  const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json();
  for(const m of ['full','l2q','l1dw','quant']){
   document.getElementById('o_'+m).textContent=j[m].text||'(空)';
   document.getElementById('t_'+m).textContent=j[m].seconds.toFixed(1)+' 秒 · '+j[m].tokens+' 词片';
  }
 }catch(e){document.getElementById('o_l2q').textContent='请求失败:'+e}
 b.disabled=false;b.textContent='生成对比';
}
</script></body></html>"""


def _read_json(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def build_page() -> str:
    vq = _read_json(f"{CKPT_L2Q}/verdict.json")
    vd = _read_json(f"{CKPT_L1DW}/verdict.json")
    sq = _read_json(f"{CKPT_L2Q}/train/summary.json")
    sd = _read_json(f"{CKPT_L1DW}/train/summary.json")

    def fmt(v, digits=2):
        return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "待训完"

    def params(s):
        n = s.get("trainable_params")
        return f"{n / 1e6:.1f}M" if isinstance(n, (int, float)) else "?"

    def pct(v):
        return f"{v * 100:.1f}" if isinstance(v, (int, float)) else "—"

    def diff(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return f"{a - b:+.2f}".replace("+", "")
        return "—"

    page = PAGE
    for token, value in {
        "__PPL_BF16__": fmt(vq.get("ppl_bf16")),
        "__PPL_QUANT__": fmt(vq.get("ppl_quant")),
        "__PPL_L2Q__": fmt(vq.get("ppl_shadow")),
        "__PPL_L1DW__": fmt(vd.get("ppl_shadow")),
        "__PARAMS_L2Q__": params(sq),
        "__PARAMS_L1DW__": params(sd),
        "__REC_L2Q__": pct(vq.get("recovery")),
        "__REC_L1DW__": pct(vd.get("recovery")),
        "__DAMAGE__": diff(vq.get("ppl_quant"), vq.get("ppl_bf16")),
        "__GAP_L2Q__": diff(vq.get("ppl_shadow"), vq.get("ppl_bf16")),
        "__GAP_L1DW__": diff(vd.get("ppl_shadow"), vq.get("ppl_bf16")),
    }.items():
        page = page.replace(token, value)
    return page


def quantized_base():
    cfg_base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
    tiers.apply_training_conversion(cfg_base, "w4a16", torch.device("cuda"), fake_quant=True)
    return cfg_base.to("cuda").eval()


def build_models():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    bos = tok.bos_token_id

    full = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda").eval()
    quant = quantized_base()

    base_l2q = quantized_base()
    peft_l2q = ShadowPeftModel.from_pretrained(base_l2q, CKPT_L2Q + "/train/final").eval()
    base_l1dw = quantized_base()
    peft_l1dw = ShadowPeftModel.from_pretrained(base_l1dw, CKPT_L1DW + "/train/final").eval()

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

    return tok, bos, {
        "full": plain(full),
        "l2q": adapter(peft_l2q),
        "l1dw": adapter(peft_l1dw),
        "quant": plain(quant),
    }


TOK, BOS, MODELS = None, None, None
PAGE_RENDERED = ""


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
    global TOK, BOS, MODELS, PAGE_RENDERED
    PAGE_RENDERED = build_page()
    print("loading models…", flush=True)
    TOK, BOS, MODELS = build_models()
    for fn in MODELS.values():
        fn(torch.tensor([[TOK.bos_token_id or 1]], device="cuda"))
    print("ready on http://0.0.0.0:8767", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8767), Handler).serve_forever()


if __name__ == "__main__":
    main()
