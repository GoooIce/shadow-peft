"""4-bit generation demo server for LFM2.5-2.6B-Base (six columns, port 8768).

    full     — full-precision 2.6B base
    quant    — w4a16 damaged base only
    n1x8     — 1-layer 10752-wide shadow, native nvfp8 weights, 10M tokens (CHAMPION)
    n2x8     — 1-layer 21504-wide (2x) shadow, native nvfp8, 10M tokens
    b1x      — 1-layer 10752-wide shadow, bf16 weights, 10M tokens
    n1x8_30m — champion recipe with 30M tokens (shows data overfit under grid reg)

ppl numbers and trainable params are read from each arm's verdict.json /
train/summary.json at startup.

    ssh goooinext 'cd ~/Code/shadow-peft && nohup .venv/bin/python \
      scripts/serve_generation_demo_2p6b.py > runs/logs/gen-demo-2p6b.log 2>&1 < /dev/null &'
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

MODEL_PATH = str(Path.home() / "models" / "LFM2.5-2.6B-Base")
MAX_NEW = 128
MAX_PROMPT_CHARS = 600

# (id, title, ckpt dir or None, verdict path or None, card css class)
COLUMNS = [
    ("full", "全精度基座(2.6B)", None, None, "bf"),
    ("n1x8", "1× 原生 nvfp8 @10M(冠军)", "runs/exp1/floor2_w4a16_2p6b_L1_1x_nvfp8_10m",
     "runs/exp1/floor2_w4a16_2p6b_L1_1x_nvfp8_10m/verdict.json", "win"),
    ("n2x8", "2× 原生 nvfp8 @10M", "runs/exp1/floor2_w4a16_2p6b_L1dw_nvfp8",
     "runs/exp1/floor2_w4a16_2p6b_L1dw_nvfp8/verdict.json", "win"),
    ("b1x", "1× bf16 影子 @10M", "runs/exp1/floor2_w4a16_2p6b_L1_1x",
     "runs/exp1/floor2_w4a16_2p6b_L1_1x/verdict.json", ""),
    ("n1x8_30m", "1× 原生 nvfp8 @30M(数据过量)", "runs/exp1/floor2_w4a16_2p6b_L1_1x_nvfp8_30m",
     "runs/exp1/floor2_w4a16_2p6b_L1_1x_nvfp8_30m/verdict.json", ""),
    ("quant", "4-bit 不修补", None, None, "q"),
]

PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>2.6B 4-bit 真实生成测试:影子配方终选现场</title><style>
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
<h1>2.6B 4-bit 档真实生成测试:影子配方终选现场</h1>
<div class="sub">六列并排 · 困惑度:全精度 __PPL_BF16__ / 冠军 __PPL_N1X8__ / 2×nvfp8 __PPL_N2X8__ / 1×bf16 __PPL_B1X__ / 冠军配方@30M __PPL_N1X830__ / 不修补 __PPL_QUANT__ · 逐词生成、无缓存全前缀重算 · 温度 0 为贪心</div>
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
<div class="cols">__CARDS__</div>
<details class="method"><summary>方法说明:4-bit 量化与影子蒸馏恢复(2.6B 终选配方)</summary>
<div class="mcols">
<div class="mblock"><h4>① 量化方法(w4a16,权重 4-bit / 激活 bf16)</h4>
<ul>
<li>仅量化<b>权重</b>:4-bit 非对称仿射,每 128 输入通道一组 fp32 scale/zero;激活与矩阵乘走 bf16。</li>
<li>量化所有输入维可被 128 整除的 Linear 权重;embedding 查表不量化;权重约压 4×。</li>
<li>本测试台跑 <b>fake-quant</b> 形态(权重反量化回 bf16 数值但保留量化误差);部署可切 tinygemm 原生 int4 kernel。</li>
</ul></div>
<div class="mblock"><h4>② 恢复过程与冠军配方</h4>
<ul>
<li>量化主干冻结,旁挂影子变换器堆蒸馏(教师 = bf16 全精度,CE + KL + 影子辅助 0.05,教师自采样池)。</li>
<li><b>原生 nvfp8 影子</b>:影子权重从初始化起就生活在 nvfp8 网格上(e4m3 元素 + 块 16 e4m3 尺度 + 全局尺度),每步优化后投影回网格——无 bf16 主权重,部署权重 = 训练权重(验证偏差 0.0)。</li>
<li>冠军 = 1 层全宽(10752)+ 原生 nvfp8 + 10M tokens:恢复 __REC_N1X8__%,影子仅 __PARAMS_N1X8__ 参数(nvfp8 约 86MB)。</li>
<li>对照发现:更宽(2×/4×)更差、bf16 更差、数据加到 30M 反而差——网格正则 + 小影子 + 短预算是 2.6B 的最优组合。</li>
</ul></div>
</div></details>
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


def _read_json(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _meta(ckpt: str | None) -> str:
    if not ckpt:
        return ""
    v = _read_json(f"{ckpt}/verdict.json")
    s = _read_json(f"{ckpt}/train/summary.json")
    ppl = v.get("ppl_shadow")
    n = s.get("trainable_params")
    rec = v.get("recovery")
    parts = []
    if isinstance(ppl, (int, float)):
        parts.append(f"{ppl:.2f}")
    if isinstance(n, (int, float)):
        parts.append(f"{n / 1e6:.1f}M 参数")
    if isinstance(rec, (int, float)):
        parts.append(f"恢复 {rec * 100:.1f}%")
    return " · ".join(parts) if parts else "待训完"


def build_page() -> str:
    champ = _read_json("runs/exp1/floor2_w4a16_2p6b_L1_1x_nvfp8_10m/verdict.json")
    cs = _read_json("runs/exp1/floor2_w4a16_2p6b_L1_1x_nvfp8_10m/train/summary.json")
    vq = _read_json("runs/exp1/floor2_w4a16_2p6b_L1dw_nvfp8/verdict.json")
    quant_v = _read_json("runs/exp1/floor2_w4a16_2p6b_L1_1x/verdict.json")

    def fmt(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

    def pct(v):
        return f"{v * 100:.1f}" if isinstance(v, (int, float)) else "—"

    cards, ids = [], []
    for cid, title, ckpt, _vp, css in COLUMNS:
        ids.append(cid)
        if cid == "full":
            meta = f"参照系 · {fmt(quant_v.get('ppl_bf16'))}"
        elif cid == "quant":
            meta = f"受损基座 · {fmt(quant_v.get('ppl_quant'))}"
        else:
            meta = _meta(ckpt)
        cards.append(
            f'<div class="card {css}"><h3>{title}</h3><div class="meta">{meta}</div>'
            f'<div class="out" id="o_{cid}">—</div><div class="t" id="t_{cid}"></div></div>')
    page = PAGE.replace("__CARDS__", "\n".join(cards)).replace("__IDS__", ",".join(f"'{i}'" for i in ids))
    for token, value in {
        "__PPL_BF16__": fmt(quant_v.get("ppl_bf16")),
        "__PPL_QUANT__": fmt(quant_v.get("ppl_quant")),
        "__PPL_N1X8__": fmt(champ.get("ppl_shadow")),
        "__PPL_N2X8__": fmt(vq.get("ppl_shadow")),
        "__PPL_B1X__": fmt(_read_json("runs/exp1/floor2_w4a16_2p6b_L1_1x/verdict.json").get("ppl_shadow")),
        "__PPL_N1X830__": fmt(_read_json("runs/exp1/floor2_w4a16_2p6b_L1_1x_nvfp8_30m/verdict.json").get("ppl_shadow")),
        "__REC_N1X8__": pct(champ.get("recovery")),
        "__PARAMS_N1X8__": f"{cs['trainable_params'] / 1e6:.1f}M" if "trainable_params" in cs else "?",
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

    models = {"full": plain(full), "quant": plain(quant)}
    for cid, _t, ckpt, _v, _c in COLUMNS:
        if ckpt is None:
            continue
        peft = ShadowPeftModel.from_pretrained(quantized_base(), ckpt + "/train/final").eval()
        models[cid] = adapter(peft)
    return tok, bos, models


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
    print("ready on http://0.0.0.0:8768", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8768), Handler).serve_forever()


if __name__ == "__main__":
    main()
