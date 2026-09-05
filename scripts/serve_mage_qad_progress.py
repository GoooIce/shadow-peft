#!/usr/bin/env python3
"""Read-only LAN progress page for an EXP-008 training campaign."""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-008 · 正式训练进度</title><style>:root{color-scheme:dark;font-family:system-ui,"PingFang SC",sans-serif;background:#11151a;color:#edf2f7}main{max-width:950px;margin:auto;padding:40px 24px}h1{font-size:30px;font-weight:550}p{line-height:1.7;color:#aab6c1}.panel{background:#1a2028;border:1px solid #35404a;border-radius:14px;padding:24px;margin:20px 0}.meta{color:#b4e4d5;font-size:13px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.value{font-size:28px;margin-top:10px}.bar{height:6px;background:#35404a;border-radius:8px;margin-top:18px}.bar span{display:block;background:#b4e4d5;height:100%;border-radius:8px}a{color:#b4e4d5}pre{white-space:pre-wrap;font:14px/1.8 system-ui}@media(max-width:600px){.grid{grid-template-columns:1fr}.value{font-size:23px}}</style>
<main><div class="meta">EXP-008 / FULL TRAINING</div><h1>174 层，图像＋文本教师监督</h1><p>从原始模型重新训练 · 1024 条训练提示词 / 128 条独立验证提示词<br>4 步去噪 · 512×512 · 文本损失初始贡献 10%，校准后固定</p>
<section class="panel"><div class="meta" id="state">正在读取状态…</div><h2 id="phase">准备中</h2><div class="grid"><div>已更新步数<div class="value" id="steps">—</div></div><div>当前阶段<div class="value" id="stage">—</div></div><div>峰值显存<div class="value" id="memory">—</div></div></div><div class="bar"><span id="bar"></span></div><p id="detail"></p></section>
<section class="panel"><h2>训练与验证</h2><pre id="metrics">首次更新后显示。</pre><p>训练损失包含文本辅助项；最佳检查点只由图像教师验证误差决定。</p></section>
<section class="panel"><h2>阶段安排</h2><p>冒烟 1 步 → 预检 32 步 → 试训 256 步 → 正式阶段最多 1024 步。每 16 步及阶段结束验证；正式阶段连续 128 步无改善则提前停止。</p><p>训练完成后自动进入一次性质量门；通过后准备人工盲测。人工评审前不会开始 ES。</p></section>
<p id="updated"></p><p><a href="http://192.168.1.177:8778">小实验盲投结果</a> · <a href="http://192.168.1.177:8780">原模型参考图</a></p></main>
<script>const $=id=>document.getElementById(id),phaseNames={calibrating_text_supervision:'校准文本辅助损失',preparing_trajectory:'准备轨迹',forward_backward:'前向与反向计算',optimizer_step:'更新权重',qad_val:'验证图像教师误差',saving_checkpoint:'保存检查点'},stages={smoke:'冒烟',preflight:'预检',pilot:'试训',formal:'正式训练'};
async function poll(){try{const r=await fetch('/status.json'),s=await r.json(),l=s.live||{},p=s.progress||{},m=s.last_step||{};const done=l.completed_optimizer_steps??p.step??0;const state=s.status||s.monitor_status||'准备中';$('state').textContent=({running:'运行中',failed:'已停止：检查失败',selected:'已选定最佳检查点',awaiting_human:'等待人工评审',monitor_error:'监控连接异常'}[state]||state);$('phase').textContent=phaseNames[l.phase]||s.monitor_phase||'加载模型与验证身份';$('steps').textContent=`${done} / 1313`;$('stage').textContent=stages[l.stage||s.stage]||'准备';$('memory').textContent=l.cuda_peak_gib?`${l.cuda_peak_gib.toFixed(1)} GiB`:'—';$('bar').style.width=`${done/1313*100}%`;$('detail').textContent=s.error||((l.phase==='qad_val')?`本次验证 ${l.completed||0} / ${l.total||128} 条`:'全部 174 层参与训练；阶段门通过后自动继续。');const parts=[];if(m.metrics){parts.push(`最近更新：第 ${m.step} 步\n图像训练 MSE：${m.metrics.image_mse.toFixed(6)}\n文本训练 MSE：${m.metrics.text_mse?.toExponential(4)??'—'}\n总训练损失：${m.metrics.objective.toFixed(6)}\n有限梯度 / 显著性覆盖：${m.gradient_layers} / ${m.saliency_layers} 层`)}if(p.best_loss!=null)parts.push(`最佳图像验证 MSE：${p.best_loss.toFixed(6)}（第 ${p.best_step} 步）`);if(s.text_supervision)parts.push(`固定文本系数：${s.text_supervision.coefficient.toExponential(5)}`);$('metrics').textContent=parts.join('\n\n')||'首次更新后显示。';$('updated').textContent=`页面更新：${new Date(s.now*1000).toLocaleTimeString()} · 训练状态：${l.updated_at?new Date(l.updated_at*1000).toLocaleTimeString():'等待首条记录'} · 每 10 秒刷新`}catch(e){$('updated').textContent='正在重新连接状态服务…'}setTimeout(poll,10000)}poll();</script></html>"""


def effective_status(campaign, monitor):
    if monitor.get("status") in {"failed", "monitor_error", "awaiting_human"}:
        return monitor["status"]
    return campaign.get("status") or monitor.get("status")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8781)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text()) if path.is_file() else {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                payload, kind = PAGE.encode(), "text/html; charset=utf-8"
            elif self.path == "/status.json":
                campaign = read(args.campaign / "campaign.json")
                monitor = read(args.monitor / "status.json")
                progress = campaign.get("progress", {})
                payload = json.dumps(
                    dict(
                        now=time.time(),
                    status=effective_status(campaign, monitor),
                        stage=campaign.get("stage"),
                        error=campaign.get("error") or monitor.get("error"),
                        monitor_status=monitor.get("status"),
                        monitor_phase=monitor.get("phase"),
                        progress={
                            k: progress.get(k) for k in ("step", "best_loss", "best_step", "stages")
                        },
                        live=read(args.campaign / "live.json"),
                        last_step=read(args.campaign / "last-step.json"),
                        text_supervision=read(args.campaign / "text-supervision.json"),
                    )
                ).encode()
                kind = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
