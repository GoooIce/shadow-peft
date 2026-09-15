#!/bin/bash
# pool4 票：v4 两池的 **封条复核（seal check）** GPU 评分 —— 待机执行，本票不运行。
#
# 前置条件：gooo / goooinext 上的 300 代诊断运行必须已经结束（本脚本自带守卫：
# 一旦 pgrep 到 pilot_search 立即 abort，绝不同机竞争）。
# 用途：在**冻结字节**上复算 q0（NVFP4 W4A4）主干臂，证明冻结文件的 prompt 渲染 /
# 停止条件 / 前缀稳定性与父生成池一致（预期准确率 = pool_manifest_v4.json 里的
# pools.*.q0_tf_acc）。这不是难度标定（那已由 v3 在生成池上完成），
# 因此评分结果**不得**用于挑选或重排条目。
#
# 预期成本（gooo GB10，v3 实测吞吐 q0 TF 19,252 tok/s、GEN 1,250 tok/s、载入 37.3 s）：
#   confirm_v4 8,344 题 : 载入 37.3 + TF 24.7 + GEN 22.1 =  84 s
#   centre_v4  2,040 题 : 载入 37.3 + TF  6.3 + GEN  5.4 =  49 s
#   合计                                                  ≈ 133 s ≈ 2.2 min
# 可选（校准中心通道的可分辨性，非必需）：
#   cand_search 在 centre_v4 上 TF     : 载入 37.3 + TF 6.3  =  44 s
#
# 用法（gooo）：
#   rsync 本脚本与冻结池到 ~/minicpm5-pool4/ 后：
#     nohup ./launch_score_v4.sh > /dev/null 2>&1 &
#   结果落在 ~/minicpm5-pool4/out/ ，日志在 ~/minicpm5-pool4/logs/
set -uo pipefail

BASE=$HOME/minicpm5-pool4
V2=$HOME/venvs/minicpm5-vllm
SC=$BASE/scripts/minicpm5
OUT=$BASE/out
LOGS=$BASE/logs
mkdir -p "$OUT" "$LOGS"

# ---- 守卫：诊断运行还活着就 abort（绝不与 GPU 上的搜索竞争） ----
if pgrep -f "pilot_search" > /dev/null 2>&1; then
  echo "ABORT: pilot_search is still running on this machine; refusing to start GPU scoring." \
    | tee "$LOGS/seal_check_v4.log"
  exit 9
fi

export PATH="$V2/bin:$PATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# 被杀进程遗留的 lock 会阻塞后续运行（06/11 票的坑）
rm -f "$HOME"/.cache/flashinfer/*/121a/cached_ops/tmp/*.lock 2>/dev/null

# ---- 0) 封条复核：冻结件的 sha256 必须是清单登记值 ----
{
  echo "[$(date '+%F %T')] === 0) sha256 of frozen v4 pools ==="
  sha256sum "$BASE/taskset_confirm_v4.jsonl" "$BASE/taskset_centre_v4.jsonl"
  echo "[$(date '+%F %T')] expected: confirm_v4 011dec6dcdffd537fa80aecf4462423c06885c733ffd829e15df5054be1d075e"
  echo "[$(date '+%F %T')] expected: centre_v4  4709437d499e988fcc28a14e9010b1b09cbfdb8406ed0a4628e0da8f67b93757"
  sha256sum "$HOME/minicpm5-payload/out/MiniCPM5-2B-NVFP4-W4A4/model.safetensors"
  echo "[$(date '+%F %T')] expected payload sha256: 25feac64b5e5f354e37db611c624b35681d71434cee08a600b9993f57ea24e07"
} >> "$LOGS/seal_check_v4.log" 2>&1

run_score () {  # $1=items file  $2=tag  $3=model  $4=quantization
  echo "[$(date '+%F %T')] === $2 : $1 ===" >> "$LOGS/seal_check_v4.log"
  "$V2/bin/python" "$SC/score_taskset.py" \
      --model "$3" --quantization "$4" \
      --items "$1" --arm both \
      --out "$OUT/scores_$2.jsonl" --summary "$OUT/scores_$2_summary.json" \
      --tag "$2" --linear-backend cutlass \
      >> "$LOGS/seal_check_v4.log" 2>&1
  echo "[$(date '+%F %T')] $2 exit=$?" >> "$LOGS/seal_check_v4.log"
}

run_score "$BASE/taskset_confirm_v4.jsonl" q0_confirm_v4 \
          "$HOME/minicpm5-payload/out/MiniCPM5-2B-NVFP4-W4A4" compressed-tensors
run_score "$BASE/taskset_centre_v4.jsonl"  q0_centre_v4 \
          "$HOME/minicpm5-payload/out/MiniCPM5-2B-NVFP4-W4A4" compressed-tensors

# 可选：中心通道可分辨性校准（近邻候选在同一个中心切片上）
if [ "${WITH_CAND:-0}" = "1" ]; then
  run_score "$BASE/taskset_centre_v4.jsonl" cand_search_centre_v4 \
            "$HOME/minicpm5-payload/out/cand_search" compressed-tensors
fi

echo "[$(date '+%F %T')] seal check done" >> "$LOGS/seal_check_v4.log"
