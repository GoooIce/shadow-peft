# Mage-Flow-Turbo ES×QUASAR NVFP4 三代 smoke（GB10，2026-09-04）

## 结论

Phase 1 整模接缝通过：31,145,408 维逐组 clip genome、QUASAR-grid 初始化、antithetic
seed replay、逐候选完整 NVFP4 W4A4 物化、4 步生成、HPS-v2.1/OpenCLIP 奖励、ES 更新与
原子 checkpoint 在 GB10 上连续运行三代。12/12 个候选成功，三代奖励均非全同，首候选的
完整重放逐位一致。

本 smoke **不证明质量改善**。`N=4, sigma=0.03, alpha=0.015` 的更新过激：候选平均重构
误差从约 38k 增至 72k、120k，中心 clip 的单步 RMSE 为 0.148、0.190、0.183。正式搜索前
必须独立校准 sigma 和 alpha；不得把这条轨迹当成主实验结果。

## 固定配置

- Hardware：`gooo`，NVIDIA GB10，compute capability 12.1。
- Torch：`2.13.0+cu130`；TorchAO：`0.18.0`；Mage Flow：`0.1.0`。
- Model：ModelScope `microsoft/Mage-Flow-Turbo` 固定资产，MMDiT Linear 174/174 层。
- Quantization：原生 NVFP4 W4A4，重构 group=128；Qwen3-VL 与 VAE 本票保持原精度。
- Search：31,145,408 维，antithetic `N=4`，`sigma=0.03`，SGD `alpha=0.015`，z-score。
- Episode：每代 4 个互不重复的 `es_train` prompt，每图固定 seed，512×512，4 steps。
- Reward：冻结 BF16 paired calibration，HPS-v2.1 与 OpenCLIP H/14 等权 z-score。

命令由 `scripts/launch_mage_flow_es_smoke_gb10.sh` 在独立 `setsid` 会话中启动；运行目录为
`~/runs/mage-flow/es-nvfp4-smoke-n4-p4-g3-20260904`。

## 初始化

- CPU BF16 Linear master：4,114,612,224 个权重；捕获 44.37s。
- QUASAR 15 点 grid：64.48s；重构误差 37,662.887。
- QUASAR clip 分布：`0.80: 105`、`0.85: 57,084`、`0.90: 1,841,022`、
  `0.95: 11,957,855`、`1.00: 18,289,342`。
- 有界 logistic 的有效初始化把 18,289,342 个 `c=1.00` 端点显式投影为约 `0.999`；
  最大位移 0.00100005。原始 grid 与有效 ES 初始化是两个分别记录的基线，不能混称相同。

## 三代结果

| 代 | 成功/总数 | reward mean | reward std | reward max | 平均重构误差 | clip 单步 RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4/4 | -4.4789 | 0.1097 | -4.2988 | 38,043.49 | 0.1475 |
| 1 | 4/4 | -4.9011 | 0.0500 | -4.8439 | 72,197.42 | 0.1905 |
| 2 | 4/4 | -4.4876 | 0.0222 | -4.4549 | 120,473.17 | 0.1829 |

各代使用不同 prompt batch；reward 由逐样本 BF16 paired baseline 标准化，但本表仍只用于
控制流与信号验收。重构误差跨代可直接比较，清楚显示更新幅度失控。

有效中心 clip 从初始化的均值 0.9748、std 0.0302，三次更新后变为均值 0.8461、std
0.2464；`c<0.8` 的比例从 0 增至 25.69%。因此后续 alpha 不能直接沿用 `sigma/2`。

## 重放、冻结与内存证据

- 首候选以同 genome/noise seed、prompt seed 重做一次；clip hash、重构误差、4 个 PNG
  SHA-256、两个奖励轴和聚合奖励全部精确相同。
- 每个成功候选都经过 `trainable_parameters_after == 0` 运行时门禁；Mage pipeline 在搜索前
  全量 `requires_grad_(False)`，没有 optimizer 或反向传播路径。
- 进程峰值 RSS 31,139,069,952 bytes（约 29.0 GiB）。内存中只保留单个候选量化模型、
  CPU BF16 master、一个 center 和当前 noise/clip，不常驻整个种群。
- checkpoint、每代摘要与候选前缀均原子提交；实验身份绑定模型目录、prompt、calibration、
  HPS/CLIP checkpoint。后续版本额外绑定 Torch 精确版本及实现源码 hash，因为跨 Torch
  版本不保证 `randn(seed)` 序列相同。

独立验证器：

```bash
PYTHONPATH=src ~/Code/Mage/mage_flow/.venv/bin/python \
  scripts/validate_mage_flow_es_run.py \
  --require-replay-audit \
  ~/runs/mage-flow/es-nvfp4-smoke-n4-p4-g3-20260904
```

结果：`valid=true`，3 generations，12 candidates，0 failures，全部代有奖励信号；最终
checkpoint SHA-256 为
`1bdbcbf07c3593d8c1cf55c7173a18a6248cd095d3645e749ab6d9e2987a4012`。

本地镜像：`runs-local/mage-flow/es-nvfp4-smoke-n4-p4-g3-20260904/`。

## 资源冲突记录

最初误在 `goooinext` 启动时，统一内存检查发现用户既有 LTX-2.5 服务常驻约 52 GB。
本项目进程在加载奖励模型前被安全终止，未触碰该服务；中间目录可恢复地改名为
`es-nvfp4-smoke-n4-p4-g3-20260904-aborted-shared-memory`。随后通过 GB10 直连把已验证的
Mage 环境复制到 `gooo`，在约 83 GiB 可用内存下完成正式 smoke；`gooo` 上既有 MinerU
使用独立环境，未被修改。

## 下一门禁

先运行只评估不累计更新的 sigma 扫描，再用目标 center clip RMSE 约束 alpha。建议首扫
`sigma ∈ {0.003, 0.01, 0.03, 0.1}`；alpha 至少加入 `{0.0003, 0.001, 0.003}`，并记录
latent/clip step RMS。主实验前应以 `N=30` 复核，因为 N=4 的梯度噪声不能外推。
