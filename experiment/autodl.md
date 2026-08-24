# AutoDL 实验环境手册（5090 / 北京区，2026-08-24 实测验证）

> 所有端点与结论均用 `.env` 中 `autodl_token` 实测过；官方文档与实测不一致处以本文件为准。

## 常量

| 项 | 值 |
|---|---|
| API HOST | `https://api.autodl.com` |
| 鉴权 | 请求头 `Authorization: <autodl_token>`（**无** Bearer 前缀），token 在仓库 `.env` |
| 现有实例 | `pro-78752d60153f`（`shadow-peft-5090`，见 `.env` 的 `autodl_instance_uuid`）|
| 实例规格 | RTX 5090 32GB（sm_120）/ 25 vCPU / 90GB 内存 / 系统盘+数据盘各 78GB，¥2.88/h |
| GPU 规格ID | 5090-32G 性能型 → `5090-p`；其他型号见[官方附录](https://www.autodl.com/docs/instance_pro_api/)文末 |
| 地区代码 | 北京B区 `beijingDC2`（5090 供货充足）；北京A `beijingDC1`、西北 `westDC3` 等 |
| 推荐镜像（5090） | PyTorch 2.12.1 / py312 / ubuntu22.04 / CUDA 13.0 → `base-image-ee43618uo7`，开机即用无需装环境 |
| 备选镜像 | torch 2.8.0+cu128 → `base-image-ec1e9vdbd3`；Miniconda cu11.8 py310 → `base-image-8hkyyugih5` |
| 实例内环境 | 主实例 `pro-78752d60153f` 用的旧路径：conda env `speft`（python 3.11 + torch 2.7.1+cu128）；**新实例直接用推荐镜像** |

## 操作步骤

### 新建实例（模板，实测可用）

```bash
curl -X POST https://api.autodl.com/api/v1/dev/instance/pro/create \
  -H "Authorization: $TOKEN" -H "Content-Type: application/json" -d '{
    "data_center_list": ["beijingDC2"], "req_gpu_amount": 1,
    "expand_system_disk_by_gb": 0, "gpu_spec_uuid": "5090-p",
    "image_uuid": "base-image-ee43618uo7", "cuda_v_from": 130,
    "instance_name": "shadow-peft-5090"}'
```

完成判据：返回 `Success` + 实例 ID；实测该镜像开机即有 torch 2.12.1+cu130 且 5090 可用（python 在 `/root/miniconda3/bin/`，非登录 shell 用绝对路径）。`cuda_v_from` 跟镜像 CUDA 大版本走（cu13 镜像填 130）。

### 开机

```bash
TOKEN=$(grep '^autodl_token=' .env | cut -d= -f2); ID=$(grep '^autodl_instance_uuid=' .env | cut -d= -f2)
curl -X POST https://api.autodl.com/api/v1/dev/instance/pro/power_on \
  -H "Authorization: $TOKEN" -H "Content-Type: application/json" -d "{\"instance_uuid\":\"$ID\"}"
```

完成判据：`GET https://api.autodl.com/api/v1/dev/instance/pro/status?instance_uuid=$ID` 返回 `running`（开机约 15 秒）。**注意：GET 端点必须用查询参数传 `instance_uuid`，文档示例的"GET + 请求Body"不生效**；POST 端点才用 JSON body。

### 取 SSH 连接信息（密码/端口以快照为准，勿缓存）

```bash
curl "https://api.autodl.com/api/v1/dev/instance/pro/snapshot?instance_uuid=$ID" -H "Authorization: $TOKEN"
# data.ssh_command / data.root_password / data.jupyter_domain
```

实测：同一实例每次开机 SSH 命令与密码不变。本机已装 `sshpass`，可直接 `sshpass -p <pw> ssh -p <port> root@connect.bjb2.seetacloud.com '<cmd>'`。

### 实例内环境检查（完成判据：输出 True + 5090）

```bash
/root/miniconda3/envs/speft/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

跑 shadow-peft 实验还需把仓库同步上去（git clone + pip install -e，或 rsync）并安装项目依赖。

### 关机（实验结束必做，计费按秒停在关机时刻）

```bash
curl -X POST https://api.autodl.com/api/v1/dev/instance/pro/power_off \
  -H "Authorization: $TOKEN" -H "Content-Type: application/json" \
  -d "{\"instance_uuid\":\"$ID\",\"release_gpu\":true}"
```

完成判据：status 返回 `shutdown`。**关机不清数据**（conda 环境、磁盘内容、SSH 密码全部保留）；但**连续关机 60 天实例会被自动释放且数据清空**。

## 省钱规则（Pro 实例，实测）

- **存储费天天收，与开关机无关**：系统盘 30GB 基础容量 ¥0.10/日 + 扩容部分 ¥0.012/日/GB，次日 0 点扣费。主实例（扩容 50GB，共 80GB）= **¥0.70/日**，只有**释放实例**才停止计费。长期不用要么缩需求要么释放。
- **GPU 用量计费按秒停在关机时刻**，不用就关（¥2.88/h）。
- **无卡模式开机 ¥0.10/时**（0.5 核/2GB/无 GPU，最低消费 ¥0.01）：传数据、改代码、网盘传输时用。**只能控制台操作**（实例列表 → 更多 → 无卡模式开机）；dev API 传 `"payload":"non_gpu"` 会报"不支持的启动模式"。每个账号同时仅一个无卡实例。
- **训练完自动关机**：`python train.py; /usr/bin/shutdown`（`;` 无条件关 / `&&` 成功才关），或在代码里 `os.system("/usr/bin/shutdown")`。**先把日志落盘**，关机后 stdout 就没了。
- **定时关机**：控制台实例列表可设置。
- **升降配置**：控制台可改 GPU 数量（调试用 1 卡、正式训练多卡）。
- **连续关机 60 天实例自动释放**（数据清空，控制台有倒计时）。长期闲置前要么开机跑个东西，要么把要紧数据传网盘后主动释放。
- **释放流程**：必须先 `power_off` 等 `shutdown`，再调 `release`（运行中直接 release 会报错）。释放不可逆。

## 已验证的坑与规则（reference）

- **余额单位**：`wallet/balance` 的 `assets` 除以 1000 才是元（毫元，不是分）。价格字段（`payg_price` 等）同理。
- **老版容器 API 已下线**：`POST /api/v1/dev/containers` 返回 404。现行[通用 API](https://www.autodl.com/docs/common_api/) 只有余额 + 专用NFS 两个端点。
- **官方文档附录的镜像表过期**（最高只列到 CUDA 11.8/torch 2.0）：完整最新镜像表在控制台创建页（`https://www.autodl.com/create/pro`），数据源 `POST https://www.autodl.com/api/v1/image/all`（需浏览器登录态 + localStorage 的 `token` 作 Authorization 头；结构是 PyTorch→版本→python→CUDA 四层树，叶子在 `label_name.uuid`）。
- **公共镜像确实有新的**：torch 2.12.1+cu130（实测开机即用）、torch 2.8.0+cu128、torch 2.5.1+cu12.4 等都有，只是没写进 API 文档附录。旧路径（Miniconda 镜像 + 自装 torch）仅当需要特定 torch 版本时才用。
- **镜像自带 conda 源是坏的**：`~/.condarc` 指向已下线的清华 `anaconda/pkgs/free`（404），先 `sed -i "/pkgs\/free/d" ~/.condarc`。pip 已预配阿里云镜像，正常用。
- **学术加速**：实例内 `source /etc/network_turbo` 只对 GitHub/HuggingFace 有效（直连 GitHub 会超时，HF 用 `hf-mirror.com` 也行）；**开了加速反而拖慢 pip**，pip 不要挂 turbo。pytorch.org 直连即有 ~4.7MB/s。
- **网盘（夸克）**：入口在 AutoPanel（JupyterLab 域名下的网页面板，须从 autodl.com 控制台实例行点击进入，直接开 URL 报"无访问权限"）。独立访问密码在 `.env`（`autodl_netdisk_password`，加密所有网盘授权信息，忘了要重配）；夸克授权导出块在仓库根 `.quark_netdisk_auth.txt`（gitignored），新实例 AutoPanel →「导入授权」粘贴即可免重扫。上传/下载均实测可用（下载落 `/root/autodl-tmp`，速度 ~5MB/s 量级，SVIP 账号）。
- **微信通知**：`POST https://www.autodl.com/api/v1/wechat/message/send`，header 同 token，body `{title, name, content}`（title 必填），实测秒达"AutoDL云服务"公众号。限 50 条/天、5 条/分钟。适合实验结束/异常时通知。
- **查货源**：`POST /api/v1/dev/machine/region/gpu_stock` body `{"region_sign":"beijingDC2"}` 返回各 GPU 闲置数（弹性部署 API，同样适用买前探货）。
- **释放实例不可逆**：`power_off` 数据保留；`release` 端点会清空一切，慎用。
- 官方文档：[Pro 实例 API](https://www.autodl.com/docs/instance_pro_api/)、[通用 API](https://www.autodl.com/docs/common_api/)、[弹性部署 API（含地区码）](https://www.autodl.com/docs/esd_api_doc/)、[网盘](https://www.autodl.com/docs/netdisk/)、[省钱](https://www.autodl.com/docs/save_money/)。
