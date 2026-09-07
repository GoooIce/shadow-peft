# GB10 双机操作手册(gooo / goooinext)

两台局域网 DGX Spark(GB10,128GB 统一内存),shadow-peft 的**默认实验环境**——一切训练、评测、数据生成都在双机执行;Mac 只做代码可行性测试(语法检查、单元级冒烟),不跑实验;AutoDL 属云端备选,使用前须用户批准,见 `docs/autodl.md`。本手册只记环境查不到的坑与纪律;文件布局用 `ls` 看,旗标用 `--help` 看。

## 主机与互联

- `ssh goooinext` = dc4a(局域网 192.168.1.177);`ssh gooo`(局域网 192.168.1.197)。两台 GPU 空闲与否用 `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader`。
- **双机大流量互联尽量走光纤直连**:用户明确要求优先用 SFP+ 光纤口直连两台 GB10,而不是经交换机的铜口(2026-09-07 实测 87GB 检查点 rsync 走 enP7s7 仅协商 2500Mb/s 铜口,约 280MB/s)。传大权重前先确认路径落在光纤口上,不在则提醒用户插线/配置。
- **dc4a 可直接 `ssh gooo`**:双机间传模型/池/权重一律走直连 rsync(如 `ssh goooinext 'rsync -a ~/models/X/ gooo:models/X/'`),绝不经 Mac 中转。传完必对 md5(模型文件、数据池都验)。
- 两台可同时跑各自训练 + 测试台(128GB 显存充裕);训练吞吐会互相无关,同机混跑只有轻微损耗。

## 代码同步(本地 Mac 工作区是唯一事实源)

- repo 在各自 `~/Code/shadow-peft`,venv 一致(`.venv/bin/python`,shadow_lab 以 editable 指向 repo src)。
- **gooo 不是 git 仓库**(rsync 拷贝);dc4a 是 git 仓库但落后于本地未提交改动。发训练前把本地 `src/` 与 `scripts/` rsync 到目标机(不带 `--delete`,exclude `__pycache__`),保证两台跑同一份代码。
- 绝不把 `.env`、`.quark_netdisk_auth.txt`、`.scratch/` 同步到远端。

## 长任务纪律(全部实战踩过)

- **本地链式命令不可靠**:`ssh A 'cmd1 && sleep && cmd2'` 的本地包装会被工具后台化导致后半截丢失。顺序任务写成**远端 bash 链脚本**再 nohup 整个脚本。
- **nohup 要三重重定向**(`> log 2>&1 < /dev/null`),且 dc4a 上观察到 ssh 会话收尾会连带清算 nohup 子进程——**长跑一律走 setsid 启动器**:小的 `launch_xxx.sh`(`exec setsid .venv/bin/python ... 三重重定向`),然后 `nohup launch_xxx.sh`。
- **pkill 两个坑**:①模式会匹配 ssh 包装自身命令行,用 `[x]` 中括号技巧;②**远端命令的文本里含模式串也会自杀**(heredoc 写的启动脚本里含脚本名时尤其如此)——kill 与 launch 永远分开两条 ssh。

## 发训练与验收

- runner:`scripts/exp1_floor_w4a16.py`(`--model/--data/--shadow-layers/--shadow-intermediate/--shadow-heads/--shadow-kv-heads/--shadow-quant/--lr/--max-tokens/--out-dir`)。
- 启动 2 分钟后 grep 日志验收:基线 ppl 跨机逐位一致、trainable params / teacher sequences / initial PPL 与预期配置相符。
- **verdict.json 是唯一判定来源**(落 `<out-dir>/verdict.json`),训完即 rsync 回本地 `runs-local/` 对应目录备份;训练轨迹在 `<out-dir>/train/train_log.jsonl`(每次 "w" 模式重写,无 resume)。
- 教师池:`python -m shadow_lab.teacher_data --model <m> --out <p> --num-seq 20000 --seq-len 512 --batch 128 --seed 301`(2.6B 约 2750 tok/s,单机 20000 条 ~1 小时;可双机各生成一半不同 seed 再 cat 拼合,拼后 md5 对齐)。`make_batch` 有放回采样,**池大小不会阻塞训练**。

## 测试台(标准库 http 服务,权重在内存,setsid 常驻)

- 端口约定:8766 = gooo 350M 四列;8767 = dc4a 1.2B 四列;8768 = dc4a 2.6B 六列。用户从 Mac 访问 `http://192.168.1.177:<port>` / `http://192.168.1.197:<port>`。
- 重启口径:`pkill -f "serve_generation_demo_[x]"` + `nohup launch_xxx_bench.sh`(分开两条 ssh)。
