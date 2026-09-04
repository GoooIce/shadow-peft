# Shadow Blind Review 使用说明

这是 Mage-Flow 预生成图像的本地 A/B 盲测应用；它不加载模型、不占用 GB10 GPU，也不会把
人工答案反馈给 ES。

## 本地体验

```bash
PYTHONPATH=src .venv/bin/python scripts/create_annotation_demo.py /tmp/shadow-annotation-demo
cat /tmp/shadow-annotation-demo/demo-access.json
PYTHONPATH=src .venv/bin/python -m shadow_lab.annotation.cli \
  --database /tmp/shadow-annotation-demo/annotation.sqlite3 \
  --assets /tmp/shadow-annotation-demo/assets \
  serve <experiment_id> --port 8765
```

打开 `http://127.0.0.1:8765`，输入 `annotator_token`。令牌只写入当前标签页的
`sessionStorage`；页面没有第三方字体、分析脚本或外链媒体。

快捷键：`1` 左胜、`2` 平局、`3` 右胜、`0` 跳过、`Enter` 确认、`Backspace`
撤销、`H` 指南、`Ctrl/Cmd+S` 显式保存。图像点击看原图，缩放和滚动在两侧同步。

要直接体验复核队列，新建另一个目录并加 `--review-ready`：

```bash
PYTHONPATH=src .venv/bin/python scripts/create_annotation_demo.py \
  /tmp/shadow-annotation-review-demo --review-ready
```

服务启动方式相同，输入 `reviewer_token`。复核队列只收录被标记或有效答案存在分歧的题；页面
说明入队原因，但不显示原始投票。复核完成前状态机拒绝去盲。

## 管理命令

创建用户（令牌仅显示一次）：

```bash
shadow-annotate --database annotations.sqlite3 --assets ./images \
  create-user --name Rater-01 --role annotator
```

导入 manifest 后，用 owner token 冻结。为避免 token 出现在 shell 历史或进程列表，管理命令
只从 `SHADOW_ANNOTATION_TOKEN` 环境变量读取，未设置时安全提示输入。

```bash
shadow-annotate --database annotations.sqlite3 --assets ./images \
  import-manifest manifest.json
SHADOW_ANNOTATION_TOKEN=... shadow-annotate \
  --database annotations.sqlite3 --assets ./images \
  transition <experiment_id> frozen
```

收集结束后顺序推进 `answers_frozen -> revealed`。普通盲态导出永远不包含 arm 或存储路径；
只有状态已到 `revealed` 的实验才能加 `--revealed`：

```bash
SHADOW_ANNOTATION_TOKEN=... shadow-annotate \
  --database annotations.sqlite3 --assets ./images \
  export <experiment_id> --format jsonl --out blind.jsonl
```

`analyze` 直接输出完成率相关计数、skip/tie/左右偏置、标注者重叠一致性、隐藏重复题自洽；
`--revealed` 后还输出 Bradley–Terry log strength 和按 comparison 成组重采样的 paired
bootstrap 95% CI：

```bash
SHADOW_ANNOTATION_TOKEN=... shadow-annotate \
  --database annotations.sqlite3 --assets ./images \
  analyze <experiment_id> --revealed --out analysis.json
```

服务默认且当前只允许监听 localhost。多人内网部署必须另加经评审的 TLS/认证反向代理 adapter，
不能用 `--host 0.0.0.0` 绕过。
