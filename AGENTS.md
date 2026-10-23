# AGENTS.md

## 语言约定

- 与用户沟通时**始终使用中文**。
- 领域名词**始终用中文定义**（首次出现时可附英文原词，如「影子模型（Shadow Model）」）。
- 代码、标识符、文件路径、git 提交信息保持英文原样。

## 实验环境

默认实验环境 = GB10 双机（`gooo` / `goooinext`）：一切训练、评测、数据生成都发到双机执行。Mac 只用来测试代码可行性（语法检查、单元级冒烟），**不在 Mac 上跑实验**。**用 AutoDL 前须先向用户报批**，未经批准不租用、不开关实例。

在 GB10 双机上发训练、传权重、起测试台前，先读 `docs/gb10-machines.md`（主机互联、代码同步、长任务与 pkill 的坑、验收口径）。获批使用 AutoDL 后，操作细节见 `docs/autodl.md`（实测验证过的平台手册，凭据在 `.env`）。

## 历史实验档案

已结题实验的数据、结论、坑清单与资产地图归档在 `docs/exp7-*.md`（exp7 循环化、exp7b 反向 base/shadow）。开启新实验前先查档：复用语料（`~/data/exp7-corpus/`）、影子权重、评测口径或 benchmark 结论时，以档案为单一事实源；动历史 checkpoint 前先读对应档案的坑清单。

## 浏览器操作

打开网页、交互、截图和 Web 验证一律先读取并只使用 `ego-browser` 技能，按其 CLI 工作流执行。连接异常时按其技能排查；切换其他浏览器工具须由用户明确授权。

## Agent skills

### Issue tracker

Issues and specs live as GitHub issues, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: read `GLOSSARY.md` at the repo root when it exists, and `docs/adr/` for decisions. See `docs/agents/domain.md`.
