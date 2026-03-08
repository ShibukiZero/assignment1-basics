# 当前状态总览（2026-03-08）

> 本页只保留“现在在哪一步、下一步做什么”。
> 详细历史与分模块 bootstrap 已移到 `notes/archive/`。

## 1) 当前阶段
- 当前本地分支：`codex/EXPERIMENTS`
- 当前远程硬件上下文：已切换到 `H800`，不再按 Blackwell / `sm_120` 兼容性问题处理。
- 核心实现状态：Chapter 4-6 core path is already in place.
- 当前工作重点：Chapter 7 logging infra 已开始落地，随后进入 TinyStories experiments。

## 2) 现在最该读什么
- handout 硬规则：`handout_hard_rules_2026-03-08.md`
- Chapter 7 原文任务单：`chapter7_requirements_2026-03-08.md`
- 活跃 handoff：`chapter4_7_active_handoff_2026-03-08.md`
- 长期规范：`assignment_workflow_and_rules.md`
- 代码风格硬约束：`transformer_style_guardrails.md`
- 历史摘要：`archive_digest_2026-03-08.md`
- 远程路径：`remote_paths_2026-03-08.md`
- 远程证据：`../logs/terminal.log`

## 3) 下一步
1. 不在当前新环境上继续追 tokenizer speed：
   - `test_train_bpe_speed` 对 CPU 性能敏感，当前环境整体 pytest 用时也显著变慢
   - Chapter 7 主线不依赖在这台机器上继续做 BPE 训练
2. Chapter 7 logging smoke on H800 passed file-level verification:
   - `config.json` exists and captures run/model/optimization metadata
   - `metrics.jsonl` includes `step`, `wallclock_seconds`, `tokens_seen`, `split`, `loss`, `perplexity`, `learning_rate`
   - `summary.json` records final wallclock and best val checkpoint info
   - TensorBoard event file exists under `/root/tf-logs/ch7_logging_smoke`
3. Logging layout was then refined:
   - checkpoints stay in data-disk `--output-dir`
   - small logs now belong in repo-local `.agents/logs/<run_name>/`
4. Chapter 7.1 infrastructure is now sufficient for baseline experiments.
5. Next:
   - use canonical in-repo tracker: `artifacts/experiments/chapter7/experiment_log.md`
   - 开始 Chapter 7 experiment log
   - 跑 TinyStories 17M baseline / LR sweep

## 4) 仍需记住的风险
- 低精度 CLI 暴露范围大于当前数值实现把握。
- 旧笔记里若出现 `codex/Training`，视为历史分支名，不是当前事实。
- 跨设备 checkpoint restore 要留意 `map_location`。
