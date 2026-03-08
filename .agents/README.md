# .agents 目录说明（精简版）

用途：只保留“下一线程 3-5 分钟内能恢复工作”的信息。

维护原则：
- `notes/` 根目录只放当前仍会被频繁读取的文件。
- 阶段性 bootstrap、评审碎片、历史快照统一移到 `notes/archive/`。
- 远程运行证据统一看 `logs/terminal.log` 与对应子目录。

实验输出策略：
- 默认将实验运行结果、临时报告与分析材料写入 `.agents/logs/`。
- 只有用户明确要求或题目明确要求时，才写入仓库持久目录（如 `artifacts/`）。

## 结构
- `reference/`：作业手册与参考资料（只读）。
- `logs/`：远程运行输出与证据。
- `notes/`：当前阶段入口、长期规则、少量活跃 handoff。
- `notes/archive/`：历史阶段资料与已被总结吸收的碎片笔记。

## 运行规范（本作业）
- 环境与测试遵循仓库 README：优先使用 `uv`。
- 本地不运行 `python` / `pytest` / `uv`；运行由用户在远程环境执行。
- 提交打包：`bash make_submission.sh`。

## 当前推荐入口
- 当前状态总览：`notes/current_status.md`
- Handout 硬规则：`notes/handout_hard_rules_2026-03-08.md`
- Chapter 7 原文任务单：`notes/chapter7_requirements_2026-03-08.md`
- Chapter 7 下一步分析：`notes/chapter7_next_steps_analysis_2026-03-08.md`
- Chapter 7 logging infra 设计：`notes/chapter7_logging_infra_design_2026-03-08.md`
- Chapter 7 TinyStories tuning plan：`notes/chapter7_tinystories_tuning_plan_2026-03-08.md`
- Chapter 7 experiment log：`logs/experiment_log.md`
- Chapter 4-7 活跃 handoff：`notes/chapter4_7_active_handoff_2026-03-08.md`
- 长期规范手册：`notes/assignment_workflow_and_rules.md`
- 历史摘要：`notes/archive_digest_2026-03-08.md`
- Transformer 风格硬约束：`notes/transformer_style_guardrails.md`
- 远程路径记录：`notes/remote_paths_2026-03-08.md`
- 最新远程证据：`logs/terminal.log`

## 当前阶段
- Chapter 4-6 core implementation is in place.
- 当前工作重心已从“补基础模块”转向“短回归 + Chapter 7 experiments”。

## 历史资料
- 旧阶段资料已移到 `notes/archive/`。
- 若需要追溯某个模块的 bootstrap、review 过程或 tokenizer 历史，请先看 `notes/archive/README.md`。
