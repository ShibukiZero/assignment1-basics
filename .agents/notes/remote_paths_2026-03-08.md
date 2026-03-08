# 远程环境路径记录（2026-03-08）

## 远程工作目录
- 远程用户家目录：`/root`
- 数据根目录：`/root/autodl-tmp`

## 原始文本数据
- TinyStories train: `/root/autodl-tmp/TinyStoriesV2-GPT4-train.txt`
- TinyStories valid: `/root/autodl-tmp/TinyStoriesV2-GPT4-valid.txt`
- OWT train: `/root/autodl-tmp/owt_train.txt`
- OWT valid: `/root/autodl-tmp/owt_valid.txt`

## 编码后的 token 数组目录
- `/root/autodl-tmp/tokenizer_exp_encoded`

### 当前已知 `.npy` 文件
- `/root/autodl-tmp/tokenizer_exp_encoded/tiny_train_ids.npy`
- `/root/autodl-tmp/tokenizer_exp_encoded/tiny_valid_ids.npy`
- `/root/autodl-tmp/tokenizer_exp_encoded/tinystories_valid_ids.npy`
- `/root/autodl-tmp/tokenizer_exp_encoded/owt_train_ids.npy`
- `/root/autodl-tmp/tokenizer_exp_encoded/owt_valid_ids.npy`

### 对应 metadata
- `/root/autodl-tmp/tokenizer_exp_encoded/tiny_train_ids_meta.json`
- `/root/autodl-tmp/tokenizer_exp_encoded/tiny_valid_ids_meta.json`
- `/root/autodl-tmp/tokenizer_exp_encoded/tinystories_valid_ids_meta.json`
- `/root/autodl-tmp/tokenizer_exp_encoded/owt_train_ids_meta.json`
- `/root/autodl-tmp/tokenizer_exp_encoded/owt_valid_ids_meta.json`

## 后续训练产物推荐目录（远程）
- 数据、checkpoint、模型权重、采样输出等真实产物统一放在：
  - `/root/autodl-tmp/training_runs/`

### 推荐子目录内容
- checkpoints
- model weights
- sampled outputs
- config snapshots

## 日志与协作记录
- `.agents/` 只放：
  - 协作笔记
  - 命令记录
  - 轻量日志索引
- 如果需要在仓库内留一份运行日志摘要，优先写：
  - `.agents/logs/`
- 不把真实训练产物（大 checkpoint、模型文件、数据副本）写进 `.agents/`

## Chapter 7 logging split
- Large artifacts stay on the data disk:
  - checkpoints
  - model weights
  - generated samples if they become large
- Small experiment-tracking files should stay under repo-local `.agents/logs/`:
  - `config.json`
  - `metrics.jsonl`
  - `summary.json`
  - TensorBoard event files
- Current training-script intent:
  - `--output-dir` => checkpoints / large artifacts
  - `--log-dir` => small logs and TensorBoard

## TensorBoard
- Remote TensorBoard log root: `/root/tf-logs`
- Chapter 7 experiment logging should be compatible with this path so runs can be observed live in TensorBoard.
- Even when TensorBoard is enabled, the assignment still needs structured experiment records for:
  - gradient steps
  - wallclock time
  - experiment log document / writeup evidence

## 备注
- 后续给远程训练命令时，优先复用这些绝对路径
- 远程训练命令默认采用：
  - 真实产物：`/root/autodl-tmp/training_runs/<run_name>/`
  - 仓库内日志摘要：`.agents/logs/<run_name>/`
