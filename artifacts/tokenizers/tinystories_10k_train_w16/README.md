# TinyStories 10K Tokenizer Artifacts (Train, w16)

This directory stores the finalized tokenizer-training artifacts for TinyStories.

## Run configuration
- Input: `~/autodl-tmp/TinyStoriesV2-GPT4-train.txt`
- Vocab size: `10000`
- Special token: `<|endoftext|>`
- Pretoken workers: `16`
- Merge selection: `heap_lazy_deletion`

## Outputs
- `report.json`: structured run summary (time/memory/longest token)
- `vocab.json`: serialized vocabulary
- `merges.json`: serialized BPE merges
- `train.prof`: cProfile output
- `terminal.log`: terminal output from the run

## Key results (from `report.json`)
- Elapsed seconds: `37.647`
- Peak RSS (main process, GB): `0.233`
- Final vocab size: `10000`
- Number of merges: `9743`
- Longest token: `" responsibility"` (15 bytes)
