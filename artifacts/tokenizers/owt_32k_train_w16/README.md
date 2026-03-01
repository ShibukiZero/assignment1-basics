# OpenWebText 32K Tokenizer Artifacts (Train, w16)

This directory stores the finalized tokenizer-training artifacts for OpenWebText.

## Run configuration
- Input: `~/autodl-tmp/owt_train.txt`
- Vocab size: `32000`
- Special token: `<|endoftext|>`
- Pretoken workers: `16`
- Merge selection: `heap_lazy_deletion`

## Outputs
- `report.json`: structured run summary (time/memory/longest token)
- `vocab.json`: serialized vocabulary
- `merges.json`: serialized BPE merges
- `terminal.log`: terminal output from the run (with temporary merge progress logs)

## Key results (from `report.json`)
- Elapsed seconds: `19508.986` (~5.42 hours)
- Peak RSS (main process, GB): `31.891`
- Final vocab size: `32000`
- Number of merges: `31743`
- Longest token: `"----------------------------------------------------------------"` (64 bytes)
