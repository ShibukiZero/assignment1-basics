# Tokenizer Artifacts

This directory stores finalized tokenizer-training outputs that are tracked in the main repository for reproducibility.

## Available runs
- `tinystories_10k_train_w16/`: TinyStories train set, vocab size 10,000, `<|endoftext|>` special token.
- `owt_32k_train_w16/`: OpenWebText train set, vocab size 32,000, `<|endoftext|>` special token.

Each run directory contains:
- `report.json`: runtime/memory summary and longest-token info
- `vocab.json`: serialized vocabulary
- `merges.json`: serialized BPE merge list

Some run directories may also include local logs for debugging/progress.
