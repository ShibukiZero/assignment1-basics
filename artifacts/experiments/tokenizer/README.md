# Tokenizer Experiment Artifacts

This directory stores finalized tokenizer-training outputs tracked in the main repository.

## Available runs
- `tinystories_10k_train_w16/`: TinyStories train set, vocab size `10000`, special token `<|endoftext|>`.
- `owt_32k_train_w16/`: OpenWebText train set, vocab size `32000`, special token `<|endoftext|>`.

Each run directory contains:
- `report.json`: runtime/memory summary and longest-token info
- `vocab.json`: serialized vocabulary
- `merges.json`: serialized BPE merge list

Some run directories also include local logs or profiling outputs.
