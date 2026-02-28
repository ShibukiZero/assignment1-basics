## Problem `unicode1`: Understanding Unicode (1 point)

### (a)
**Question:** What Unicode character does `chr(0)` return?  
**Deliverable:** A one-sentence response.

**Answer:** `chr(0)` returns the null character (Unicode code point U+0000, often written as `'\x00'`).

### (b)
**Question:** How does this character’s string representation (`__repr__()`) differ from its printed representation?  
**Deliverable:** A one-sentence response.

**Answer:** Its string representation is the escaped form (e.g., `'\x00'`), while printing it outputs an invisible control character.

### (c)
**Question:** What happens when this character occurs in text?  
**Deliverable:** A one-sentence response.

**Answer:** When this character appears in text, it is invisible to readers and can cause issues in systems that treat null bytes as terminators or control characters.

---

## Problem `unicode2`: Unicode Encodings (3 points)

### (a)
**Question:** What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32?  
**Deliverable:** A one-to-two sentence response.

**Answer:**

### (b)
**Question:** Consider the following (incorrect) function intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example input byte string that yields incorrect results.

```python
def decode_utf8_bytes_to_str_wrong(bytestring):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

**Deliverable:** An example input byte string for which `decode_utf8_bytes_to_str_wrong` produces incorrect output, with a one-sentence explanation.

**Answer (example bytes):**  
**Answer (explanation):**

### (c)
**Question:** Give a two-byte sequence that does not decode to any Unicode character(s).  
**Deliverable:** An example, with a one-sentence explanation.

**Answer (example bytes):**  
**Answer (explanation):**

---

## Problem `train_bpe_tinystories`: BPE Training on TinyStories (2 points)

### (a)
**Question:** Train a byte-level BPE tokenizer on TinyStories with max vocab size 10,000 and include `<|endoftext|>` as a special token. Serialize vocab and merges. Report: training hours and memory usage; longest token in vocabulary; whether it makes sense.  
**Deliverable:** A one-to-two sentence response.

**Answer:**

### (b)
**Question:** Profile your code. What part of tokenizer training takes the most time?  
**Deliverable:** A one-to-two sentence response.

**Answer:**

**Evidence paths (profiles/logs):**

---

## Problem `train_bpe_expts_owt`: BPE Training on OpenWebText (2 points)

### (a)
**Question:** Train a byte-level BPE tokenizer on OpenWebText with max vocab size 32,000. Serialize vocab and merges. What is the longest token in the vocabulary? Does it make sense?  
**Deliverable:** A one-to-two sentence response.

**Answer:**

### (b)
**Question:** Compare and contrast the tokenizer trained on TinyStories vs. the tokenizer trained on OpenWebText.  
**Deliverable:** A one-to-two sentence response.

**Answer:**

---

## Problem `tokenizer_experiments`: Experiments with tokenizers (4 points)

### (a)
**Question:** Sample 10 documents from TinyStories and OpenWebText. Using the previously trained TinyStories (10K) and OpenWebText (32K) tokenizers, encode sampled documents. What is each tokenizer’s compression ratio (bytes/token)?  
**Deliverable:** A one-to-two sentence response.

**Answer:**

### (b)
**Question:** What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare compression ratio and/or qualitatively describe behavior.  
**Deliverable:** A one-to-two sentence response.

**Answer:**

### (c)
**Question:** Estimate tokenizer throughput (bytes/second). How long would it take to tokenize The Pile dataset (825GB of text)?  
**Deliverable:** A one-to-two sentence response.

**Answer:**

### (d)
**Question:** Using TinyStories and OpenWebText tokenizers, encode train/dev datasets into integer IDs. We recommend serializing IDs as NumPy `uint16`. Why is `uint16` appropriate?  
**Deliverable:** (Written explanation required for `uint16` choice.)

**Answer:**

---

## Problem `transformer_accounting`: Transformer LM resource accounting (5 points)

### (a)
**Question:** For GPT-2 XL (`vocab_size=50,257`, `context_length=1,024`, `num_layers=48`, `d_model=1,600`, `num_heads=25`, `d_ff=6,400`), how many trainable parameters does the model have? Assuming float32 parameters, how much memory is needed to load the model?  
**Deliverable:** A one-to-two sentence response.

**Answer:**

### (b)
**Question:** Identify the matrix multiplies required for one forward pass of the GPT-2 XL-shaped model. How many FLOPs do they require in total (sequence length = `context_length`)?  
**Deliverable:** A list of matrix multiplies (with descriptions), and total FLOPs.

**Answer:**

### (c)
**Question:** Based on your analysis, which parts of the model require the most FLOPs?  
**Deliverable:** A one-to-two sentence response.

**Answer:**

### (d)
**Question:** Repeat analysis for GPT-2 small (12L, 768d, 12H), medium (24L, 1024d, 16H), and large (36L, 1280d, 20H). As model size increases, which components take proportionally more or less FLOPs?  
**Deliverable:** For each model, provide component-wise FLOP breakdown (proportion of total), plus a one-to-two sentence summary.

**Answer:**

### (e)
**Question:** For GPT-2 XL, increase context length to 16,384. How do total forward FLOPs and relative component contributions change?  
**Deliverable:** A one-to-two sentence response.

**Answer:**

---

## Problem `learning_rate_tuning`: Tuning the learning rate (1 point)

**Question:** Run the toy SGD example with learning rates `1e1`, `1e2`, and `1e3` for 10 iterations. What happens to loss for each LR (faster decay, slower decay, or divergence)?  
**Deliverable:** A one-to-two sentence response.

**Answer:**

---

## Problem `adamwAccounting`: Resource accounting for training with AdamW (2 points)

### (a)
**Question:** Compute peak memory for AdamW training in float32. Decompose into parameters, activations, gradients, optimizer state. Express in terms of `batch_size`, `vocab_size`, `context_length`, `num_layers`, `d_model`, `num_heads` (assume `d_ff = 4 * d_model`).  
**Deliverable:** Algebraic expression for each component and total.

**Answer:**

### (b)
**Question:** Instantiate for GPT-2 XL so expression depends only on `batch_size`. What maximum `batch_size` fits in 80GB?  
**Deliverable:** Expression of form `a * batch_size + b`, and max batch size.

**Answer:**

### (c)
**Question:** How many FLOPs does one AdamW step take?  
**Deliverable:** Algebraic expression with brief justification.

**Answer:**

### (d)
**Question:** A100 FP32 peak is 19.5 TFLOP/s. Assuming 50% MFU, how long (days) to train GPT-2 XL for 400K steps with batch size 1024 on one A100? Assume backward FLOPs = 2x forward FLOPs.  
**Deliverable:** Number of days with brief justification.

**Answer:**

---

## Problem `experiment_log`: Experiment logging (3 points)

**Question:** Build experiment tracking infrastructure to record losses by gradient steps and wall-clock time.  
**Deliverable:** Logging infrastructure code and an experiment log document of attempted runs.

**Writeup content (summary of experiment log):**

**Links/paths to logs and curves:**

---

## Problem `learning_rate`: Tune the learning rate (3 points)

### (a)
**Question:** Sweep learning rates for the base TinyStories model and report final losses (or divergence).  
**Deliverable 1:** Learning curves for multiple LRs and explanation of hyperparameter search strategy.  
**Deliverable 2:** A model with TinyStories validation loss (per-token) of at most 1.45 (or adjusted target if using low-resource setting explicitly allowed by the handout).

**Answer:**

### (b)
**Question:** Investigate whether the best LR is “at the edge of stability.” Include increasing-LR curves with at least one divergent run, and analyze relation to convergence.

**Deliverable:** Learning curves and analysis.

**Answer:**

---

## Problem `batch_size_experiment`: Batch size variations (1 point)

**Question:** Vary batch size from 1 to memory limit (including intermediate values such as 64 and 128). Re-tune LR if needed.  
**Deliverable 1:** Learning curves for different batch sizes.  
**Deliverable 2:** A few sentences discussing findings.

**Answer:**

---

## Problem `generate`: Generate text (1 point)

**Question:** Using your decoder and trained checkpoint, report generated text (at least 256 tokens, or until first `<|endoftext|>`). Briefly comment on fluency and at least two factors affecting quality.

**Deliverable:** Text dump + brief commentary.

**Generated text:**

**Commentary:**

---

## Problem `layer_norm_ablation`: Remove RMSNorm and train (1 point)

**Question:** Remove RMSNorm from transformer blocks and train. What happens at previous optimal LR? Can training be stabilized with lower LR?  
**Deliverable 1:** Learning curve without RMSNorm and learning curve at best LR.  
**Deliverable 2:** A few sentences on RMSNorm impact.

**Answer:**

---

## Problem `pre_norm_ablation`: Implement post-norm and train (1 point)

**Question:** Change pre-norm transformer to post-norm and train.  
**Deliverable:** Learning curve for post-norm model compared to pre-norm.

**Answer:**

---

## Problem `no_pos_emb`: Implement NoPE (1 point)

**Question:** Remove positional embedding information (NoPE) and compare with RoPE.  
**Deliverable:** Learning curve comparing RoPE and NoPE.

**Answer:**

---

## Problem `swiglu_ablation`: SwiGLU vs. SiLU (1 point)

**Question:** Compare SwiGLU FFN vs. SiLU FFN (approximately matched parameter counts).  
**Deliverable:** Learning curve comparison.

**Answer:**

---

## Problem `main_experiment`: Experiment on OpenWebText (2 points)

### (a)
**Question:** Train LM on OWT with same architecture and total training iterations as TinyStories. How well does it perform?  
**Deliverable:** Learning curve on OWT + explanation of loss differences vs. TinyStories.

**Answer:**

### (b)
**Question:** Provide generated text from OWT LM (same format as TinyStories outputs). How fluent is it? Why is output quality worse despite same model and compute budget?  
**Deliverable:** Generated text + analysis.

**Answer:**

---

## Problem `leaderboard`: Leaderboard (6 points, optional if participating)

**Question:** Train under leaderboard rules to minimize validation loss within 1.5 H100-hour.

**Deliverable:** Final validation loss, learning curve with wall-clock x-axis under 1.5 hours, and description of what was changed.

**Answer:**

---

## Appendix: Evidence Index
- Curves:
- Tables:
- Key logs:
- Notes:
