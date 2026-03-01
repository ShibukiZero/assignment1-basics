from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import regex as re

# GPT-2 pre-tokenization pattern from the assignment handout.
GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Type aliases to keep BPE-related signatures readable.
Token = bytes
Pair = tuple[Token, Token]
Pretoken = tuple[Token, ...]
PretokenCounts = Counter[Pretoken]
Vocab = dict[int, Token]


def build_initial_vocab(special_tokens: list[str]) -> Vocab:
    """Create the initial byte vocabulary and append unique special tokens.

    The base byte-level BPE vocabulary always contains 256 single-byte tokens.
    Special tokens are added as fixed whole-token entries.
    """
    vocab: Vocab = {i: bytes([i]) for i in range(256)}
    vocab_values = set(vocab.values())

    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab_values:
            vocab[len(vocab)] = token_bytes
            vocab_values.add(token_bytes)

    return vocab


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split text by special tokens so merges cannot cross those boundaries.

    This returns only non-special segments that should be pre-tokenized.
    If no special tokens are provided, the full text is returned as one segment.
    """
    if not special_tokens:
        return [text]

    # Match longer specials first to avoid partial matches when specials overlap.
    ordered_special_tokens = sorted(special_tokens, key=len, reverse=True)
    delimiter = "|".join(re.escape(token) for token in ordered_special_tokens)
    return [segment for segment in re.split(delimiter, text) if segment]


def iter_pretokens(segment: str, pattern: str = GPT2_PRETOKEN_PATTERN) -> Iterator[str]:
    """Yield GPT-2 pre-tokens using regex finditer for low memory overhead."""
    for match in re.finditer(pattern, segment):
        yield match.group(0)


def pretoken_to_byte_tokens(pretoken: str) -> Pretoken:
    """Convert one pre-token string into a tuple of single-byte tokens."""
    encoded = pretoken.encode("utf-8")
    return tuple(bytes([byte]) for byte in encoded)


def build_pretoken_counts(text: str, special_tokens: list[str]) -> PretokenCounts:
    """Build frequency counts of byte-tokenized pre-tokens.

    Pipeline:
    1) Split on special tokens (as hard boundaries).
    2) GPT-2 pre-tokenize each non-special segment.
    3) Convert each pre-token into tuple[bytes, ...].
    4) Aggregate counts in a Counter.
    """
    counts: PretokenCounts = Counter()

    for segment in split_on_special_tokens(text, special_tokens):
        for pretoken in iter_pretokens(segment):
            counts[pretoken_to_byte_tokens(pretoken)] += 1

    return counts


def count_adjacent_pairs(pretoken_counts: PretokenCounts) -> Counter[Pair]:
    """Count adjacent token pairs, weighted by pre-token frequency."""
    pair_counts: Counter[Pair] = Counter()

    for sequence, freq in pretoken_counts.items():
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            pair_counts[pair] += freq

    return pair_counts


def select_best_pair(pair_counts: Counter[Pair]) -> Pair | None:
    """Select the next merge pair.

    Tie-break rule required by the assignment:
    - maximize frequency first
    - if tied, pick the lexicographically greater pair
    """
    if not pair_counts:
        return None

    # tuple comparison gives: max by count, then by pair lexicographic order.
    return max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]


def replace_pair_non_overlapping(
    sequence: Pretoken,
    pair: Pair,
    merged_token: Token | None = None,
) -> Pretoken:
    """Replace one merge pair in a pre-token sequence with non-overlapping semantics.

    Required behavior:
    - left-to-right scan
    - non-overlapping replacements
    - for (A, A, A) with pair (A, A), output must be (AA, A)
    """
    if merged_token is None:
        merged_token = pair[0] + pair[1]

    merged_sequence: list[Token] = []
    idx = 0
    while idx < len(sequence):
        if idx + 1 < len(sequence) and (sequence[idx], sequence[idx + 1]) == pair:
            merged_sequence.append(merged_token)
            idx += 2
        else:
            merged_sequence.append(sequence[idx])
            idx += 1

    return tuple(merged_sequence)


def apply_merge_to_pretoken_counts(
    pretoken_counts: PretokenCounts,
    pair: Pair,
    merged_token: Token | None = None,
) -> PretokenCounts:
    """Apply one merge step to all unique pre-token sequences.

    This is the correct-first implementation path. It scans all unique pre-token
    keys each merge round. You can optimize later by indexing affected sequences.
    """
    if merged_token is None:
        merged_token = pair[0] + pair[1]

    updated_counts: PretokenCounts = Counter()
    for pretoken, freq in pretoken_counts.items():
        replaced = replace_pair_non_overlapping(pretoken, pair, merged_token)
        updated_counts[replaced] += freq

    return updated_counts


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[Vocab, list[Pair]]:
    """Train a byte-level BPE tokenizer and return (vocab, merges).

    This function is intentionally scaffolded for incremental implementation.

    Args:
        input_path: Path to training text.
        vocab_size: Maximum final vocabulary size, including bytes and specials.
        special_tokens: User-defined special tokens to include in vocabulary.
        **kwargs: Reserved for optional tuning knobs (e.g. profiling hooks).

    Returns:
        vocab: Mapping from token id to token bytes.
        merges: Merge operations in order of creation.
    """
    # Reserved for future optional knobs (profiling hooks, debug switches, etc.).
    _ = kwargs

    if vocab_size <= 0:
        raise ValueError(f"Expected vocab_size > 0, got {vocab_size}.")

    # Step 1) Load corpus.
    text = Path(input_path).read_text(encoding="utf-8")

    # Step 2) Initialize vocabulary (base bytes + unique special tokens).
    vocab = build_initial_vocab(special_tokens)
    vocab_values = set(vocab.values())
    merges: list[Pair] = []

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} is too small; minimum is {len(vocab)} "
            f"(256 byte tokens + unique special tokens)."
        )

    # Step 3) Build pre-token counts once.
    pretoken_counts = build_pretoken_counts(text, special_tokens)

    # Step 4) Iteratively learn merges until reaching target vocab size.
    while len(vocab) < vocab_size:
        pair_counts = count_adjacent_pairs(pretoken_counts)
        best_pair = select_best_pair(pair_counts)

        # Stop early if there are no merge candidates left.
        if best_pair is None:
            break

        merged_token = best_pair[0] + best_pair[1]

        merges.append(best_pair)
        if merged_token not in vocab_values:
            vocab[len(vocab)] = merged_token
            vocab_values.add(merged_token)
        pretoken_counts = apply_merge_to_pretoken_counts(pretoken_counts, best_pair, merged_token)

    return vocab, merges
