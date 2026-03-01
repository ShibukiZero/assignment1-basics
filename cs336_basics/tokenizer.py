from __future__ import annotations

import heapq
import json
import multiprocessing as mp
import os
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import BinaryIO

import regex as re

# GPT-2 pre-tokenization pattern from the assignment handout.
GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_PRETOKEN_REGEX = re.compile(GPT2_PRETOKEN_PATTERN)
DEFAULT_PRETOKEN_BOUNDARY_TOKEN = "<|endoftext|>"
DEFAULT_PRETOKEN_CACHE_MAX_SIZE = 100_000
BYTE_TOKEN_TABLE: tuple[Token, ...] = tuple(bytes([i]) for i in range(256))

# Type aliases to keep BPE-related signatures readable.
Token = bytes
Pair = tuple[Token, Token]
Pretoken = tuple[Token, ...]
PretokenCounts = Counter[Pretoken]
Vocab = dict[int, Token]
PairHeap = list[tuple[int, Pair]]
ChunkTask = tuple[str, int, int, tuple[str, ...]]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Find chunk boundaries aligned to the given special token.

    This is adapted from the assignment helper logic, kept local to avoid
    importing `pretokenization_example.py` which contains executable demo code.
    """
    if desired_num_chunks <= 0:
        raise ValueError(f"Expected desired_num_chunks > 0, got {desired_num_chunks}.")
    if not isinstance(split_special_token, bytes):
        raise TypeError("split_special_token must be a bytestring.")
    if not split_special_token:
        raise ValueError("split_special_token must be non-empty.")

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


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


def split_on_special_tokens(text: str, special_tokens: Sequence[str]) -> list[str]:
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


def iter_pretokens(
    segment: str,
    pattern: str | re.Pattern = GPT2_PRETOKEN_REGEX,
) -> Iterator[str]:
    """Yield GPT-2 pre-tokens using a compiled regex for low overhead."""
    compiled_pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
    for match in compiled_pattern.finditer(segment):
        yield match.group(0)


def pretoken_to_byte_tokens(pretoken: str) -> Pretoken:
    """Convert one pre-token string into a tuple of single-byte tokens."""
    encoded = pretoken.encode("utf-8")
    return tuple(BYTE_TOKEN_TABLE[byte] for byte in encoded)


def count_pretoken_strings(text: str, special_tokens: Sequence[str]) -> Counter[str]:
    """Count pre-token strings before byte conversion."""
    pretoken_string_counts: Counter[str] = Counter()
    for segment in split_on_special_tokens(text, special_tokens):
        for pretoken in iter_pretokens(segment):
            pretoken_string_counts[pretoken] += 1
    return pretoken_string_counts


def convert_pretoken_string_counts_to_byte_counts(pretoken_string_counts: Counter[str]) -> PretokenCounts:
    """Convert pre-token string counts into byte-tokenized pre-token counts."""
    counts: PretokenCounts = Counter()
    for pretoken, freq in pretoken_string_counts.items():
        counts[pretoken_to_byte_tokens(pretoken)] += freq
    return counts


def _count_pretokens_in_chunk(task: ChunkTask) -> Counter[str]:
    """Worker: count pre-token strings in one byte chunk of the corpus file."""
    input_path, start, end, special_tokens = task

    with Path(input_path).open("rb") as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)

    # Boundaries are aligned to special-token starts, so strict UTF-8 decode is expected.
    chunk_text = chunk_bytes.decode("utf-8")
    return count_pretoken_strings(chunk_text, special_tokens)


def build_pretoken_counts_parallel(
    input_path: str | Path,
    special_tokens: Sequence[str],
    num_workers: int,
    boundary_token: str = DEFAULT_PRETOKEN_BOUNDARY_TOKEN,
) -> PretokenCounts:
    """Build pre-token counts with multiprocessing during pre-tokenization.

    Chunk boundaries are aligned to `boundary_token` so each process can count
    independently without introducing cross-document merges.
    """
    corpus_path = Path(input_path)
    split_token = boundary_token.encode("utf-8")
    if num_workers <= 0:
        raise ValueError(f"Expected num_workers > 0, got {num_workers}.")
    if not split_token:
        raise ValueError("boundary_token must be non-empty.")

    with corpus_path.open("rb") as file:
        boundaries = find_chunk_boundaries(
            file=file,
            desired_num_chunks=num_workers,
            split_special_token=split_token,
        )

    if len(boundaries) < 2:
        return Counter()

    tasks: list[ChunkTask] = [
        (str(corpus_path), start, end, tuple(special_tokens))
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    if not tasks:
        return Counter()

    if len(tasks) == 1:
        return convert_pretoken_string_counts_to_byte_counts(_count_pretokens_in_chunk(tasks[0]))

    worker_count = min(num_workers, len(tasks))
    with mp.Pool(processes=worker_count) as pool:
        shard_counters = pool.map(_count_pretokens_in_chunk, tasks)

    pretoken_string_counts: Counter[str] = Counter()
    for shard_counter in shard_counters:
        pretoken_string_counts.update(shard_counter)

    return convert_pretoken_string_counts_to_byte_counts(pretoken_string_counts)


def build_pretoken_counts(text: str, special_tokens: Sequence[str]) -> PretokenCounts:
    """Build frequency counts of byte-tokenized pre-tokens.

    Pipeline:
    1) Split on special tokens (as hard boundaries).
    2) GPT-2 pre-tokenize each non-special segment.
    3) Count pre-token strings.
    4) Convert each unique pre-token once into tuple[bytes, ...].
    5) Aggregate weighted counts in a Counter.
    """
    pretoken_string_counts = count_pretoken_strings(text, special_tokens)
    return convert_pretoken_string_counts_to_byte_counts(pretoken_string_counts)


def count_adjacent_pairs(pretoken_counts: PretokenCounts) -> Counter[Pair]:
    """Count adjacent token pairs, weighted by pre-token frequency."""
    pair_counts: Counter[Pair] = Counter()

    for sequence, freq in pretoken_counts.items():
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            pair_counts[pair] += freq

    return pair_counts


def count_pairs_in_sequence(sequence: Pretoken) -> Counter[Pair]:
    """Count adjacent pairs within one sequence (unweighted)."""
    local_counts: Counter[Pair] = Counter()
    for i in range(len(sequence) - 1):
        local_counts[(sequence[i], sequence[i + 1])] += 1
    return local_counts


def build_pair_to_sequences_index(pretoken_counts: PretokenCounts) -> dict[Pair, set[Pretoken]]:
    """Build inverted index: pair -> set of pretoken sequences containing that pair."""
    index: dict[Pair, set[Pretoken]] = defaultdict(set)
    for sequence in pretoken_counts.keys():
        for pair in count_pairs_in_sequence(sequence):
            index[pair].add(sequence)
    return index


def build_pair_max_heap(pair_counts: Counter[Pair]) -> PairHeap:
    """Build a max-heap of pair-count snapshots for lazy deletion.

    Heap entries are (-count, pair). `pair_counts` remains the source of truth.
    """
    heap: PairHeap = [(-count, pair) for pair, count in pair_counts.items() if count > 0]
    heapq.heapify(heap)
    return heap


def push_pair_snapshot(pair_heap: PairHeap, pair: Pair, pair_counts: Counter[Pair]) -> None:
    """Push one new pair-count snapshot into heap if still active."""
    count = pair_counts.get(pair, 0)
    if count > 0:
        heapq.heappush(pair_heap, (-count, pair))


def pop_best_pair_lazy(pair_heap: PairHeap, pair_counts: Counter[Pair]) -> Pair | None:
    """Pop best pair via heap + lazy deletion.

    Lazy deletion:
    - heap holds historical snapshots
    - pair_counts holds the latest count
    - stale snapshots are discarded at pop-time

    Tie-break is kept consistent with assignment by selecting lexicographically
    greatest pair among currently-valid candidates with equal count.
    """
    while pair_heap:
        neg_count, pair = heapq.heappop(pair_heap)
        candidate_count = -neg_count
        current_count = pair_counts.get(pair, 0)

        # Discard stale snapshots or inactive pairs.
        if current_count <= 0 or current_count != candidate_count:
            continue

        # Half-step prototype: resolve same-count ties by consuming same-count
        # valid snapshots currently visible on the heap and picking max(pair).
        same_count_pairs = [pair]
        while pair_heap and -pair_heap[0][0] == candidate_count:
            _, tied_pair = heapq.heappop(pair_heap)
            tied_current = pair_counts.get(tied_pair, 0)
            if tied_current == candidate_count and tied_current > 0:
                same_count_pairs.append(tied_pair)

        best_pair = max(same_count_pairs)
        for tied_pair in same_count_pairs:
            if tied_pair != best_pair:
                heapq.heappush(pair_heap, (-candidate_count, tied_pair))
        return best_pair

    return None


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
        **kwargs: Optional tuning knobs:
            - pretoken_workers: worker count for multiprocessing pre-tokenization.
            - pretoken_boundary_token: special token used for chunk boundaries.

    Returns:
        vocab: Mapping from token id to token bytes.
        merges: Merge operations in order of creation.
    """
    pretoken_workers = int(kwargs.pop("pretoken_workers", 1))
    pretoken_boundary_token = str(
        kwargs.pop("pretoken_boundary_token", DEFAULT_PRETOKEN_BOUNDARY_TOKEN),
    )
    # Reserved for future optional knobs (profiling hooks, debug switches, etc.).
    _ = kwargs

    if vocab_size <= 0:
        raise ValueError(f"Expected vocab_size > 0, got {vocab_size}.")
    if pretoken_workers <= 0:
        raise ValueError(f"Expected pretoken_workers > 0, got {pretoken_workers}.")

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
    # Default path stays single-process; multiprocessing is opt-in.
    if pretoken_workers > 1 and pretoken_boundary_token in special_tokens:
        pretoken_counts = build_pretoken_counts_parallel(
            input_path=input_path,
            special_tokens=special_tokens,
            num_workers=pretoken_workers,
            boundary_token=pretoken_boundary_token,
        )
    else:
        text = Path(input_path).read_text(encoding="utf-8")
        pretoken_counts = build_pretoken_counts(text, special_tokens)
    pair_counts = count_adjacent_pairs(pretoken_counts)
    pair_to_sequences = build_pair_to_sequences_index(pretoken_counts)
    pair_heap = build_pair_max_heap(pair_counts)

    # Step 4) Iteratively learn merges until reaching target vocab size.
    while len(vocab) < vocab_size:
        best_pair = pop_best_pair_lazy(pair_heap, pair_counts)

        # Stop early if there are no merge candidates left.
        if best_pair is None:
            break

        merged_token = best_pair[0] + best_pair[1]

        merges.append(best_pair)
        if merged_token not in vocab_values:
            vocab[len(vocab)] = merged_token
            vocab_values.add(merged_token)

        affected_sequences = list(pair_to_sequences.get(best_pair, ()))
        if not affected_sequences:
            # Stale candidate, remove and continue.
            pair_counts.pop(best_pair, None)
            continue

        # Snapshot original frequencies to prevent same-round double processing.
        original_freqs = {
            sequence: pretoken_counts.get(sequence, 0)
            for sequence in affected_sequences
            if pretoken_counts.get(sequence, 0) > 0
        }
        if not original_freqs:
            pair_counts.pop(best_pair, None)
            continue

        new_sequence_additions: Counter[Pretoken] = Counter()

        # Phase 1: remove old sequence contributions.
        for old_sequence, freq in original_freqs.items():
            pretoken_counts.pop(old_sequence, None)
            old_local_pairs = count_pairs_in_sequence(old_sequence)

            for pair, local_count in old_local_pairs.items():
                # Update inverted index: old sequence no longer contributes this pair.
                sequences = pair_to_sequences.get(pair)
                if sequences is not None:
                    sequences.discard(old_sequence)
                    if not sequences:
                        pair_to_sequences.pop(pair, None)

                # Update global weighted pair counts.
                pair_counts[pair] -= local_count * freq
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                else:
                    push_pair_snapshot(pair_heap, pair, pair_counts)

            new_sequence = replace_pair_non_overlapping(old_sequence, best_pair, merged_token)
            new_sequence_additions[new_sequence] += freq

        # Phase 2: add transformed sequence contributions.
        for new_sequence, freq in new_sequence_additions.items():
            pretoken_counts[new_sequence] += freq
            new_local_pairs = count_pairs_in_sequence(new_sequence)
            for pair, local_count in new_local_pairs.items():
                pair_to_sequences[pair].add(new_sequence)
                pair_counts[pair] += local_count * freq
                push_pair_snapshot(pair_heap, pair, pair_counts)

    return vocab, merges


class Tokenizer:
    """Byte-level BPE tokenizer for section 2.6 (encode/decode)."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        pretoken_cache_max_size: int = DEFAULT_PRETOKEN_CACHE_MAX_SIZE,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens or [])

        # id -> bytes is provided; build bytes -> id for fast lookup.
        self.token_to_id: dict[bytes, int] = {}
        for token_id, token_bytes in self.vocab.items():
            if token_bytes in self.token_to_id:
                raise ValueError("Vocabulary bytes must be unique across token IDs.")
            self.token_to_id[token_bytes] = token_id

        # Ensure configured special tokens exist in vocab.
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in self.token_to_id:
                new_id = max(self.vocab.keys(), default=-1) + 1
                self.vocab[new_id] = special_token_bytes
                self.token_to_id[special_token_bytes] = new_id

        # Longest-first order ensures deterministic handling of overlapping specials.
        self._ordered_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self._special_token_to_id = {
            token: self.token_to_id[token.encode("utf-8")] for token in self._ordered_special_tokens
        }

        # pair -> merge rank (smaller rank = earlier training merge = higher priority).
        self._merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        if pretoken_cache_max_size < 0:
            raise ValueError(f"Expected pretoken_cache_max_size >= 0, got {pretoken_cache_max_size}.")
        self._pretoken_cache_max_size = pretoken_cache_max_size
        # Cache hot repeated pre-tokens across long corpus runs to avoid
        # re-running the expensive merge loop for identical inputs.
        self._pretoken_id_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load a tokenizer from serialized vocab/merges JSON files.

        Supported formats:
        - vocab: list[{"id": int, "bytes_hex": str, ...}]
        - merges: list[{"token_1_hex": str, "token_2_hex": str, ...}]
        """
        vocab_raw = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        merges_raw = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))

        if not isinstance(vocab_raw, list):
            raise ValueError("Unsupported vocab format in from_files; expected a JSON list.")
        if not isinstance(merges_raw, list):
            raise ValueError("Unsupported merges format in from_files; expected a JSON list.")

        vocab: dict[int, bytes] = {}
        for row in vocab_raw:
            if not isinstance(row, dict) or "id" not in row or "bytes_hex" not in row:
                raise ValueError("Malformed vocab row; expected {'id', 'bytes_hex'} fields.")
            vocab[int(row["id"])] = bytes.fromhex(str(row["bytes_hex"]))

        merges: list[tuple[bytes, bytes]] = []
        for row in merges_raw:
            if not isinstance(row, dict) or "token_1_hex" not in row or "token_2_hex" not in row:
                raise ValueError("Malformed merges row; expected {'token_1_hex', 'token_2_hex'} fields.")
            merges.append((bytes.fromhex(str(row["token_1_hex"])), bytes.fromhex(str(row["token_2_hex"]))))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _split_text_preserving_special_tokens(self, text: str) -> list[tuple[bool, str]]:
        """Split text into ``(is_special, segment)`` pieces.

        Non-special segments preserve original order and can be pre-tokenized
        independently; special segments are emitted verbatim.
        """
        if not self._ordered_special_tokens:
            return [(False, text)] if text else []

        parts: list[tuple[bool, str]] = []
        index = 0
        while index < len(text):
            matched_special: str | None = None
            for special_token in self._ordered_special_tokens:
                if text.startswith(special_token, index):
                    matched_special = special_token
                    break

            if matched_special is not None:
                parts.append((True, matched_special))
                index += len(matched_special)
                continue

            next_special_start = len(text)
            for special_token in self._ordered_special_tokens:
                pos = text.find(special_token, index)
                if pos != -1:
                    next_special_start = min(next_special_start, pos)

            if next_special_start > index:
                parts.append((False, text[index:next_special_start]))
            index = next_special_start

        return parts

    def _find_best_merge_pair(self, sequence: Pretoken) -> Pair | None:
        """Return the highest-priority mergeable adjacent pair in the sequence."""
        best_rank = float("inf")
        best_pair: Pair | None = None
        for idx in range(len(sequence) - 1):
            pair = (sequence[idx], sequence[idx + 1])
            rank = self._merge_ranks.get(pair)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_pair = pair
        return best_pair

    def _encode_pretoken_ids_uncached(self, pretoken: str) -> tuple[int, ...]:
        """Encode one pre-token via iterative BPE merges (no cache)."""
        sequence = pretoken_to_byte_tokens(pretoken)

        while True:
            best_pair = self._find_best_merge_pair(sequence)
            if best_pair is None:
                break
            sequence = replace_pair_non_overlapping(sequence, best_pair)

        return tuple(self.token_to_id[tok] for tok in sequence)

    def _encode_pretoken_ids_cached(self, pretoken: str) -> tuple[int, ...]:
        """Encode one pre-token with an LRU-style cache for repeated strings."""
        if self._pretoken_cache_max_size == 0:
            return self._encode_pretoken_ids_uncached(pretoken)

        cached = self._pretoken_id_cache.get(pretoken)
        if cached is not None:
            self._pretoken_id_cache.move_to_end(pretoken)
            return cached

        encoded = self._encode_pretoken_ids_uncached(pretoken)
        self._pretoken_id_cache[pretoken] = encoded
        if len(self._pretoken_id_cache) > self._pretoken_cache_max_size:
            self._pretoken_id_cache.popitem(last=False)
        return encoded

    def _encode_pretoken_bytes(self, pretoken: str) -> list[int]:
        """Encode one pre-token via iterative BPE merges."""
        return list(self._encode_pretoken_ids_cached(pretoken))

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        ids: list[int] = []
        segments = self._split_text_preserving_special_tokens(text)
        for is_special, segment in segments:
            if is_special:
                ids.append(self._special_token_to_id[segment])
                continue

            for pretoken in iter_pretokens(segment):
                ids.extend(self._encode_pretoken_ids_cached(pretoken))

        return ids

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        """Lazily encode each incoming chunk and yield token IDs."""
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into UTF-8 text with replacement for malformed bytes."""
        token_bytes: list[bytes] = []
        for token_id in ids:
            if token_id not in self.vocab:
                raise KeyError(f"Unknown token id during decode: {token_id}")
            token_bytes.append(self.vocab[token_id])
        return b"".join(token_bytes).decode("utf-8", errors="replace")
