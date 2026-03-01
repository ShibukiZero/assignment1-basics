from __future__ import annotations

import argparse
import cProfile
import json
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Support both:
# 1) `python -m cs336_basics.train_bpe_experiment` (module mode)
# 2) `python cs336_basics/train_bpe_experiment.py` (direct script mode, no install)
if __package__:
    from .tokenizer import train_bpe
else:
    from tokenizer import train_bpe


@dataclass
class TrainBPEReport:
    """Serializable summary for one BPE training run."""

    input_path: str
    output_dir: str
    vocab_size_target: int
    special_tokens: list[str]
    pretoken_workers: int
    pretoken_boundary_token: str
    merge_selection: str
    elapsed_seconds: float
    max_rss_gb: float
    final_vocab_size: int
    num_merges: int
    longest_token_id: int
    longest_token_num_bytes: int
    longest_token_hex: str
    longest_token_utf8_preview: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one BPE training experiment and serialize outputs for analysis.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to training text file (e.g., TinyStoriesV2-GPT4-valid.txt or -train.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where report/vocab/merges artifacts will be written.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help="Target tokenizer vocabulary size (includes byte tokens and specials).",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="Special token to include in vocab. Pass multiple times for multiple tokens.",
    )
    parser.add_argument(
        "--profile-out",
        type=Path,
        default=None,
        help="Optional cProfile output path (e.g., ./logs/train_bpe_valid.prof).",
    )
    parser.add_argument(
        "--pretoken-workers",
        type=int,
        default=1,
        help="Number of worker processes for pre-tokenization. 1 disables multiprocessing.",
    )
    parser.add_argument(
        "--pretoken-boundary-token",
        type=str,
        default="<|endoftext|>",
        help="Special token used as chunk boundary in multiprocessing pre-tokenization.",
    )
    return parser.parse_args()


def get_peak_rss_gb() -> float:
    """Return peak resident set size in GB for current process."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    if sys.platform == "darwin":
        bytes_used = float(max_rss)
    else:
        bytes_used = float(max_rss) * 1024.0
    return bytes_used / (1024.0**3)


def token_preview(token: bytes, max_chars: int = 120) -> str:
    preview = token.decode("utf-8", errors="replace")
    preview = preview.replace("\n", "\\n").replace("\t", "\\t")
    if len(preview) > max_chars:
        return preview[:max_chars] + "..."
    return preview


def find_longest_token(vocab: dict[int, bytes]) -> tuple[int, bytes]:
    """Pick one longest token; tie-break by larger token ID for determinism."""
    return max(vocab.items(), key=lambda item: (len(item[1]), item[0]))


def save_vocab(vocab: dict[int, bytes], output_path: Path) -> None:
    """Write vocab as JSON list to keep byte data lossless and inspectable."""
    rows = [
        {
            "id": token_id,
            "bytes_hex": token.hex(),
            "utf8_preview": token_preview(token),
            "num_bytes": len(token),
        }
        for token_id, token in sorted(vocab.items())
    ]
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def save_merges(merges: list[tuple[bytes, bytes]], output_path: Path) -> None:
    rows = [
        {
            "step": i,
            "token_1_hex": token_1.hex(),
            "token_2_hex": token_2.hex(),
            "merged_hex": (token_1 + token_2).hex(),
        }
        for i, (token_1, token_2) in enumerate(merges)
    ]
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> tuple[TrainBPEReport, dict[int, bytes], list[tuple[bytes, bytes]]]:
    profiler: cProfile.Profile | None = None
    if args.profile_out is not None:
        profiler = cProfile.Profile()
        profiler.enable()

    start = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
        pretoken_workers=args.pretoken_workers,
        pretoken_boundary_token=args.pretoken_boundary_token,
    )
    elapsed_seconds = time.perf_counter() - start

    if profiler is not None:
        profiler.disable()
        args.profile_out.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(args.profile_out))

    longest_token_id, longest_token = find_longest_token(vocab)
    report = TrainBPEReport(
        input_path=str(args.input_path),
        output_dir=str(args.output_dir),
        vocab_size_target=args.vocab_size,
        special_tokens=list(args.special_token),
        pretoken_workers=args.pretoken_workers,
        pretoken_boundary_token=args.pretoken_boundary_token,
        merge_selection="heap_lazy_deletion",
        elapsed_seconds=elapsed_seconds,
        max_rss_gb=get_peak_rss_gb(),
        final_vocab_size=len(vocab),
        num_merges=len(merges),
        longest_token_id=longest_token_id,
        longest_token_num_bytes=len(longest_token),
        longest_token_hex=longest_token.hex(),
        longest_token_utf8_preview=token_preview(longest_token),
    )
    return report, vocab, merges


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report, vocab, merges = run_experiment(args)

    report_path = args.output_dir / "report.json"
    vocab_path = args.output_dir / "vocab.json"
    merges_path = args.output_dir / "merges.json"

    report_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)

    # Keep this stdout short so remote logs stay easy to parse.
    print(f"Saved report to: {report_path}")
    print(f"Saved vocab to: {vocab_path}")
    print(f"Saved merges to: {merges_path}")
    print(f"Elapsed seconds: {report.elapsed_seconds:.3f}")
    print(f"Peak RSS (GB): {report.max_rss_gb:.3f}")
    print(f"Merge selection: {report.merge_selection}")
    print(
        "Longest token:",
        f"id={report.longest_token_id},",
        f"bytes={report.longest_token_num_bytes},",
        f"preview={report.longest_token_utf8_preview!r}",
    )


if __name__ == "__main__":
    main()
