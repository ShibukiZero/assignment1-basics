from __future__ import annotations

import argparse
import json
import time
import codecs
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from cs336_basics.tokenizer import Tokenizer


PILE_SIZE_BYTES_DECIMAL = 825_000_000_000


@dataclass
class CompressionRow:
    dataset_name: str
    tokenizer_name: str
    total_bytes: int
    total_tokens: int
    bytes_per_token: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold CLI for Assignment 1 section 2.7 tokenizer experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser(
        "sample-docs",
        help="Extract sample documents from TinyStories/OpenWebText using boundary tokens.",
    )
    sample.add_argument("--tiny-path", type=Path, required=True)
    sample.add_argument("--owt-path", type=Path, required=True)
    sample.add_argument("--output-dir", type=Path, required=True)
    sample.add_argument("--num-docs", type=int, default=10)
    sample.add_argument("--boundary-token", type=str, default="<|endoftext|>")
    sample.add_argument("--chunk-size", type=int, default=1 << 20)

    compression = subparsers.add_parser(
        "compression",
        help="Compute bytes/token compression ratios for tokenizer and dataset combinations.",
    )
    compression.add_argument("--tiny-tokenizer-dir", type=Path, required=True)
    compression.add_argument("--owt-tokenizer-dir", type=Path, required=True)
    compression.add_argument("--tiny-docs-json", type=Path, required=True)
    compression.add_argument("--owt-docs-json", type=Path, required=True)
    compression.add_argument("--output-json", type=Path, required=True)

    throughput = subparsers.add_parser(
        "throughput",
        help="Estimate tokenizer throughput and extrapolate runtime for The Pile (825GB).",
    )
    throughput.add_argument("--tokenizer-dir", type=Path, required=True)
    throughput.add_argument("--input-path", type=Path, required=True)
    throughput.add_argument("--output-json", type=Path, required=True)
    throughput.add_argument("--max-bytes", type=int, default=200_000_000)
    throughput.add_argument("--chunk-size", type=int, default=1 << 20)
    throughput.add_argument("--repeats", type=int, default=1)

    encode = subparsers.add_parser(
        "encode-corpus",
        help="Encode a corpus into token IDs and save as uint16 NumPy array.",
    )
    encode.add_argument("--tokenizer-dir", type=Path, required=True)
    encode.add_argument("--input-path", type=Path, required=True)
    encode.add_argument("--output-npy", type=Path, required=True)
    encode.add_argument("--output-meta-json", type=Path, required=True)

    return parser.parse_args()


def _resolve_vocab_and_merges(tokenizer_dir: Path) -> tuple[Path, Path]:
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.json"
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(
            f"Tokenizer directory must contain vocab.json and merges.json: {tokenizer_dir}",
        )
    return vocab_path, merges_path


def _load_tokenizer(tokenizer_dir: Path, special_tokens: list[str] | None = None) -> Tokenizer:
    vocab_path, merges_path = _resolve_vocab_and_merges(tokenizer_dir)
    return Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )


def _sample_first_n_docs(
    input_path: Path,
    boundary_token: str,
    num_docs: int,
    chunk_size: int,
) -> list[str]:
    docs: list[str] = []
    boundary = boundary_token
    buffer = ""

    with input_path.open("r", encoding="utf-8") as f:
        while len(docs) < num_docs:
            chunk = f.read(chunk_size)
            if chunk == "":
                break
            buffer += chunk
            pieces = buffer.split(boundary)
            buffer = pieces.pop() if pieces else ""
            for doc in pieces:
                if doc:
                    docs.append(doc)
                    if len(docs) >= num_docs:
                        break

    if len(docs) < num_docs and buffer:
        docs.append(buffer)
    return docs[:num_docs]


def cmd_sample_docs(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tiny_docs = _sample_first_n_docs(
        input_path=args.tiny_path,
        boundary_token=args.boundary_token,
        num_docs=args.num_docs,
        chunk_size=args.chunk_size,
    )
    owt_docs = _sample_first_n_docs(
        input_path=args.owt_path,
        boundary_token=args.boundary_token,
        num_docs=args.num_docs,
        chunk_size=args.chunk_size,
    )

    tiny_json = args.output_dir / "tiny_docs.json"
    owt_json = args.output_dir / "owt_docs.json"
    meta_json = args.output_dir / "meta.json"

    tiny_json.write_text(json.dumps(tiny_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    owt_json.write_text(json.dumps(owt_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_json.write_text(
        json.dumps(
            {
                "num_docs_requested": args.num_docs,
                "num_tiny_docs": len(tiny_docs),
                "num_owt_docs": len(owt_docs),
                "boundary_token": args.boundary_token,
                "tiny_source": str(args.tiny_path),
                "owt_source": str(args.owt_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved TinyStories samples: {tiny_json}")
    print(f"Saved OWT samples: {owt_json}")
    print(f"Saved metadata: {meta_json}")


def _compression_row(
    docs: Iterable[str],
    dataset_name: str,
    tokenizer_name: str,
    tokenizer: Tokenizer,
) -> CompressionRow:
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    bytes_per_token = (total_bytes / total_tokens) if total_tokens > 0 else float("inf")
    return CompressionRow(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        total_bytes=total_bytes,
        total_tokens=total_tokens,
        bytes_per_token=bytes_per_token,
    )


def cmd_compression(args: argparse.Namespace) -> None:
    tiny_tokenizer = _load_tokenizer(args.tiny_tokenizer_dir, special_tokens=["<|endoftext|>"])
    owt_tokenizer = _load_tokenizer(args.owt_tokenizer_dir, special_tokens=["<|endoftext|>"])

    tiny_docs = json.loads(args.tiny_docs_json.read_text(encoding="utf-8"))
    owt_docs = json.loads(args.owt_docs_json.read_text(encoding="utf-8"))

    rows = [
        _compression_row(tiny_docs, "tiny_sample", "tiny_tokenizer_10k", tiny_tokenizer),
        _compression_row(owt_docs, "owt_sample", "owt_tokenizer_32k", owt_tokenizer),
        _compression_row(owt_docs, "owt_sample", "tiny_tokenizer_10k", tiny_tokenizer),
        _compression_row(tiny_docs, "tiny_sample", "owt_tokenizer_32k", owt_tokenizer),
    ]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"rows": [asdict(row) for row in rows]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved compression report: {args.output_json}")


def _iter_text_chunks(input_path: Path, chunk_size: int, max_bytes: int) -> Iterable[tuple[str, int]]:
    consumed = 0
    decoder = codecs.getincrementaldecoder("utf-8")()
    with input_path.open("rb") as f:
        while consumed < max_bytes:
            read_size = min(chunk_size, max_bytes - consumed)
            chunk_bytes = f.read(read_size)
            if chunk_bytes == b"":
                break

            pending_before = len(decoder.getstate()[0])
            chunk = decoder.decode(chunk_bytes, final=False)
            pending_after = len(decoder.getstate()[0])
            decoded_bytes = pending_before + len(chunk_bytes) - pending_after

            consumed += len(chunk_bytes)
            if chunk:
                yield chunk, decoded_bytes


def cmd_throughput(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer(args.tokenizer_dir, special_tokens=["<|endoftext|>"])
    run_rows: list[dict[str, float]] = []

    for repeat_idx in range(args.repeats):
        start = time.perf_counter()
        total_bytes = 0
        def _text_stream() -> Iterable[str]:
            nonlocal total_bytes
            for chunk, chunk_num_bytes in _iter_text_chunks(args.input_path, args.chunk_size, args.max_bytes):
                total_bytes += chunk_num_bytes
                yield chunk

        total_tokens = sum(1 for _ in tokenizer.encode_iterable(_text_stream()))
        elapsed = time.perf_counter() - start
        bytes_per_sec = total_bytes / elapsed if elapsed > 0 else 0.0
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
        run_rows.append(
            {
                "repeat": repeat_idx,
                "elapsed_seconds": elapsed,
                "total_bytes": float(total_bytes),
                "total_tokens": float(total_tokens),
                "bytes_per_sec": bytes_per_sec,
                "tokens_per_sec": tokens_per_sec,
            },
        )

    avg_bytes_per_sec = sum(row["bytes_per_sec"] for row in run_rows) / len(run_rows)
    est_seconds_for_pile = PILE_SIZE_BYTES_DECIMAL / avg_bytes_per_sec if avg_bytes_per_sec > 0 else float("inf")
    output = {
        "runs": run_rows,
        "avg_bytes_per_sec": avg_bytes_per_sec,
        "pile_size_bytes_decimal": PILE_SIZE_BYTES_DECIMAL,
        "est_seconds_for_pile": est_seconds_for_pile,
        "est_hours_for_pile": est_seconds_for_pile / 3600.0,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved throughput report: {args.output_json}")


def cmd_encode_corpus(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer(args.tokenizer_dir, special_tokens=["<|endoftext|>"])
    with args.input_path.open("r", encoding="utf-8") as f:
        token_ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.int64)

    if token_ids.size == 0:
        token_ids_uint16 = token_ids.astype(np.uint16)
        max_token_id = -1
    else:
        max_token_id = int(token_ids.max())
        if max_token_id >= 2**16:
            raise ValueError(
                f"Found token id {max_token_id}, cannot serialize to uint16 without overflow.",
            )
        token_ids_uint16 = token_ids.astype(np.uint16)

    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, token_ids_uint16)

    meta = {
        "input_path": str(args.input_path),
        "num_tokens": int(token_ids_uint16.shape[0]),
        "dtype": str(token_ids_uint16.dtype),
        "max_token_id": max_token_id,
        "output_npy": str(args.output_npy),
    }
    args.output_meta_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved token IDs: {args.output_npy}")
    print(f"Saved metadata: {args.output_meta_json}")


def main() -> None:
    args = parse_args()
    if args.command == "sample-docs":
        cmd_sample_docs(args)
    elif args.command == "compression":
        cmd_compression(args)
    elif args.command == "throughput":
        cmd_throughput(args)
    elif args.command == "encode-corpus":
        cmd_encode_corpus(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
