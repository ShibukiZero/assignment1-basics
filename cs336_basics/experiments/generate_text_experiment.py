from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

if __package__:
    from ..decoding import decode
    from ..optim import AdamW
    from ..tokenizer import Tokenizer
    from ..training import load_checkpoint
    from ..transformer import TransformerLM
else:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from cs336_basics.decoding import decode
    from cs336_basics.optim import AdamW
    from cs336_basics.tokenizer import Tokenizer
    from cs336_basics.training import load_checkpoint
    from cs336_basics.transformer import TransformerLM


@dataclass
class GenerationRecord:
    run_name: str
    checkpoint_path: str
    config_path: str
    tokenizer_dir: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    generated_token_count: int
    generated_text: str
    checkpoint_iteration: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TinyStories text from a trained checkpoint.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        required=True,
        help="Path to the training run config.json written by train_transformer_lm.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the model checkpoint to decode from.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("artifacts/tokenizers/tinystories_10k_train_w16"),
        help="Directory containing vocab.json and merges.json.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time, there was",
        help="Prompt text used to start decoding.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Softmax temperature used during sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus-sampling threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the device recorded in config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write generated text artifacts into.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="generation",
        help="Stem for generated output files.",
    )
    return parser.parse_args()


def resolve_dtype(raw: str) -> torch.dtype:
    if raw == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype for generation: {raw}.")


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.json"
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(
            f"Tokenizer directory must contain vocab.json and merges.json: {tokenizer_dir}"
        )
    return Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"],
    )


def build_model_from_config(config: dict, *, device: str | None) -> TransformerLM:
    model_cfg = config["model"]
    resolved_device = device or config["device"]
    return TransformerLM(
        vocab_size=int(model_cfg["vocab_size"]),
        context_length=int(model_cfg["context_length"]),
        d_model=int(model_cfg["d_model"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        rope_theta=float(model_cfg["rope_theta"]),
        device=torch.device(resolved_device),
        dtype=resolve_dtype(str(config["dtype"])),
    )


def build_optimizer_from_config(config: dict, model: TransformerLM) -> AdamW:
    opt_cfg = config["optimization"]
    return AdamW(
        model.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        betas=tuple(float(x) for x in opt_cfg["betas"]),
        eps=float(opt_cfg["eps"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )


def load_run_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_run_config(args.config_json)
    tokenizer = load_tokenizer(args.tokenizer_dir)

    model = build_model_from_config(config, device=args.device)
    optimizer = build_optimizer_from_config(config, model)
    checkpoint_iteration = load_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=optimizer,
        map_location=args.device or config["device"],
    )

    eot_ids = tokenizer.encode("<|endoftext|>")
    if len(eot_ids) != 1:
        raise ValueError(
            "Expected <|endoftext|> to map to exactly one token ID, "
            f"got {eot_ids}."
        )
    end_of_text_token_id = eot_ids[0]

    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
    generated_ids = decode(
        model,
        prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        end_of_text_token_id=end_of_text_token_id,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    generated_list = generated_ids.detach().cpu().tolist()
    generated_text = tokenizer.decode(generated_list)

    record = GenerationRecord(
        run_name=str(config["run_name"]),
        checkpoint_path=str(args.checkpoint),
        config_path=str(args.config_json),
        tokenizer_dir=str(args.tokenizer_dir),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generated_token_count=len(generated_list),
        generated_text=generated_text,
        checkpoint_iteration=checkpoint_iteration,
    )

    text_path = args.output_dir / f"{args.output_name}.txt"
    json_path = args.output_dir / f"{args.output_name}.json"
    text_path.write_text(generated_text + "\n", encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(record), f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"generated_text_path={text_path}")
    print(f"generation_record_path={json_path}")
    print("generated_text:")
    print(generated_text)


if __name__ == "__main__":
    main()
