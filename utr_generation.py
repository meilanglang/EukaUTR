"""Generate 3'UTR sequences with the EukaUTR model.

The default command generates 10,000 sequences with target lengths sampled
uniformly from 100 to 1,000 nt and writes them to results/EukaUTR_sequence.csv.
Users can also pass a CSV with id and target_length columns to reproduce an
exact set of requested lengths.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints/S2/EukaUTR-S2.ckpt"
DEFAULT_OUTPUT = PROJECT_DIR / "results/EukaUTR_sequence.csv"

from src.model import ESM2
from src.rna_esm.data import Alphabet
from utils.config.utr_transformer_config import TransformerConfig
from utils.evo.tokenization import Vocab


class UTRGenerator:
    """
    Parallel-friendly 3'UTR generator.

    The model forward pass is batched, while each sequence keeps its own length.
    """

    def __init__(self, model, vocab: Vocab, device="cuda"):
        self.vocab = vocab
        self.device = device
        try:
            self.model = model.to(device)
        except RuntimeError as exc:
            if device == "cuda" and "out of memory" in str(exc).lower():
                print("[Device] CUDA OOM while moving model to GPU; falling back to CPU.")
                self.device = "cpu"
                self.model = model.to(self.device)
            else:
                raise
        self.model.eval()

        self.rna_tokens = ["A", "C", "G", "U"]
        self.rna_ids = [vocab.index(t) for t in self.rna_tokens if t in vocab.tokens]
        if not self.rna_ids:
            raise ValueError("No RNA tokens found in vocab. Expected at least one of A/C/G/U.")

        print("[Vocab] RNA token ids:")
        for token, token_id in zip(self.rna_tokens, self.rna_ids):
            print(f"  {token}: {token_id}")

    def _apply_allowed_filter(self, row_logits: torch.Tensor) -> torch.Tensor:
        row = row_logits.clone()
        mask = torch.full_like(row, float("-inf"))
        mask[self.rna_ids] = 0.0
        return row + mask

    def _apply_temp_and_top_p(self, row_logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.0) -> torch.Tensor:
        if temperature != 1.0:
            row = row_logits / max(1e-6, temperature)
        else:
            row = row_logits.clone()

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(row, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            to_remove = cum > top_p
            to_remove[..., 1:] = to_remove[..., :-1].clone()
            to_remove[..., 0] = False
            row[sorted_idx[to_remove]] = float("-inf")

        return row

    def _pick_token(self, row_after: torch.Tensor, probs: torch.Tensor, strategy: str) -> int:
        if strategy == "greedy":
            return torch.argmax(row_after).item()
        return torch.multinomial(probs, 1).item()

    @torch.no_grad()
    def generate_batch(
        self,
        lengths: List[int],
        steps: int = 20,
        temperature: float = 1.0,
        seed_ratio: float = 0.0,
        strategy: str = "multinomial",
        top_p: float = 0.0,
        fill_schedule: str = "linear",
        debug: bool = False,
    ) -> List[str]:
        if not lengths:
            return []
        if any(length <= 0 for length in lengths):
            raise ValueError("All batch lengths must be positive.")
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if not (0.0 <= seed_ratio <= 1.0):
            raise ValueError("seed_ratio must be in [0.0, 1.0].")

        fill_schedule = fill_schedule.lower().strip()
        if fill_schedule not in {"linear", "geometric"}:
            raise ValueError("fill_schedule must be either 'linear' or 'geometric'.")

        strategy = strategy.lower().strip()
        if strategy not in {"greedy", "multinomial"}:
            raise ValueError("strategy must be either 'greedy' or 'multinomial'.")

        batch_size = len(lengths)
        max_length = max(lengths)
        input_ids = torch.full(
            (batch_size, max_length + 2),
            self.vocab.pad_idx,
            dtype=torch.long,
            device=self.device,
        )

        for row_idx, length in enumerate(lengths):
            input_ids[row_idx, 0] = self.vocab.bos_idx
            input_ids[row_idx, length + 1] = self.vocab.eos_idx
            input_ids[row_idx, 1 : length + 1] = self.vocab.mask_idx

            if seed_ratio > 0:
                num_seeds = int(length * seed_ratio)
                if num_seeds > 0:
                    seed_indices = random.sample(range(length), num_seeds)
                    for idx in seed_indices:
                        input_ids[row_idx, 1 + idx] = random.choice(self.rna_ids)

        def fill_counts(remaining: int, step: int, total_steps: int) -> int:
            if fill_schedule == "geometric":
                ratio = 0.5 ** (step / max(1, total_steps - 1))
                k = max(1, int(ratio * remaining))
            else:
                remaining_steps = max(1, total_steps - step)
                k = max(1, int((remaining + remaining_steps - 1) // remaining_steps))
            return min(k, remaining)

        for s in range(steps):
            mask_counts = []
            active_rows = []
            for row_idx, length in enumerate(lengths):
                current = int((input_ids[row_idx, 1 : length + 1] == self.vocab.mask_idx).sum().item())
                mask_counts.append(current)
                if current > 0:
                    active_rows.append(row_idx)

            if not active_rows:
                break

            if debug:
                print(
                    f"\n[Generation batch] step={s + 1}/{steps}, "
                    f"active={len(active_rows)}/{batch_size}, remaining={mask_counts}"
                )

            num_to_unmask = [fill_counts(count, s, steps) for count in mask_counts]
            outputs = self.model(input_ids)
            logits = outputs["logits"]

            for row_idx in active_rows:
                length = lengths[row_idx]
                row_mask_positions = (
                    (input_ids[row_idx, 1 : length + 1] == self.vocab.mask_idx)
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                if row_mask_positions.numel() == 0:
                    continue

                row_num_to_unmask = num_to_unmask[row_idx]
                if row_num_to_unmask <= 0:
                    continue

                row_logits = logits[row_idx, 1 : length + 1]
                position_infos = []
                for local_pos in row_mask_positions.tolist():
                    row_filtered = self._apply_allowed_filter(row_logits[local_pos])
                    row_after = self._apply_temp_and_top_p(
                        row_filtered,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    probs = torch.softmax(row_after, dim=-1)
                    position_infos.append(
                        {
                            "local_pos": local_pos,
                            "row_after": row_after,
                            "probs": probs,
                            "max_conf": probs.max().item(),
                        }
                    )

                position_infos.sort(key=lambda x: x["max_conf"], reverse=True)
                for info in position_infos[:row_num_to_unmask]:
                    tok = self._pick_token(info["row_after"], info["probs"], strategy=strategy)
                    input_ids[row_idx, info["local_pos"] + 1] = tok

        sequences = []
        for row_idx, length in enumerate(lengths):
            sequences.append(self.vocab.decode(input_ids[row_idx, 1 : length + 1]))
        return sequences

    @torch.no_grad()
    def generate(
        self,
        length: int = 100,
        steps: int = 20,
        temperature: float = 1.0,
        seed_ratio: float = 0.0,
        strategy: str = "multinomial",
        top_p: float = 0.0,
        fill_schedule: str = "linear",
        debug: bool = False,
    ) -> str:
        return self.generate_batch(
            [length],
            steps=steps,
            temperature=temperature,
            seed_ratio=seed_ratio,
            strategy=strategy,
            top_p=top_p,
            fill_schedule=fill_schedule,
            debug=debug,
        )[0]


def load_checkpoint_with_check(model, ckpt_path, device="cpu", strict=True):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"[Checkpoint] NOT FOUND: {ckpt_path}")
        print("[Checkpoint] Model is using random initialized weights.")
        return False

    print(f"[Checkpoint] Found: {ckpt_path}")
    print("[Checkpoint] Loading...")

    first_param_name = None
    first_param_before = None
    for name, param in model.named_parameters():
        first_param_name = name
        first_param_before = param.detach().cpu().clone()
        break

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[Checkpoint] FAILED to read checkpoint: {e}")
        print("[Checkpoint] Model is using random initialized weights.")
        return False

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("[Checkpoint] Using key: model_state_dict")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("[Checkpoint] Using key: state_dict")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        print("[Checkpoint] Using checkpoint itself as state_dict")
    else:
        print("[Checkpoint] Invalid checkpoint format.")
        print("[Checkpoint] Model is using random initialized weights.")
        return False

    try:
        load_result = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        print("[Checkpoint] FAILED to load state_dict.")
        print(e)
        print("[Checkpoint] Model may still be randomly initialized or partially unchanged.")
        return False

    missing_keys = getattr(load_result, "missing_keys", [])
    unexpected_keys = getattr(load_result, "unexpected_keys", [])

    if missing_keys:
        print(f"[Checkpoint] Missing keys: {len(missing_keys)}")
        for k in missing_keys[:20]:
            print(f"  - {k}")
    if unexpected_keys:
        print(f"[Checkpoint] Unexpected keys: {len(unexpected_keys)}")
        for k in unexpected_keys[:20]:
            print(f"  - {k}")

    changed = None
    if first_param_name is not None:
        after_params = dict(model.named_parameters())
        first_param_after = after_params[first_param_name].detach().cpu()
        changed = not torch.equal(first_param_before, first_param_after)

    print("[Checkpoint] Load finished.")
    if strict:
        print("[Checkpoint] strict=True, all keys matched.")
    else:
        if not missing_keys and not unexpected_keys:
            print("[Checkpoint] strict=False, but all keys matched.")
        else:
            print("[Checkpoint] strict=False, partial mismatch exists.")

    if changed is True:
        print(f"[Checkpoint] Parameter changed after loading: {first_param_name}")
    elif changed is False:
        print(f"[Checkpoint] WARNING: First parameter did not change: {first_param_name}")
    else:
        print("[Checkpoint] Could not verify parameter change.")

    print("[Checkpoint] SUCCESS.")
    return True


def read_target_lengths(path, id_column, length_column):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Target-length CSV not found: {path}")

    rows = []
    seen_ids = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Target-length CSV has no header: {path}")
        missing = [column for column in (id_column, length_column) if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Target-length CSV is missing columns {missing}; found {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            sequence_id = str(row[id_column]).strip()
            if not sequence_id:
                raise ValueError(f"Empty {id_column!r} at line {line_number}")
            if sequence_id in seen_ids:
                raise ValueError(f"Duplicate id {sequence_id!r} at line {line_number}")
            try:
                target_length = int(row[length_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid target length {row[length_column]!r} at line {line_number}") from exc
            if target_length <= 0:
                raise ValueError(f"Target length must be positive at line {line_number}")
            seen_ids.add(sequence_id)
            rows.append((sequence_id, target_length))

    if not rows:
        raise ValueError(f"Target-length CSV contains no data rows: {path}")
    return rows


def parse_length_range(value: str) -> Tuple[int, int]:
    """Parse a target-length range such as '100-1000', '100:1000', or '100..1000'."""
    text = str(value).strip().lower().replace("nt", "").replace(" ", "")
    for separator in ("..", "-", ":"):
        if separator in text:
            parts = text.split(separator)
            if len(parts) != 2:
                break
            try:
                min_length = int(parts[0])
                max_length = int(parts[1])
            except ValueError as exc:
                raise ValueError(f"Invalid target-length range: {value!r}") from exc
            if min_length <= 0 or max_length <= 0:
                raise ValueError("Target lengths must be positive.")
            if min_length > max_length:
                raise ValueError(
                    f"Invalid target-length range: min={min_length} is greater than max={max_length}."
                )
            return min_length, max_length
    raise ValueError(
        f"Invalid --target-lengths value: {value!r}. Use a range like 100-1000, "
        "a comma-separated list like 100,250,500, or a CSV path."
    )


def parse_length_list(value: str) -> List[int]:
    """Parse comma-separated exact target lengths."""
    lengths = []
    for item in str(value).replace("nt", "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            length = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid target length in list: {item!r}") from exc
        if length <= 0:
            raise ValueError("Target lengths must be positive.")
        lengths.append(length)
    if not lengths:
        raise ValueError("No target lengths were provided.")
    return lengths


def build_target_lengths(target_lengths, id_column, length_column, num_sequences, seed):
    """Return (id, target_length) rows from a CSV path, exact list, or sampled range."""
    path = Path(str(target_lengths)).expanduser()
    if path.exists():
        return read_target_lengths(path, id_column, length_column)

    text = str(target_lengths).strip()
    if not text:
        raise ValueError("--target-lengths must not be empty.")
    if path.suffix.lower() == ".csv" or "/" in text or "\\" in text:
        raise FileNotFoundError(f"Target-length CSV not found: {path}")

    if "," in text:
        lengths = parse_length_list(text)
        return [(f"seq_{idx:06d}", length) for idx, length in enumerate(lengths, start=1)]

    if num_sequences <= 0:
        raise ValueError("--num-sequences must be positive when --target-lengths is a range.")
    min_length, max_length = parse_length_range(text)
    rng = random.Random(seed)
    return [
        (f"seq_{idx:06d}", rng.randint(min_length, max_length))
        for idx in range(1, num_sequences + 1)
    ]


def read_completed_lengths(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return {}
    completed = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "target_length", "sequence", "generated_length"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Existing output has incompatible columns: {reader.fieldnames}; expected {sorted(required)}"
            )
        for row in reader:
            sequence = str(row["sequence"]).strip().upper()
            try:
                target_length = int(row["target_length"])
                generated_length = int(row["generated_length"])
            except (TypeError, ValueError):
                continue
            if sequence and len(sequence) == target_length == generated_length:
                completed[str(row["id"]).strip()] = target_length
    return completed


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate 3'UTR sequences with EukaUTR. By default, target lengths "
            "are sampled uniformly from 100 to 1,000 nt."
        )
    )
    parser.add_argument(
        "--target-lengths",
        default="100-1000",
        help=(
            "Target generated-sequence length range, exact length list, or CSV path. "
            "Examples: 100-1000, 100,250,500, or targets.csv with id,target_length columns."
        ),
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10000,
        help="Number of sequences to generate when --target-lengths is a range. Default: 10000.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--length-column", default="target_length")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed-ratio", type=float, default=0.0)
    parser.add_argument("--strategy", choices=["greedy", "multinomial"], default="multinomial")
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--fill-schedule", choices=["linear", "geometric"], default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device to run generation on.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of sequences to generate in parallel.")
    parser.add_argument("--limit", type=int, default=0, help="Generate only the first N rows; 0 means all")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--debug-first", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output. By default valid existing rows are resumed/skipped.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.num_sequences <= 0:
        raise ValueError("--num-sequences must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Device] CUDA requested but not available; falling back to CPU.")
        device = "cpu"
    print(f"[Device] Using device: {device}")

    alphabet = Alphabet.from_architecture("utr_rna-esm")
    vocab = Vocab.from_esm_alphabet(alphabet)
    config = TransformerConfig()
    model = ESM2(vocab=vocab, model_config=config)

    loaded_ok = load_checkpoint_with_check(
        model=model,
        ckpt_path=args.checkpoint,
        device="cpu",
        strict=True,
    )
    if not loaded_ok:
        raise RuntimeError("Checkpoint loading failed. Stop generation to avoid using random weights.")

    generator = UTRGenerator(model, vocab, device=device)

    targets = build_target_lengths(
        target_lengths=args.target_lengths,
        id_column=args.id_column,
        length_column=args.length_column,
        num_sequences=args.num_sequences,
        seed=args.seed,
    )
    if args.limit:
        targets = targets[:args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    completed_lengths = read_completed_lengths(output_path)
    mismatched = [
        (sequence_id, completed_lengths[sequence_id], length)
        for sequence_id, length in targets
        if sequence_id in completed_lengths and completed_lengths[sequence_id] != length
    ]
    if mismatched:
        sequence_id, saved_length, requested_length = mismatched[0]
        raise ValueError(
            f"Existing output length mismatch for {sequence_id}: saved={saved_length}, requested={requested_length}. "
            "Use --overwrite to start a new output."
        )

    pending = [
        (sequence_id, length)
        for sequence_id, length in targets
        if completed_lengths.get(sequence_id) != length
    ]
    print(
        f"[Generation] targets={len(targets)}, completed={len(targets) - len(pending)}, "
        f"pending={len(pending)}, batch_size={args.batch_size}, output={output_path}"
    )

    fieldnames = ["id", "target_length", "sequence", "generated_length"]
    needs_header = not output_path.exists() or output_path.stat().st_size == 0
    total_generated = 0
    with output_path.open("a", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()

        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            batch_ids = [sequence_id for sequence_id, _ in batch]
            batch_lengths = [target_length for _, target_length in batch]
            sequences = generator.generate_batch(
                lengths=batch_lengths,
                steps=args.steps,
                temperature=args.temperature,
                seed_ratio=args.seed_ratio,
                strategy=args.strategy,
                top_p=args.top_p,
                fill_schedule=args.fill_schedule,
                debug=args.debug_first and batch_start == 0,
            )

            for sequence_id, target_length, sequence in zip(batch_ids, batch_lengths, sequences):
                sequence = "".join(base for base in sequence.upper() if base in "ACGU")
                if len(sequence) != target_length:
                    raise RuntimeError(
                        f"Generated length mismatch for {sequence_id}: expected {target_length}, got {len(sequence)}"
                    )
                writer.writerow(
                    {
                        "id": sequence_id,
                        "target_length": target_length,
                        "sequence": sequence,
                        "generated_length": len(sequence),
                    }
                )
                total_generated += 1

            handle.flush()
            print(
                f"[Generation] {total_generated}/{len(pending)} newly generated; "
                f"batch_size={len(batch)}, last_id={batch_ids[-1]}, last_length={batch_lengths[-1]}",
                flush=True,
            )

    print(f"[Generation] Done: {output_path}")


if __name__ == "__main__":
    main()
