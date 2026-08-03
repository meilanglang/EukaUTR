"""Extract sequence embeddings from the EukaUTR model.

Default behavior loads the S2 checkpoint from checkpoints/S2/EukaUTR-S2.ckpt,
extracts mean-pooled layer-12 embeddings, and saves them to
results/EukaUTR_embeddings.pt.

Examples:
    python extract_embedding.py --sequence AUCGAUCGGAUUCGUUUCCCGGGG

    python extract_embedding.py \
        --input sequences.csv \
        --seq-column sequence \
        --id-column id \
        --output results/my_embeddings.pt

    python extract_embedding.py --input sequences.fa --input-format fasta
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints/S2/EukaUTR-S2.ckpt"
DEFAULT_OUTPUT = PROJECT_DIR / "results/EukaUTR_embeddings.pt"

from src.model import ESM2
from src.rna_esm.data import Alphabet
from utils.config.utr_transformer_config import TransformerConfig
from utils.evo.tokenization import Vocab


SequenceRecord = Tuple[str, str]


def resolve_project_path(path: Optional[str]) -> Optional[Path]:
    """Resolve relative paths from the EukaUTR project directory."""
    if path is None or str(path).strip() == "":
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_DIR / p
    return p


def clean_rna_sequence(seq: str) -> str:
    """Normalize a DNA/RNA sequence to the EukaUTR RNA alphabet."""
    seq = "".join(str(seq).strip().upper().split())
    seq = seq.replace("T", "U")
    seq = re.sub(r"[RYKMSWBDHVN~]|\.|\*", "X", seq)
    invalid = sorted(set(seq) - set("ACGUX"))
    if invalid:
        raise ValueError(f"Invalid sequence characters: {invalid}. Allowed: A/C/G/T/U/X.")
    if not seq:
        raise ValueError("Sequence is empty after normalization.")
    return seq


def read_fasta(path: Path) -> List[SequenceRecord]:
    """Read sequences from a FASTA file."""
    records: List[SequenceRecord] = []
    current_id: Optional[str] = None
    chunks: List[str] = []

    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, clean_rna_sequence("".join(chunks))))
                current_id = line[1:].strip().split()[0] or f"seq_{len(records) + 1:06d}"
                chunks = []
            else:
                if current_id is None:
                    raise ValueError(f"FASTA sequence appears before header at line {line_number}: {path}")
                chunks.append(line)

    if current_id is not None:
        records.append((current_id, clean_rna_sequence("".join(chunks))))
    if not records:
        raise ValueError(f"No FASTA records found: {path}")
    return records


def read_csv_sequences(path: Path, seq_column: str, id_column: Optional[str]) -> List[SequenceRecord]:
    """Read sequences from a CSV file."""
    records: List[SequenceRecord] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if seq_column not in reader.fieldnames:
            raise ValueError(f"CSV is missing sequence column {seq_column!r}; found {reader.fieldnames}")
        if id_column and id_column not in reader.fieldnames:
            raise ValueError(f"CSV is missing id column {id_column!r}; found {reader.fieldnames}")

        for row_index, row in enumerate(reader, start=1):
            sequence_id = row.get(id_column, "") if id_column else ""
            sequence_id = str(sequence_id).strip() or f"seq_{row_index:06d}"
            records.append((sequence_id, clean_rna_sequence(row[seq_column])))

    if not records:
        raise ValueError(f"CSV contains no sequence rows: {path}")
    return records


def read_text_sequences(path: Path) -> List[SequenceRecord]:
    """Read one sequence per line from a plain text file."""
    records: List[SequenceRecord] = []
    with path.open() as handle:
        for row_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            records.append((f"seq_{row_index:06d}", clean_rna_sequence(line)))
    if not records:
        raise ValueError(f"Text file contains no sequence rows: {path}")
    return records


def read_input_sequences(
    input_path: Path,
    input_format: str,
    seq_column: str,
    id_column: Optional[str],
) -> List[SequenceRecord]:
    """Read input sequences from CSV, FASTA, or plain text."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fmt = input_format.lower().strip()
    if fmt == "auto":
        suffix = input_path.suffix.lower()
        if suffix in {".fa", ".fasta", ".fna"}:
            fmt = "fasta"
        elif suffix == ".csv":
            fmt = "csv"
        else:
            fmt = "txt"

    if fmt == "fasta":
        return read_fasta(input_path)
    if fmt == "csv":
        return read_csv_sequences(input_path, seq_column=seq_column, id_column=id_column)
    if fmt == "txt":
        return read_text_sequences(input_path)
    raise ValueError(f"Unsupported input format: {input_format!r}")


def build_model_and_vocab(checkpoint_path: Path, device: torch.device, strict: bool = True):
    """Build the EukaUTR model and load checkpoint weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    alphabet = Alphabet.from_architecture("utr_rna-esm")
    vocab = Vocab.from_esm_alphabet(alphabet)
    model = ESM2(vocab=vocab, model_config=TransformerConfig(), token_dropout=True)

    print(f"[Checkpoint] Loading: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    load_result = model.load_state_dict(state_dict, strict=strict)
    missing_keys = getattr(load_result, "missing_keys", [])
    unexpected_keys = getattr(load_result, "unexpected_keys", [])
    if missing_keys:
        print(f"[Checkpoint] Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"[Checkpoint] Unexpected keys: {len(unexpected_keys)}")

    model = model.to(device)
    model.eval()
    print(f"[Checkpoint] Loaded on device: {device}")
    return model, vocab


def encode_batch(records: Sequence[SequenceRecord], vocab: Vocab, device: torch.device) -> torch.Tensor:
    """Encode and pad a batch of normalized RNA sequences."""
    encoded = [torch.from_numpy(vocab.encode(list(seq))).long() for _, seq in records]
    max_len = max(t.numel() for t in encoded)
    tokens = torch.full(
        (len(encoded), max_len),
        vocab.pad_idx,
        dtype=torch.long,
        device=device,
    )
    for row_index, row_tokens in enumerate(encoded):
        tokens[row_index, : row_tokens.numel()] = row_tokens.to(device)
    return tokens


def pool_embedding(
    hidden: torch.Tensor,
    tokens: torch.Tensor,
    vocab: Vocab,
    mode: str,
    include_special_tokens: bool,
) -> torch.Tensor:
    """Pool token embeddings into one vector per sequence."""
    mode = mode.lower().strip()
    if mode == "cls":
        return hidden[:, 0, :]
    if mode != "mean":
        raise ValueError("pooling must be either 'mean' or 'cls'.")

    mask = tokens.ne(vocab.pad_idx)
    if not include_special_tokens:
        mask = mask & tokens.ne(vocab.bos_idx) & tokens.ne(vocab.eos_idx)
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom


@torch.no_grad()
def extract_embeddings(
    records: Sequence[SequenceRecord],
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    repr_layer: int = 12,
    pooling: str = "mean",
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    strict_load: bool = True,
    include_special_tokens: bool = False,
) -> Dict[str, object]:
    """Extract pooled embeddings for a sequence collection."""
    if not records:
        raise ValueError("No sequences were provided.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clean_records = [(str(seq_id), clean_rna_sequence(seq)) for seq_id, seq in records]
    model, vocab = build_model_and_vocab(checkpoint_path, device=device, strict=strict_load)

    all_embeddings: List[torch.Tensor] = []
    lengths: List[int] = []
    for start in range(0, len(clean_records), batch_size):
        batch_records = clean_records[start : start + batch_size]
        tokens = encode_batch(batch_records, vocab=vocab, device=device)
        outputs = model(tokens, repr_layers=[repr_layer])
        representations = outputs.get("representations", {})
        if repr_layer not in representations:
            raise KeyError(f"Layer {repr_layer} not found in model representations.")
        hidden = representations[repr_layer]
        pooled = pool_embedding(
            hidden=hidden,
            tokens=tokens,
            vocab=vocab,
            mode=pooling,
            include_special_tokens=include_special_tokens,
        )
        all_embeddings.append(pooled.detach().cpu())
        lengths.extend(len(seq) for _, seq in batch_records)
        print(f"[Embedding] processed {min(start + batch_size, len(clean_records))}/{len(clean_records)}")

    return {
        "ids": [seq_id for seq_id, _ in clean_records],
        "sequences": [seq for _, seq in clean_records],
        "lengths": lengths,
        "embeddings": torch.cat(all_embeddings, dim=0),
        "repr_layer": repr_layer,
        "pooling": pooling,
        "checkpoint": str(checkpoint_path),
    }


def get_embedding(
    seq: str,
    checkpoint_path: str = str(DEFAULT_CHECKPOINT),
    repr_layer: int = 12,
    pooling: str = "mean",
) -> torch.Tensor:
    """Convenience API for extracting one pooled sequence embedding."""
    result = extract_embeddings(
        records=[("sequence", seq)],
        checkpoint_path=Path(checkpoint_path),
        repr_layer=repr_layer,
        pooling=pooling,
        batch_size=1,
    )
    return result["embeddings"][0]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract EukaUTR sequence embeddings.")
    parser.add_argument(
        "--sequence",
        default="AUCGAUCGGAUUCGUUUCCCGGGG",
        help="Single DNA/RNA sequence. Ignored when --input is provided.",
    )
    parser.add_argument("--input", default=None, help="Optional input file: CSV, FASTA, or one-sequence-per-line text.")
    parser.add_argument("--input-format", choices=["auto", "csv", "fasta", "txt"], default="auto")
    parser.add_argument("--seq-column", default="sequence", help="Sequence column name for CSV input.")
    parser.add_argument("--id-column", default="id", help="Optional ID column name for CSV input.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output .pt file for embeddings.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="EukaUTR checkpoint path.")
    parser.add_argument("--repr-layer", type=int, default=12, help="Transformer layer to extract. Default: 12.")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean", help="Pooling method for sequence embeddings.")
    parser.add_argument("--include-special-tokens", action="store_true", help="Include BOS/EOS tokens in mean pooling.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-strict-load", action="store_true", help="Allow partial checkpoint loading.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    checkpoint = resolve_project_path(args.checkpoint)
    output = resolve_project_path(args.output)
    if checkpoint is None:
        raise ValueError("--checkpoint is required.")
    if output is None:
        raise ValueError("--output is required.")

    if args.input:
        input_path = resolve_project_path(args.input)
        assert input_path is not None
        records = read_input_sequences(
            input_path=input_path,
            input_format=args.input_format,
            seq_column=args.seq_column,
            id_column=args.id_column,
        )
    else:
        records = [("sequence", clean_rna_sequence(args.sequence))]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[Device] CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")

    print(f"[Project] {PROJECT_DIR}")
    print(f"[Input] sequences={len(records)}")
    print(f"[Output] {output}")

    result = extract_embeddings(
        records=records,
        checkpoint_path=checkpoint,
        repr_layer=args.repr_layer,
        pooling=args.pooling,
        batch_size=args.batch_size,
        device=device,
        strict_load=not args.no_strict_load,
        include_special_tokens=args.include_special_tokens,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(f"[Done] Saved embeddings with shape {tuple(result['embeddings'].shape)} to: {output}")


if __name__ == "__main__":
    main()
