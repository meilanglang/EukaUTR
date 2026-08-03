#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reference-guided 3'UTR optimization with EukaUTR activation steering.

Two steering modes are supported:
1. Provide both --positive_steering_csv and --negative_steering_csv to compute a
   task-specific steering vector from the current EukaUTR model activations.
2. Provide only --reference_csv to optimize reference sequences with the packaged
   default vector: utr_optimazaition/guided_vector/EukaUTR_Guide.pt.

Examples:
    python utr_optimization.py \
        --reference_csv refs.csv \
        --out_csv results/optimized_refs.csv

    python utr_optimization.py \
        --reference_csv refs.csv \
        --positive_steering_csv positive.csv \
        --negative_steering_csv negative.csv \
        --save_steering_vector_path utr_optimazaition/guided_vector/custom_task.pt \
        --out_csv results/optimized_refs_custom.csv
"""

import argparse
import csv
import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from utils.evo.tokenization import Vocab
from utils.config.utr_transformer_config import (
    Config,
    DataConfig,
    LoggingConfig,
    OptimizerConfig,
    ProduceConfig,
    TrainConfig,
    TransformerConfig,
)


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CKPT_PATH = "checkpoints/S2/EukaUTR-S2.ckpt"
DEFAULT_STEERING_VECTOR_PATH = "utr_optimazaition/guided_vector/EukaUTR_Guide.pt"
DEFAULT_OUTPUT_CSV = "results/eukautr_optimized_sequences.csv"


def resolve_project_path(path: Optional[str]) -> Optional[str]:
    """Resolve relative paths from this script's project directory."""
    if path is None or str(path).strip() == "":
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return str(p)


def require_existing_file(path: Optional[str], label: str) -> str:
    """Resolve and validate a required input file path."""
    resolved = resolve_project_path(path)
    if resolved is None:
        raise ValueError(f"{label} is required.")
    if not Path(resolved).exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


# =====================================================================
# Sequence utilities
# =====================================================================


def normalize_reference_sequence(seq: str) -> str:
    """
    Normalize input sequence into RNA alphabet used by the model.

    Input may be DNA or RNA:
    - whitespace is removed;
    - lowercase is converted to uppercase;
    - T is converted to U;
    - only A/C/G/U are allowed after normalization.
    """
    if seq is None:
        raise ValueError("sequence is None.")

    seq = str(seq).strip().upper()
    seq = "".join(seq.split())
    seq = seq.replace("T", "U")

    valid = {"A", "C", "G", "U"}
    invalid = sorted(set(seq) - valid)
    if invalid:
        raise ValueError(
            f"Invalid characters in sequence: {invalid}. "
            "Only A/C/G/T/U are allowed before normalization."
        )
    if len(seq) == 0:
        raise ValueError("sequence is empty after normalization.")

    return seq


def rna_to_dna(seq: str) -> str:
    """Convert RNA-style U back to DNA-style T for convenient output."""
    return seq.replace("U", "T")


def max_homopolymer_length(seq: str) -> int:
    """Return the longest run of the same base."""
    if not seq:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def get_basic_stats(seq: str) -> Dict[str, Union[int, float]]:
    """Basic sequence statistics for RNA sequences."""
    if not seq:
        return {}
    return {
        "length": len(seq),
        "GC_content": (seq.count("G") + seq.count("C")) / len(seq) * 100.0,
        "A_count": seq.count("A"),
        "C_count": seq.count("C"),
        "G_count": seq.count("G"),
        "U_count": seq.count("U"),
        "max_homopolymer": max_homopolymer_length(seq),
    }


def count_changed_positions(ref_seq: str, gen_seq: str) -> Tuple[int, float]:
    """Count final sequence differences relative to the reference sequence."""
    if len(ref_seq) != len(gen_seq):
        raise ValueError(
            f"Length mismatch: reference length={len(ref_seq)}, generated length={len(gen_seq)}"
        )
    if len(ref_seq) == 0:
        return 0, 0.0
    n_changed = sum(1 for a, b in zip(ref_seq, gen_seq) if a != b)
    return n_changed, n_changed / len(ref_seq)


def set_random_seed(seed: Optional[int]) -> None:
    """Set random seeds for reproducible masking and multinomial sampling."""
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# CSV utilities
# =====================================================================


def read_sequence_csv(
    csv_path: str,
    seq_col: str = "sequence",
    start_index: int = 0,
    num_records: Optional[int] = None,
    skip_invalid: bool = False,
) -> Tuple[List[Dict], List[str]]:
    """
    Read sequences from a CSV file.

    Returns:
    - records: list of dicts with csv_row_index, row, raw_sequence, sequence_rna.
    - fieldnames: original CSV columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    records: List[Dict] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if seq_col not in fieldnames:
            raise ValueError(f"seq_col='{seq_col}' not found in columns: {fieldnames}")

        for csv_row_index, row in enumerate(reader):
            if csv_row_index < start_index:
                continue
            raw_seq = row.get(seq_col, "")
            try:
                seq_rna = normalize_reference_sequence(raw_seq)
            except Exception as e:
                if skip_invalid:
                    print(f"[Skip invalid] {csv_path.name}, row={csv_row_index}: {e}")
                    continue
                raise ValueError(
                    f"Invalid sequence in {csv_path}, row={csv_row_index}, column={seq_col}: {e}"
                ) from e

            records.append(
                {
                    "csv_row_index": csv_row_index,
                    "row": row,
                    "raw_sequence": raw_seq,
                    "sequence_rna": seq_rna,
                }
            )

            if num_records is not None and len(records) >= num_records:
                break

    return records, fieldnames


def maybe_sample_records(records: List[Dict], n: Optional[int], seed: Optional[int]) -> List[Dict]:
    """Randomly sample up to n records. If n is None or >= len(records), return all."""
    if n is None or n >= len(records):
        return records
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    indices = sorted(indices[:n])
    return [records[i] for i in indices]


# =====================================================================
# Model/layer utility functions
# =====================================================================


def _get_attr_by_path(obj: object, path: str) -> object:
    """
    Get nested attributes by a dotted path.

    Supports module list indices, e.g.:
        encoder.layers
        model.encoder.layers
        layers.0
    """
    cur = obj
    for part in path.split("."):
        if part == "":
            continue
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _is_layer_container(obj: object) -> bool:
    """Return True if obj looks like an ordered container of modules."""
    if isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
        return all(isinstance(x, nn.Module) for x in obj)
    return False


def find_transformer_layers(
    model: nn.Module,
    layer_module_path: Optional[str] = None,
) -> Tuple[List[nn.Module], List[str]]:
    """
    Locate Transformer layer modules.

    For your current ESM2-like model this is usually model.layers or encoder.layers.
    If automatic detection fails, pass --layer_module_path explicitly.
    """
    if layer_module_path:
        container = _get_attr_by_path(model, layer_module_path)
        if not _is_layer_container(container):
            raise ValueError(
                f"--layer_module_path '{layer_module_path}' does not point to a non-empty module list."
            )
        layers = list(container)
        names = [f"{layer_module_path}.{i}" for i in range(len(layers))]
        return layers, names

    candidate_paths = [
        "layers",
        "encoder.layers",
        "encoder.sentence_encoder.layers",
        "sentence_encoder.layers",
        "model.layers",
        "model.encoder.layers",
        "transformer.layers",
        "transformer.blocks",
        "blocks",
        "module.layers",
        "module.encoder.layers",
    ]

    for path in candidate_paths:
        try:
            container = _get_attr_by_path(model, path)
        except Exception:
            continue
        if _is_layer_container(container):
            layers = list(container)
            names = [f"{path}.{i}" for i in range(len(layers))]
            print(f"[Layer detection] Found {len(layers)} layers at model.{path}")
            return layers, names

    # Conservative fallback by class name. This is less reliable, so we expose the names.
    fallback: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__.lower()
        if name and (
            "transformerlayer" in cls
            or cls in {"transformerblock", "encoderlayer"}
            or cls.endswith("transformerlayer")
        ):
            fallback.append((name, module))

    if fallback:
        print("[Layer detection] Automatic path detection failed; using class-name fallback:")
        for name, module in fallback:
            print(f"  - {name}: {module.__class__.__name__}")
        return [m for _, m in fallback], [n for n, _ in fallback]

    raise RuntimeError(
        "Could not locate Transformer layers automatically. "
        "Run once with --print_model_modules, inspect names, then set --layer_module_path."
    )


def first_tensor_from_output(output):
    """
    Extract the hidden-state tensor from a module output.

    Transformer layers commonly return either:
    - Tensor
    - tuple/list whose first element is the hidden-state tensor
    - dict containing one of several common hidden-state keys
    """
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.dim() == 3:
                return item
        for item in output:
            if torch.is_tensor(item):
                return item
    if isinstance(output, dict):
        for key in ["hidden_states", "hidden_state", "x", "output", "last_hidden_state"]:
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError(
        "Could not extract a tensor hidden state from a layer output of type "
        f"{type(output)}. You may need to adapt first_tensor_from_output()."
    )


def replace_first_tensor_in_output(output, new_tensor: torch.Tensor):
    """
    Replace the hidden-state tensor in a layer output, preserving the output structure.
    """
    if torch.is_tensor(output):
        return new_tensor

    if isinstance(output, tuple):
        out = list(output)
        for i, item in enumerate(out):
            if torch.is_tensor(item) and item.dim() == new_tensor.dim() and item.shape == new_tensor.shape:
                out[i] = new_tensor
                return tuple(out)
        # fallback: replace the first tensor
        for i, item in enumerate(out):
            if torch.is_tensor(item):
                out[i] = new_tensor
                return tuple(out)

    if isinstance(output, list):
        out = list(output)
        for i, item in enumerate(out):
            if torch.is_tensor(item) and item.dim() == new_tensor.dim() and item.shape == new_tensor.shape:
                out[i] = new_tensor
                return out
        for i, item in enumerate(out):
            if torch.is_tensor(item):
                out[i] = new_tensor
                return out

    if isinstance(output, dict):
        out = dict(output)
        for key in ["hidden_states", "hidden_state", "x", "output", "last_hidden_state"]:
            if key in out and torch.is_tensor(out[key]) and out[key].shape == new_tensor.shape:
                out[key] = new_tensor
                return out

    raise TypeError(
        "Could not replace hidden-state tensor in layer output. "
        "You may need to adapt replace_first_tensor_in_output()."
    )


def infer_hidden_layout(hidden: torch.Tensor, input_ids: torch.Tensor) -> str:
    """
    Infer whether hidden is batch-first [B, L, D] or sequence-first [L, B, D].
    """
    if hidden.dim() != 3:
        raise ValueError(f"Expected 3D hidden states, got shape={tuple(hidden.shape)}")

    batch_size, seq_len = input_ids.shape
    if hidden.shape[0] == batch_size and hidden.shape[1] == seq_len:
        return "BLD"
    if hidden.shape[1] == batch_size and hidden.shape[0] == seq_len:
        return "LBD"

    raise ValueError(
        "Cannot infer hidden-state layout. "
        f"hidden shape={tuple(hidden.shape)}, input_ids shape={tuple(input_ids.shape)}"
    )


def get_token_attr(vocab: Vocab, attr_name: str) -> Optional[int]:
    """Safely read a token index attribute from the Vocab object."""
    return getattr(vocab, attr_name, None)


def build_pool_mask(
    input_ids: torch.Tensor,
    vocab: Vocab,
    include_special_tokens: bool = False,
) -> torch.Tensor:
    """
    Build a boolean mask for token pooling.

    Default excludes artificial tokens: PAD, BOS, EOS, MASK.
    This matches biological-token pooling for UTR sequences. If you want to include
    BOS/EOS exactly as raw model tokens, set --include_special_tokens_in_steering.
    """
    mask = torch.ones_like(input_ids, dtype=torch.bool)

    pad_idx = get_token_attr(vocab, "pad_idx")
    if pad_idx is not None:
        mask &= input_ids != pad_idx

    if not include_special_tokens:
        for attr_name in ["bos_idx", "eos_idx", "mask_idx", "cls_idx", "sep_idx"]:
            idx = get_token_attr(vocab, attr_name)
            if idx is not None:
                mask &= input_ids != idx

    # Ensure every row has at least one pooled token.
    if not torch.all(mask.any(dim=1)):
        bad_rows = (~mask.any(dim=1)).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(f"No valid pooling tokens for batch rows: {bad_rows}")

    return mask


def mean_pool_hidden(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    vocab: Vocab,
    include_special_tokens: bool = False,
) -> torch.Tensor:
    """
    Mean-pool layer hidden states over valid tokens.

    Returns shape [B, D].
    Supports hidden layouts [B, L, D] and [L, B, D].
    """
    layout = infer_hidden_layout(hidden, input_ids)
    pool_mask = build_pool_mask(
        input_ids=input_ids,
        vocab=vocab,
        include_special_tokens=include_special_tokens,
    )

    if layout == "LBD":
        hidden_bld = hidden.transpose(0, 1)  # [B, L, D]
    else:
        hidden_bld = hidden

    mask_f = pool_mask.to(dtype=hidden_bld.dtype).unsqueeze(-1)  # [B, L, 1]
    summed = (hidden_bld * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp_min(1.0)
    return summed / counts


def edit_hidden_with_vector(
    hidden: torch.Tensor,
    vector: torch.Tensor,
    alpha: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Apply Activation Steering to a layer hidden-state tensor.

    h_tilde = h + alpha * v
    h_out   = h_tilde * ||h|| / ||h_tilde||

    The norm rescaling is per token, which preserves the original L2 norm of each
    hidden vector before it enters the next layer.
    """
    if hidden.dim() != 3:
        raise ValueError(f"Expected hidden dim=3, got shape={tuple(hidden.shape)}")
    if vector.dim() != 1:
        raise ValueError(f"Expected steering vector dim=1, got shape={tuple(vector.shape)}")
    if hidden.shape[-1] != vector.shape[0]:
        raise ValueError(
            f"Hidden dim and steering vector dim mismatch: {hidden.shape[-1]} vs {vector.shape[0]}"
        )

    v = vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
    original_norm = hidden.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    edited = hidden + alpha * v
    edited_norm = edited.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    edited = edited * (original_norm / edited_norm)
    return edited


# =====================================================================
# Activation capture / steering hook manager
# =====================================================================


class ActivationSteeringController:
    """
    Manage forward hooks for activation capture and activation steering.

    Modes:
    - capture: store per-layer hidden states during a forward pass.
    - steer: edit each layer output using a precomputed steering vector.
    - off: do nothing.
    """

    def __init__(
        self,
        model: nn.Module,
        layers: List[nn.Module],
        layer_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.layers = layers
        self.layer_names = layer_names or [f"layer_{i}" for i in range(len(layers))]
        if len(self.layers) != len(self.layer_names):
            raise ValueError("layers and layer_names must have the same length.")

        self.mode = "off"
        self.alpha = 0.0
        self.steering_vectors: Optional[List[torch.Tensor]] = None
        self.captured: Dict[int, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def register_hooks(self) -> None:
        """Register hooks once. Hooks are controlled by self.mode."""
        if self.handles:
            return

        for layer_idx, layer in enumerate(self.layers):
            handle = layer.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)

        print(f"[Hooks] Registered hooks on {len(self.handles)} Transformer layers.")

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        print("[Hooks] Removed hooks.")

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if self.mode == "off":
                return output

            hidden = first_tensor_from_output(output)

            if self.mode == "capture":
                self.captured[layer_idx] = hidden.detach()
                return output

            if self.mode == "steer":
                if self.steering_vectors is None:
                    raise RuntimeError("steering_vectors is None while mode='steer'.")
                vector = self.steering_vectors[layer_idx]
                edited = edit_hidden_with_vector(
                    hidden=hidden,
                    vector=vector,
                    alpha=self.alpha,
                )
                return replace_first_tensor_in_output(output, edited)

            raise RuntimeError(f"Unknown hook mode: {self.mode}")

        return hook

    @contextmanager
    def capture(self):
        """Context manager for activation capture."""
        old_mode = self.mode
        self.captured = {}
        self.mode = "capture"
        try:
            yield self.captured
        finally:
            self.mode = old_mode

    @contextmanager
    def steer(self, steering_vectors: List[torch.Tensor], alpha: float):
        """Context manager for activation steering during model forward."""
        old_mode = self.mode
        old_vectors = self.steering_vectors
        old_alpha = self.alpha

        self.steering_vectors = steering_vectors
        self.alpha = float(alpha)
        self.mode = "steer"
        try:
            yield
        finally:
            self.mode = old_mode
            self.steering_vectors = old_vectors
            self.alpha = old_alpha


# =====================================================================
# Input-id construction
# =====================================================================


def sequence_to_ids(seq_rna: str, vocab: Vocab) -> List[int]:
    """Convert an RNA sequence into token ids without special tokens."""
    base_to_id = {base: vocab.index(base) for base in ["A", "C", "G", "U"]}
    return [base_to_id[base] for base in seq_rna]


def build_input_ids_batch(
    seqs_rna: Sequence[str],
    vocab: Vocab,
    device: str,
) -> torch.Tensor:
    """
    Build padded model input ids:
        [BOS] + sequence + [EOS] + PAD...
    """
    if len(seqs_rna) == 0:
        raise ValueError("seqs_rna is empty.")

    bos_idx = getattr(vocab, "bos_idx")
    eos_idx = getattr(vocab, "eos_idx")
    pad_idx = getattr(vocab, "pad_idx", eos_idx)

    rows: List[List[int]] = []
    max_len = 0
    for seq in seqs_rna:
        seq = normalize_reference_sequence(seq)
        row = [bos_idx] + sequence_to_ids(seq, vocab) + [eos_idx]
        rows.append(row)
        max_len = max(max_len, len(row))

    padded = []
    for row in rows:
        padded.append(row + [pad_idx] * (max_len - len(row)))

    return torch.tensor(padded, dtype=torch.long, device=device)


# =====================================================================
# Steering-vector extraction
# =====================================================================


@torch.no_grad()
def compute_mean_layer_activations(
    model: nn.Module,
    vocab: Vocab,
    controller: ActivationSteeringController,
    seqs_rna: Sequence[str],
    device: str,
    batch_size: int = 8,
    include_special_tokens: bool = False,
    progress_name: str = "set",
) -> List[torch.Tensor]:
    """
    Compute mean pooled activation for each Transformer layer over a sequence set.

    For AE-PLM activation steering, each sequence representation is average-pooled
    over tokens at each layer, then averaged over all sequences in the set.
    """
    if len(seqs_rna) == 0:
        raise ValueError(f"No sequences provided for {progress_name}.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model.eval()
    n_layers = len(controller.layers)
    sums: Optional[List[torch.Tensor]] = None
    n_total = 0

    for start in range(0, len(seqs_rna), batch_size):
        batch_seqs = seqs_rna[start : start + batch_size]
        input_ids = build_input_ids_batch(batch_seqs, vocab=vocab, device=device)

        with controller.capture() as captured:
            _ = model(input_ids)

        if len(captured) != n_layers:
            missing = sorted(set(range(n_layers)) - set(captured.keys()))
            raise RuntimeError(
                f"Only captured {len(captured)}/{n_layers} layers. Missing layer indices: {missing[:20]}"
            )

        batch_means: List[torch.Tensor] = []
        for layer_idx in range(n_layers):
            hidden = captured[layer_idx]
            pooled = mean_pool_hidden(
                hidden=hidden,
                input_ids=input_ids,
                vocab=vocab,
                include_special_tokens=include_special_tokens,
            )  # [B, D]
            batch_means.append(pooled.detach())

        if sums is None:
            sums = [x.sum(dim=0).to(dtype=torch.float32) for x in batch_means]
        else:
            for i in range(n_layers):
                sums[i] += batch_means[i].sum(dim=0).to(dtype=torch.float32)

        n_total += len(batch_seqs)
        print(
            f"[Steering extraction] {progress_name}: "
            f"{min(start + batch_size, len(seqs_rna))}/{len(seqs_rna)}",
            flush=True,
        )

    assert sums is not None
    means = [x / float(n_total) for x in sums]
    return means


@torch.no_grad()
def compute_steering_vectors_from_csv(
    model: nn.Module,
    vocab: Vocab,
    controller: ActivationSteeringController,
    positive_csv: str,
    negative_csv: str,
    seq_col: str,
    device: str,
    num_samples: Optional[int] = None,
    seed: Optional[int] = 42,
    batch_size: int = 8,
    include_special_tokens: bool = False,
    skip_invalid: bool = False,
) -> Tuple[List[torch.Tensor], Dict]:
    """
    Compute per-layer steering vectors from positive and negative CSV files.

    Direction:
        steering_vector_l = mean_positive_l - mean_negative_l

    Therefore, to steer toward low beta, use low-beta sequences as positive and
    high-beta sequences as negative. To steer toward high beta, swap files.
    """
    pos_records, _ = read_sequence_csv(
        positive_csv,
        seq_col=seq_col,
        start_index=0,
        num_records=None,
        skip_invalid=skip_invalid,
    )
    neg_records, _ = read_sequence_csv(
        negative_csv,
        seq_col=seq_col,
        start_index=0,
        num_records=None,
        skip_invalid=skip_invalid,
    )

    if num_samples is not None:
        print(
            "[Steering extraction] --steering_num_samples is ignored; "
            "using all rows directly from positive/negative steering CSVs."
        )

    pos_seqs = [r["sequence_rna"] for r in pos_records]
    neg_seqs = [r["sequence_rna"] for r in neg_records]

    if len(pos_seqs) == 0 or len(neg_seqs) == 0:
        raise ValueError(
            f"Empty steering sets after loading. positive={len(pos_seqs)}, negative={len(neg_seqs)}"
        )

    print("[Steering extraction] Positive CSV:", positive_csv)
    print("[Steering extraction] Negative CSV:", negative_csv)
    print(f"[Steering extraction] Positive samples used: {len(pos_seqs)}")
    print(f"[Steering extraction] Negative samples used: {len(neg_seqs)}")
    print("[Steering extraction] Direction: positive mean - negative mean")

    pos_means = compute_mean_layer_activations(
        model=model,
        vocab=vocab,
        controller=controller,
        seqs_rna=pos_seqs,
        device=device,
        batch_size=batch_size,
        include_special_tokens=include_special_tokens,
        progress_name="positive",
    )
    neg_means = compute_mean_layer_activations(
        model=model,
        vocab=vocab,
        controller=controller,
        seqs_rna=neg_seqs,
        device=device,
        batch_size=batch_size,
        include_special_tokens=include_special_tokens,
        progress_name="negative",
    )

    vectors: List[torch.Tensor] = []
    vector_norms: List[float] = []
    for layer_idx, (p, n) in enumerate(zip(pos_means, neg_means)):
        v = (p - n).detach().to(dtype=torch.float32)
        vectors.append(v)
        vector_norms.append(float(v.norm(p=2).item()))
        print(
            f"[Steering vector] layer={layer_idx:02d}, "
            f"dim={v.numel()}, norm={vector_norms[-1]:.6f}"
        )

    meta = {
        "positive_csv": str(positive_csv),
        "negative_csv": str(negative_csv),
        "seq_col": seq_col,
        "num_positive_used": len(pos_seqs),
        "num_negative_used": len(neg_seqs),
        "include_special_tokens": include_special_tokens,
        "direction": "positive_mean_minus_negative_mean",
        "vector_norms": vector_norms,
        "layer_names": controller.layer_names,
    }
    return vectors, meta


def save_steering_vectors(path: str, vectors: List[torch.Tensor], meta: Dict) -> None:
    """Save steering vectors for reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vectors": [v.detach().cpu() for v in vectors],
        "meta": meta,
    }
    torch.save(payload, path)
    print(f"[Steering vectors] Saved to: {path}")


def load_steering_vectors(path: str, device: str) -> Tuple[List[torch.Tensor], Dict]:
    """Load steering vectors from a .pt file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Steering vector file not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    vectors = [v.to(device=device) for v in payload["vectors"]]
    meta = payload.get("meta", {})
    print(f"[Steering vectors] Loaded from: {path}")
    return vectors, meta


# =====================================================================
# UTR generator with optional activation steering
# =====================================================================


class UTRGenerator:
    """
    Reference-guided iterative MASK generator for 3'UTR sequences.

    This preserves the existing MASK filling logic:
    filtered logits -> temperature/top_p -> confidence ranking -> greedy/multinomial fill.

    New part:
    if steering_controller and steering_vectors are provided, model(input_ids) is
    executed under activation steering, so the logits used for MASK filling are
    property-steered logits.
    """

    def __init__(
        self,
        model: nn.Module,
        vocab: Vocab,
        device: str = "cuda",
        steering_controller: Optional[ActivationSteeringController] = None,
        steering_vectors: Optional[List[torch.Tensor]] = None,
        alpha: float = 1.0,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.device = device
        self.steering_controller = steering_controller
        self.steering_vectors = steering_vectors
        self.alpha = float(alpha)

        self.rna_tokens = ["A", "C", "G", "U"]
        self.rna_ids = [vocab.index(t) for t in self.rna_tokens if t in vocab.tokens]

        if len(self.rna_ids) != 4:
            raise ValueError(
                "RNA token ids are incomplete. Expected A/C/G/U in vocab, "
                f"but found ids for {len(self.rna_ids)} tokens."
            )

        self.base_to_id = {base: vocab.index(base) for base in self.rna_tokens}
        self.id_to_base = {vocab.index(base): base for base in self.rna_tokens}

        print("[Vocab] RNA token ids:")
        for token in self.rna_tokens:
            print(f"  {token}: {self.base_to_id[token]}")

    def _model_forward(self, input_ids: torch.Tensor, use_steering: bool = True):
        """Forward pass with optional activation steering."""
        if (
            use_steering
            and self.steering_controller is not None
            and self.steering_vectors is not None
        ):
            with self.steering_controller.steer(
                steering_vectors=self.steering_vectors,
                alpha=self.alpha,
            ):
                return self.model(input_ids)
        return self.model(input_ids)

    def _apply_allowed_filter(self, row_logits: torch.Tensor) -> torch.Tensor:
        """Only allow A/C/G/U logits; all other tokens are set to -inf."""
        row = row_logits.clone()
        mask = torch.full_like(row, float("-inf"))
        mask[self.rna_ids] = 0.0
        return row + mask

    def _apply_temp_and_top_p(
        self,
        row_logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.0,
    ) -> torch.Tensor:
        """Apply temperature and optional nucleus top-p truncation."""
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

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

    @torch.no_grad()
    def _unmask_step(
        self,
        input_ids: torch.Tensor,
        num_to_unmask: int,
        temperature: float = 1.0,
        strategy: str = "multinomial",
        top_p: float = 0.0,
        debug: bool = False,
        debug_max_positions: int = 3,
        base_count_penalty: float = 0.0,
        use_steering: bool = True,
    ) -> torch.Tensor:
        """
        Fill selected MASK positions.

        The only change relative to your existing code is that logits may come from
        activation-steered model forward. After logits are obtained, the token
        selection procedure is unchanged.
        """
        outputs = self._model_forward(input_ids, use_steering=use_steering)
        logits = outputs["logits"]  # [B, L, V]

        current_masks = (input_ids == self.vocab.mask_idx).nonzero(as_tuple=True)
        if current_masks[1].numel() == 0:
            return input_ids

        strategy = strategy.lower().strip()
        if strategy not in {"greedy", "multinomial"}:
            raise ValueError("strategy must be either 'greedy' or 'multinomial'.")

        mask_positions = list(zip(current_masks[0].tolist(), current_masks[1].tolist()))

        position_infos = []
        for n, (row_idx, col_idx) in enumerate(mask_positions):
            row_logits = logits[row_idx, col_idx]
            row_filtered = self._apply_allowed_filter(row_logits)

            # Keep the optional global base-count penalty from the previous code.
            if base_count_penalty > 0:
                cur_seq = input_ids[row_idx]
                filled = cur_seq[cur_seq != self.vocab.mask_idx]
                filled = filled[filled != self.vocab.bos_idx]
                filled = filled[filled != self.vocab.eos_idx]
                if filled.numel() > 0:
                    for base_id in self.rna_ids:
                        count = (filled == base_id).sum().item()
                        row_filtered[base_id] -= base_count_penalty * count

            row_after = self._apply_temp_and_top_p(
                row_filtered,
                temperature=temperature,
                top_p=top_p,
            )
            probs = torch.softmax(row_after, dim=-1)
            max_conf = probs.max().item()
            max_tok = torch.argmax(row_after).item()

            position_infos.append(
                {
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "row_after": row_after,
                    "probs": probs,
                    "max_conf": max_conf,
                    "max_tok": max_tok,
                }
            )

            if debug and n < debug_max_positions:
                print(f"\n[Debug candidate] MASK position: {col_idx}")
                for base, base_id in zip(self.rna_tokens, self.rna_ids):
                    print(f"  {base}: {float(probs[base_id]):.6f}")
                selected_base = self.id_to_base.get(max_tok, None)
                print(f"  max token id: {max_tok}")
                print(f"  max base: {selected_base}")
                print(f"  max confidence: {max_conf:.6f}")

        num_to_unmask = min(num_to_unmask, len(position_infos))
        position_infos.sort(key=lambda x: x["max_conf"], reverse=True)
        fill_infos = position_infos[:num_to_unmask]

        filled_records = []
        for info in fill_infos:
            row_idx = info["row_idx"]
            col_idx = info["col_idx"]
            row_after = info["row_after"]
            probs = info["probs"]

            if strategy == "greedy":
                tok = torch.argmax(row_after).item()
                conf = probs[tok].item()
            else:
                tok = torch.multinomial(probs, 1).item()
                conf = probs[tok].item()

            input_ids[row_idx, col_idx] = tok
            selected_base = self.id_to_base.get(tok, None)
            filled_records.append((col_idx, tok, selected_base, conf))

        if debug:
            print("\n[Debug filled positions]")
            for col_idx, tok, selected_base, conf in filled_records[:debug_max_positions]:
                print(
                    f"  position={col_idx}, token_id={tok}, "
                    f"base={selected_base}, conf={conf:.6f}"
                )

        return input_ids

    def _reference_seq_to_input_ids(self, reference_seq: str) -> torch.Tensor:
        """Convert one normalized RNA reference sequence to [BOS] seq [EOS]."""
        reference_seq = normalize_reference_sequence(reference_seq)
        seq_ids = [self.base_to_id[base] for base in reference_seq]
        input_ids = torch.tensor([seq_ids], dtype=torch.long, device=self.device)
        bos_id = torch.tensor([[self.vocab.bos_idx]], dtype=torch.long, device=self.device)
        eos_id = torch.tensor([[self.vocab.eos_idx]], dtype=torch.long, device=self.device)
        return torch.cat([bos_id, input_ids, eos_id], dim=1)

    @staticmethod
    def _decode_generated_tokens(vocab: Vocab, ids: torch.Tensor) -> str:
        """Decode generated token ids and keep only A/C/G/U characters."""
        decoded = vocab.decode(ids)
        if isinstance(decoded, (list, tuple)):
            decoded = "".join(decoded)
        decoded = str(decoded)
        return "".join(ch for ch in decoded if ch in {"A", "C", "G", "U"})

    @torch.no_grad()
    def generate_from_reference(
        self,
        reference_seq: str,
        steps: int = 20,
        mask_ratio: float = 0.10,
        temperature: float = 1.0,
        strategy: str = "multinomial",
        top_p: float = 0.0,
        debug: bool = False,
        base_count_penalty: float = 0.0,
        use_steering: bool = True,
        return_info: bool = False,
    ):
        """
        Reference-guided iterative re-generation.

        Each iteration:
        1. Randomly select round(seq_len * mask_ratio) real sequence positions without replacement.
        2. Replace them with MASK.
        3. Fill these MASK positions with the model.
           If steering vectors are set, this model forward uses Activation Steering.
        """
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if not (0.0 < mask_ratio <= 1.0):
            raise ValueError("mask_ratio must be in (0.0, 1.0].")

        strategy = strategy.lower().strip()
        if strategy not in {"greedy", "multinomial"}:
            raise ValueError("strategy must be either 'greedy' or 'multinomial'.")

        reference_seq = normalize_reference_sequence(reference_seq)
        seq_len = len(reference_seq)
        input_ids = self._reference_seq_to_input_ids(reference_seq)

        num_to_mask = max(1, int(round(seq_len * mask_ratio)))
        num_to_mask = min(num_to_mask, seq_len)

        # Real sequence positions in model input are 1..seq_len.
        sequence_positions = list(range(1, seq_len + 1))
        mask_history: List[List[int]] = []

        for s in range(steps):
            selected_positions = random.sample(sequence_positions, num_to_mask)
            for pos in selected_positions:
                input_ids[0, pos] = self.vocab.mask_idx

            if debug:
                print(
                    f"\n[Reference generation] step={s + 1}/{steps}, "
                    f"seq_len={seq_len}, mask_ratio={mask_ratio:.4f}, "
                    f"num_masked={num_to_mask}, "
                    f"use_steering={use_steering}, "
                    f"alpha={self.alpha if use_steering else 0.0}"
                )
                print(
                    "  selected 0-based sequence positions: "
                    f"{sorted([p - 1 for p in selected_positions])[:30]}"
                )

            input_ids = self._unmask_step(
                input_ids=input_ids,
                num_to_unmask=num_to_mask,
                temperature=temperature,
                strategy=strategy,
                top_p=top_p,
                debug=debug and s == 0,
                base_count_penalty=base_count_penalty,
                use_steering=bool(use_steering and abs(self.alpha) > 0.0),
            )

            remaining = int((input_ids == self.vocab.mask_idx).sum().item())
            if remaining != 0:
                raise RuntimeError(
                    f"After unmask step {s + 1}, {remaining} MASK tokens remain."
                )

            mask_history.append(sorted([p - 1 for p in selected_positions]))

        generated_seq = self._decode_generated_tokens(self.vocab, input_ids[0, 1:-1])

        if return_info:
            return {
                "reference_sequence_rna": reference_seq,
                "generated_sequence_rna": generated_seq,
                "seq_len": seq_len,
                "steps": steps,
                "mask_ratio": mask_ratio,
                "num_masked_per_step": num_to_mask,
                "mask_history": mask_history,
                "activation_steering_applied": bool(
                    use_steering and self.steering_vectors is not None and abs(self.alpha) > 0.0
                ),
                "diso_site_selection_uses_steering_vector": False,
            }

        return generated_seq


    @torch.no_grad()
    def _select_diso_positions(
        self,
        input_ids: torch.Tensor,
        T: int,
        relatedness_layer: int = -1,
        exclude_already_selected: Optional[set] = None,
        debug: bool = False,
    ) -> List[int]:
        """
        Select DISO mutation sites using token-level relatedness scores.

        For a selected layer l, DISO computes:
            s_k = cos(h_k^l, v_l)
        where h_k^l is the representation of token k at layer l, and v_l is the
        steering vector for the same layer. The T tokens with the lowest scores
        are selected as mutation sites.

        Returns:
            model-input positions in 1..seq_len, because position 0 is BOS.
        """
        if self.steering_controller is None or self.steering_vectors is None:
            raise RuntimeError("DISO requires steering_controller and steering_vectors.")
        if T <= 0:
            raise ValueError("T must be positive.")

        n_layers = len(self.steering_vectors)
        if relatedness_layer < 0:
            relatedness_layer = n_layers + relatedness_layer
        if not (0 <= relatedness_layer < n_layers):
            raise ValueError(
                f"Invalid relatedness_layer={relatedness_layer}; number of layers={n_layers}"
            )

        # Capture current token representations without activation steering.
        # The steering vector is only used as the direction for scoring sites.
        old_mode = self.steering_controller.mode
        with self.steering_controller.capture() as captured:
            _ = self.model(input_ids)
        if self.steering_controller.mode != old_mode:
            # The context manager should restore the previous mode; this is a safety check.
            self.steering_controller.mode = old_mode

        if relatedness_layer not in captured:
            raise RuntimeError(f"Layer {relatedness_layer} was not captured.")

        hidden = captured[relatedness_layer]
        layout = infer_hidden_layout(hidden, input_ids)
        hidden_bld = hidden.transpose(0, 1) if layout == "LBD" else hidden
        h = hidden_bld[0]  # [L, D], batch size 1 for generation

        seq_len = input_ids.shape[1] - 2  # exclude BOS/EOS
        candidate_positions = list(range(1, seq_len + 1))
        if exclude_already_selected is not None:
            candidate_positions = [
                p for p in candidate_positions
                if (p - 1) not in exclude_already_selected
            ]
        if not candidate_positions:
            raise RuntimeError("No candidate positions are available for DISO selection.")

        v = self.steering_vectors[relatedness_layer].to(device=h.device, dtype=h.dtype)
        h_tokens = h[candidate_positions]  # [K, D]

        h_norm = h_tokens / h_tokens.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        v_norm = v / v.norm(p=2).clamp_min(1e-12)
        scores = torch.matmul(h_norm, v_norm)  # [K]

        k = min(int(T), len(candidate_positions))
        selected_local = torch.topk(scores, k=k, largest=False).indices.tolist()
        selected_positions = [candidate_positions[i] for i in selected_local]

        if debug:
            print(
                f"[DISO site selection] relatedness_layer={relatedness_layer}, "
                f"T={T}, selected_0based={[p - 1 for p in selected_positions]}"
            )
            for p in selected_positions[:20]:
                local_idx = candidate_positions.index(p)
                print(f"  pos0={p - 1}, relatedness={float(scores[local_idx]):.6f}")

        return selected_positions

    @torch.no_grad()
    def generate_from_reference_diso(
        self,
        reference_seq: str,
        R: int = 8,
        T: int = 4,
        relatedness_layer: int = -1,
        temperature: float = 1.0,
        strategy: str = "multinomial",
        top_p: float = 0.0,
        debug: bool = False,
        base_count_penalty: float = 0.0,
        use_steering: bool = True,
        allow_reselect_positions: bool = True,
        return_info: bool = False,
    ):
        """
        DISO-style reference-guided optimization.

        Each round:
          1. Capture token representations at relatedness_layer.
          2. Compute relatedness score s_k = cos(h_k, v_layer).
          3. Select T positions with the lowest scores.
          4. Mask those positions.
          5. Fill the masked positions using the original MASK-filling logic,
             optionally under activation steering.
        """
        if R <= 0:
            raise ValueError("R must be positive.")
        if T <= 0:
            raise ValueError("T must be positive.")

        strategy = strategy.lower().strip()
        if strategy not in {"greedy", "multinomial"}:
            raise ValueError("strategy must be either 'greedy' or 'multinomial'.")

        reference_seq = normalize_reference_sequence(reference_seq)
        seq_len = len(reference_seq)
        input_ids = self._reference_seq_to_input_ids(reference_seq)

        mask_history: List[List[int]] = []
        selected_global = set()

        for r in range(R):
            selected_positions = self._select_diso_positions(
                input_ids=input_ids,
                T=T,
                relatedness_layer=relatedness_layer,
                exclude_already_selected=None if allow_reselect_positions else selected_global,
                debug=debug,
            )

            for pos in selected_positions:
                input_ids[0, pos] = self.vocab.mask_idx

            if debug:
                print(
                    f"\n[DISO generation] round={r + 1}/{R}, seq_len={seq_len}, "
                    f"T={T}, use_steering={use_steering}, "
                    f"alpha={self.alpha if use_steering else 0.0}"
                )
                print(
                    "  selected 0-based sequence positions: "
                    f"{sorted([p - 1 for p in selected_positions])}"
                )

            input_ids = self._unmask_step(
                input_ids=input_ids,
                num_to_unmask=len(selected_positions),
                temperature=temperature,
                strategy=strategy,
                top_p=top_p,
                debug=debug and r == 0,
                base_count_penalty=base_count_penalty,
                use_steering=bool(use_steering and abs(self.alpha) > 0.0),
            )

            remaining = int((input_ids == self.vocab.mask_idx).sum().item())
            if remaining != 0:
                raise RuntimeError(
                    f"After DISO round {r + 1}, {remaining} MASK tokens remain."
                )

            pos0 = sorted([p - 1 for p in selected_positions])
            mask_history.append(pos0)
            selected_global.update(pos0)

        generated_seq = self._decode_generated_tokens(self.vocab, input_ids[0, 1:-1])

        if return_info:
            return {
                "reference_sequence_rna": reference_seq,
                "generated_sequence_rna": generated_seq,
                "seq_len": seq_len,
                "steps": R,
                "mask_ratio": "",
                "num_masked_per_step": T,
                "diso_R": R,
                "diso_T": T,
                "diso_relatedness_layer": relatedness_layer,
                "mask_history": mask_history,
                "activation_steering_applied": bool(
                    use_steering and self.steering_vectors is not None and abs(self.alpha) > 0.0
                ),
                "diso_site_selection_uses_steering_vector": bool(
                    self.steering_controller is not None and self.steering_vectors is not None
                ),
            }

        return generated_seq

# =====================================================================
# Checkpoint loading
# =====================================================================


def load_checkpoint_with_check(
    model: nn.Module,
    ckpt_path: str,
    device: str = "cpu",
    strict: bool = True,
) -> bool:
    """
    Load checkpoint and verify that model weights actually changed.
    """
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
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print(f"[Checkpoint] FAILED to read checkpoint: {e}")
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
        return False

    try:
        load_result = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        print("[Checkpoint] FAILED to load state_dict.")
        print(e)
        return False

    missing_keys = getattr(load_result, "missing_keys", [])
    unexpected_keys = getattr(load_result, "unexpected_keys", [])

    if missing_keys:
        print(f"[Checkpoint] Missing keys: {len(missing_keys)}")
        for k in missing_keys[:20]:
            print(f"  - {k}")
        if len(missing_keys) > 20:
            print(f"  ... and {len(missing_keys) - 20} more")

    if unexpected_keys:
        print(f"[Checkpoint] Unexpected keys: {len(unexpected_keys)}")
        for k in unexpected_keys[:20]:
            print(f"  - {k}")
        if len(unexpected_keys) > 20:
            print(f"  ... and {len(unexpected_keys) - 20} more")

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


# =====================================================================
# CSV generation output
# =====================================================================


def generate_csv_from_references(
    generator: UTRGenerator,
    reference_csv: str,
    out_csv: str,
    seq_col: str = "sequence",
    steps: int = 20,
    mask_ratio: float = 0.10,
    temperature: float = 1.0,
    strategy: str = "multinomial",
    top_p: float = 0.0,
    base_count_penalty: float = 0.0,
    start_index: int = 0,
    num_refs: Optional[int] = None,
    num_samples_per_ref: int = 1,
    skip_invalid: bool = False,
    save_mask_history: bool = False,
    debug: bool = False,
    progress_every: int = 50,
    steering_meta: Optional[Dict] = None,
    stream_generated: bool = True,
    steering_reference_beta_col: str = "beta",
    steering_reference_beta_min: Optional[float] = None,
    reference_sample_seed: Optional[int] = 42,
    diso_mode: bool = False,
    R: int = 8,
    T: int = 4,
    relatedness_layer: int = -1,
    allow_reselect_positions: bool = True,
) -> None:
    """Batch-generate sequences from reference CSV and write an output CSV."""
    if num_samples_per_ref <= 0:
        raise ValueError("num_samples_per_ref must be positive.")

    all_records, input_fieldnames = read_sequence_csv(
        csv_path=reference_csv,
        seq_col=seq_col,
        start_index=0,
        num_records=None,
        skip_invalid=skip_invalid,
    )

    if steering_reference_beta_min is None:
        candidate_records = all_records
        print(
            f"[Reference selection] loaded={len(all_records)}, beta_filter=disabled; "
            "using reference CSV directly"
        )
    else:
        candidate_records = []
        for rec in all_records:
            raw_beta = rec["row"].get(steering_reference_beta_col, "")
            try:
                ref_beta_for_filter = float(raw_beta)
            except (TypeError, ValueError):
                ref_beta_for_filter = None
            if ref_beta_for_filter is not None and ref_beta_for_filter > steering_reference_beta_min:
                candidate_records.append(rec)
        print(
            f"[Reference selection] loaded={len(all_records)}, "
            f"kept_after_{steering_reference_beta_col}_gt_{steering_reference_beta_min}="
            f"{len(candidate_records)}"
        )

    if start_index < 0:
        raise ValueError("start_index must be non-negative.")

    candidate_records = candidate_records[start_index:]
    if num_refs is not None and num_refs < 0:
        raise ValueError("num_refs must be non-negative or omitted.")
    if num_refs is None or num_refs >= len(candidate_records):
        records = list(candidate_records)
        sample_mode = "all_after_start_index"
    else:
        rng = random.Random(reference_sample_seed)
        selected_indices = sorted(rng.sample(range(len(candidate_records)), num_refs))
        records = [candidate_records[i] for i in selected_indices]
        sample_mode = "random_without_replacement"

    print(
        f"[Reference selection] selected={len(records)}, "
        f"start_index={start_index}, num_refs={num_refs}, "
        f"sample_mode={sample_mode}, sample_seed={reference_sample_seed}"
    )

    if len(records) == 0:
        raise ValueError("No reference records were loaded from CSV after reference selection.")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    extra_fields = [
        "csv_row_index",
        "sample_index_for_reference",
        "reference_sequence_rna",
        "reference_sequence_dna",
        "generated_sequence_rna",
        "generated_sequence_dna",
        "generated_length",
        "changed_positions_count",
        "changed_positions_ratio",
        "generated_GC_content",
        "generated_A_count",
        "generated_C_count",
        "generated_G_count",
        "generated_U_count",
        "generated_max_homopolymer",
        "steps",
        "mask_ratio",
        "num_masked_per_step",
        "diso_mode",
        "diso_R",
        "diso_T",
        "diso_relatedness_layer",
        "diso_allow_reselect_positions",
        "strategy",
        "temperature",
        "top_p",
        "base_count_penalty",
        "activation_steering_enabled",
        "activation_steering_available",
        "activation_steering_applied",
        "activation_steering_alpha",
        "activation_steering_reference_beta_col",
        "activation_steering_reference_beta",
        "activation_steering_reference_beta_min",
        "activation_steering_positive_csv",
        "activation_steering_negative_csv",
        "activation_steering_num_positive_used",
        "activation_steering_num_negative_used",
        "activation_steering_direction",
        "diso_site_selection_uses_steering_vector",
    ]
    if save_mask_history:
        extra_fields.append("mask_history_json")

    output_fieldnames = list(input_fieldnames)
    for field in extra_fields:
        if field not in output_fieldnames:
            output_fieldnames.append(field)

    steering_meta = steering_meta or {}
    n_success = 0
    n_failed = 0

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        f.flush()

        for record_i, rec in enumerate(records, start=1):
            csv_row_index = rec["csv_row_index"]
            row = rec["row"]
            ref_seq_rna = rec["sequence_rna"]
            raw_ref_beta = row.get(steering_reference_beta_col, "")
            try:
                ref_beta = float(raw_ref_beta)
            except (TypeError, ValueError):
                ref_beta = None

            steering_available = generator.steering_vectors is not None
            if steering_reference_beta_min is None:
                use_steering_for_record = steering_available
            else:
                use_steering_for_record = (
                    steering_available
                    and ref_beta is not None
                    and ref_beta > steering_reference_beta_min
                )

            for sample_idx in range(num_samples_per_ref):
                try:
                    if diso_mode:
                        info = generator.generate_from_reference_diso(
                            reference_seq=ref_seq_rna,
                            R=R,
                            T=T,
                            relatedness_layer=relatedness_layer,
                            temperature=temperature,
                            strategy=strategy,
                            top_p=top_p,
                            debug=debug,
                            base_count_penalty=base_count_penalty,
                            use_steering=use_steering_for_record,
                            allow_reselect_positions=allow_reselect_positions,
                            return_info=True,
                        )
                    else:
                        info = generator.generate_from_reference(
                            reference_seq=ref_seq_rna,
                            steps=steps,
                            mask_ratio=mask_ratio,
                            temperature=temperature,
                            strategy=strategy,
                            top_p=top_p,
                            debug=debug,
                            base_count_penalty=base_count_penalty,
                            use_steering=use_steering_for_record,
                            return_info=True,
                        )

                    gen_seq_rna = info["generated_sequence_rna"]
                    n_changed, changed_ratio = count_changed_positions(ref_seq_rna, gen_seq_rna)
                    stats = get_basic_stats(gen_seq_rna)

                    out_row = dict(row)
                    out_row.update(
                        {
                            "csv_row_index": csv_row_index,
                            "sample_index_for_reference": sample_idx,
                            "reference_sequence_rna": ref_seq_rna,
                            "reference_sequence_dna": rna_to_dna(ref_seq_rna),
                            "generated_sequence_rna": gen_seq_rna,
                            "generated_sequence_dna": rna_to_dna(gen_seq_rna),
                            "generated_length": len(gen_seq_rna),
                            "changed_positions_count": n_changed,
                            "changed_positions_ratio": changed_ratio,
                            "generated_GC_content": stats.get("GC_content"),
                            "generated_A_count": stats.get("A_count"),
                            "generated_C_count": stats.get("C_count"),
                            "generated_G_count": stats.get("G_count"),
                            "generated_U_count": stats.get("U_count"),
                            "generated_max_homopolymer": stats.get("max_homopolymer"),
                            "steps": steps,
                            "mask_ratio": mask_ratio,
                            "num_masked_per_step": info["num_masked_per_step"],
                            "diso_mode": diso_mode,
                            "diso_R": R if diso_mode else "",
                            "diso_T": T if diso_mode else "",
                            "diso_relatedness_layer": relatedness_layer if diso_mode else "",
                            "diso_allow_reselect_positions": allow_reselect_positions if diso_mode else "",
                            "strategy": strategy,
                            "temperature": temperature,
                            "top_p": top_p,
                            "base_count_penalty": base_count_penalty,
                            "activation_steering_enabled": bool(use_steering_for_record and abs(generator.alpha) > 0.0),
                            "activation_steering_available": steering_available,
                            "activation_steering_applied": info.get("activation_steering_applied", False),
                            "activation_steering_alpha": generator.alpha if use_steering_for_record else 0.0,
                            "activation_steering_reference_beta_col": steering_reference_beta_col,
                            "activation_steering_reference_beta": ref_beta if ref_beta is not None else "",
                            "activation_steering_reference_beta_min": (
                                "" if steering_reference_beta_min is None else steering_reference_beta_min
                            ),
                            "activation_steering_positive_csv": steering_meta.get("positive_csv", ""),
                            "activation_steering_negative_csv": steering_meta.get("negative_csv", ""),
                            "activation_steering_num_positive_used": steering_meta.get("num_positive_used", ""),
                            "activation_steering_num_negative_used": steering_meta.get("num_negative_used", ""),
                            "activation_steering_direction": steering_meta.get("direction", ""),
                            "diso_site_selection_uses_steering_vector": info.get("diso_site_selection_uses_steering_vector", False),
                        }
                    )
                    if save_mask_history:
                        out_row["mask_history_json"] = json.dumps(info["mask_history"])

                    writer.writerow(out_row)
                    f.flush()
                    n_success += 1

                    if stream_generated:
                        print(
                            f"[Generated] csv_row_index={csv_row_index} "
                            f"sample_idx={sample_idx} gene={row.get('gene', '')} "
                            f"beta={ref_beta if ref_beta is not None else 'NA'} "
                            f"steered={use_steering_for_record} "
                            f"changed={n_changed}/{len(ref_seq_rna)} "
                            f"seq={gen_seq_rna}",
                            flush=True,
                        )

                except Exception as e:
                    n_failed += 1
                    msg = (
                        f"[Failed] csv_row_index={csv_row_index}, "
                        f"sample_idx={sample_idx}: {e}"
                    )
                    if skip_invalid:
                        print(msg)
                        continue
                    raise RuntimeError(msg) from e

            if progress_every > 0 and record_i % progress_every == 0:
                print(
                    f"[Progress] processed_reference_records={record_i}/{len(records)}, "
                    f"success_outputs={n_success}, failed={n_failed}"
                )

    print(f"[Done] output CSV: {out_csv}")
    print(f"[Done] success_outputs={n_success}, failed={n_failed}")


# =====================================================================
# Argument parsing / main
# =====================================================================


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize reference 3'UTR sequences with EukaUTR activation steering. "
            "By default, the script loads utr_optimazaition/guided_vector/EukaUTR_Guide.pt. "
            "If both positive and negative steering CSVs are provided, it computes "
            "a new steering vector from those CSVs instead."
        )
    )

    # Reference generation input/output
    parser.add_argument(
        "--reference_csv",
        type=str,
        default=None,
        help="CSV containing reference 3'UTR sequences to optimize. Required for generation.",
    )
    parser.add_argument("--seq_col", type=str, default="sequence")
    parser.add_argument("--out_csv", type=str, default=DEFAULT_OUTPUT_CSV)

    # Steering datasets
    parser.add_argument(
        "--positive_steering_csv",
        type=str,
        default=None,
        help="Positive sequence CSV for computing a task-specific steering vector.",
    )
    parser.add_argument(
        "--negative_steering_csv",
        type=str,
        default=None,
        help="Negative sequence CSV for computing a task-specific steering vector.",
    )
    parser.add_argument(
        "--steering_num_samples",
        type=int,
        default=-1,
        help="Deprecated: steering CSVs are now used directly without sampling/filtering.",
    )
    parser.add_argument("--steering_batch_size", type=int, default=8)
    parser.add_argument(
        "--include_special_tokens_in_steering",
        action="store_true",
        help="Include BOS/EOS/PAD/MASK tokens in the average activation. Default pools real bases only.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Activation steering strength.")
    parser.add_argument(
        "--steering_vector_path",
        type=str,
        default=DEFAULT_STEERING_VECTOR_PATH,
        help=(
            "Precomputed steering-vector .pt path used when positive/negative steering CSVs "
            "are not provided. Default: utr_optimazaition/guided_vector/EukaUTR_Guide.pt."
        ),
    )
    parser.add_argument(
        "--save_steering_vector_path",
        type=str,
        default=None,
        help="Optional .pt path to save computed steering vectors.",
    )
    parser.add_argument(
        "--steer_reference_beta_col",
        type=str,
        default="beta",
        help="Reference CSV column used to decide whether steering is applied.",
    )
    parser.add_argument(
        "--steer_reference_beta_min",
        type=float,
        default=None,
        help="Optional extra reference beta filter. Default: disabled because the default reference CSV is already beta > 0.4.",
    )
    parser.add_argument(
        "--steer_all_references",
        action="store_true",
        help="Disable reference beta filtering and apply steering to every reference.",
    )

    # Generation hyperparameters
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--mask_ratio", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--strategy", type=str, choices=["greedy", "multinomial"], default="multinomial")
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--base_count_penalty", type=float, default=0.0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--num_refs",
        type=int,
        default=None,
        help="Number of reference sequences to process. Default: all after start_index.",
    )
    parser.add_argument("--num_samples_per_ref", type=int, default=1)

    # DISO hyperparameters
    parser.add_argument(
        "--diso_mode",
        dest="diso_mode",
        action="store_true",
        default=True,
        help="Use DISO relatedness-based site selection. Enabled by default.",
    )
    parser.add_argument(
        "--random_mask_mode",
        dest="diso_mode",
        action="store_false",
        help="Use random mask selection instead of DISO site selection.",
    )
    parser.add_argument("--R", type=int, default=8, help="Number of DISO optimization rounds.")
    parser.add_argument("--T", type=int, default=4, help="Number of DISO mutation sites per round.")
    parser.add_argument(
        "--relatedness_layer",
        type=int,
        default=-1,
        help="Transformer layer used for DISO relatedness scores. Default -1 means last layer.",
    )
    parser.add_argument(
        "--no_allow_reselect_positions",
        dest="allow_reselect_positions",
        action="store_false",
        help="Do not allow DISO to select positions that were selected in previous rounds.",
    )
    parser.set_defaults(allow_reselect_positions=True)

    # Model/checkpoint
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_strict_load", action="store_true")
    parser.add_argument(
        "--layer_module_path",
        type=str,
        default=None,
        help="Explicit path to Transformer layer container, e.g. 'layers' or 'encoder.layers'.",
    )
    parser.add_argument(
        "--print_model_modules",
        action="store_true",
        help="Print model module names and exit. Use this if layer detection fails.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_invalid", action="store_true")
    parser.add_argument("--save_mask_history", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument(
        "--stream_generated",
        dest="stream_generated",
        action="store_true",
        default=True,
        help="Print each generated sequence immediately and flush the output CSV after each row.",
    )
    parser.add_argument(
        "--no_stream_generated",
        dest="stream_generated",
        action="store_false",
        help="Disable per-sequence real-time generated sequence printing.",
    )

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    has_positive_csv = bool(args.positive_steering_csv)
    has_negative_csv = bool(args.negative_steering_csv)
    if has_positive_csv != has_negative_csv:
        raise ValueError(
            "Provide both --positive_steering_csv and --negative_steering_csv, "
            "or provide neither to use the default precomputed steering vector."
        )

    compute_steering_from_csv = has_positive_csv and has_negative_csv

    args.ckpt_path = require_existing_file(args.ckpt_path, "EukaUTR checkpoint")
    args.out_csv = resolve_project_path(args.out_csv)
    args.save_steering_vector_path = resolve_project_path(args.save_steering_vector_path)

    if not args.print_model_modules:
        args.reference_csv = require_existing_file(args.reference_csv, "reference CSV")
        if compute_steering_from_csv:
            args.positive_steering_csv = require_existing_file(
                args.positive_steering_csv,
                "positive steering CSV",
            )
            args.negative_steering_csv = require_existing_file(
                args.negative_steering_csv,
                "negative steering CSV",
            )
            args.steering_vector_path = None
        else:
            args.steering_vector_path = require_existing_file(
                args.steering_vector_path or DEFAULT_STEERING_VECTOR_PATH,
                "steering vector",
            )

    set_random_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Project] {SCRIPT_DIR}")
    print(f"[Device] Using device: {device}")
    print(f"[Seed] {args.seed}")

    # Keep the same project imports as your existing generation script.
    from src.model import ESM2
    from src.rna_esm.data import Alphabet
    from utils.config.utr_transformer_config import TransformerConfig

    alphabet = Alphabet.from_architecture("utr_rna-esm")
    vocab = Vocab.from_esm_alphabet(alphabet)
    config = TransformerConfig()
    model = ESM2(vocab=vocab, model_config=config)

    if args.print_model_modules:
        print("[Model modules]")
        for name, module in model.named_modules():
            print(f"{name}\t{module.__class__.__name__}")
        return

    loaded_ok = load_checkpoint_with_check(
        model=model,
        ckpt_path=args.ckpt_path,
        device=device,
        strict=not args.no_strict_load,
    )
    if not loaded_ok:
        raise RuntimeError(
            "Checkpoint loading failed. Stop generation to avoid using random weights."
        )

    model = model.to(device)
    model.eval()

    layers, layer_names = find_transformer_layers(
        model=model,
        layer_module_path=args.layer_module_path,
    )
    print(f"[Layers] Number of steered layers: {len(layers)}")
    for i, name in enumerate(layer_names):
        print(f"  layer {i:02d}: {name}")

    controller = ActivationSteeringController(
        model=model,
        layers=layers,
        layer_names=layer_names,
    )
    controller.register_hooks()

    # Compute or load steering vectors. Positive/negative steering CSVs are used directly.
    steering_num_samples: Optional[int] = None
    if args.steering_num_samples is not None and args.steering_num_samples >= 0:
        print(
            "[Steering extraction] Ignoring --steering_num_samples; "
            "positive/negative steering CSV rows are used directly."
        )

    if compute_steering_from_csv:
        print("[Steering mode] computing vector from positive/negative CSVs")
        steering_vectors, steering_meta = compute_steering_vectors_from_csv(
            model=model,
            vocab=vocab,
            controller=controller,
            positive_csv=args.positive_steering_csv,
            negative_csv=args.negative_steering_csv,
            seq_col=args.seq_col,
            device=device,
            num_samples=steering_num_samples,
            seed=args.seed,
            batch_size=args.steering_batch_size,
            include_special_tokens=args.include_special_tokens_in_steering,
            skip_invalid=args.skip_invalid,
        )
        steering_meta["source"] = "computed_from_positive_negative_csv"
        if args.save_steering_vector_path:
            save_steering_vectors(
                args.save_steering_vector_path,
                vectors=steering_vectors,
                meta=steering_meta,
            )
    else:
        print(f"[Steering mode] loading precomputed vector: {args.steering_vector_path}")
        steering_vectors, steering_meta = load_steering_vectors(
            args.steering_vector_path,
            device=device,
        )
        if len(steering_vectors) != len(layers):
            raise ValueError(
                f"Loaded {len(steering_vectors)} steering vectors, but model has {len(layers)} layers."
            )
        steering_meta.setdefault("source", "loaded_precomputed_vector")
        steering_meta.setdefault("steering_vector_path", args.steering_vector_path)

    steering_vectors = [v.to(device=device) for v in steering_vectors]

    generator = UTRGenerator(
        model=model,
        vocab=vocab,
        device=device,
        steering_controller=controller,
        steering_vectors=steering_vectors,
        alpha=args.alpha,
    )

    print("\n[Generation] Starting activation-steered reference-guided generation...")
    print(f"[Reference CSV] {args.reference_csv}")
    print(f"[Output CSV] {args.out_csv}")
    if compute_steering_from_csv:
        print(f"[Positive steering CSV] {steering_meta.get('positive_csv', args.positive_steering_csv)}")
        print(f"[Negative steering CSV] {steering_meta.get('negative_csv', args.negative_steering_csv)}")
    else:
        print(f"[Steering vector] {steering_meta.get('steering_vector_path', args.steering_vector_path)}")
    print(f"[Direction] positive mean - negative mean")
    steering_reference_beta_min = None if args.steer_all_references else args.steer_reference_beta_min
    if steering_reference_beta_min is None:
        print("[Reference steering filter] disabled; steering all references")
    else:
        print(
            f"[Reference steering filter] steer only when "
            f"{args.steer_reference_beta_col} > {steering_reference_beta_min}"
        )
    print(
        f"[Generation params] steps={args.steps}, mask_ratio={args.mask_ratio}, "
        f"strategy={args.strategy}, temperature={args.temperature}, top_p={args.top_p}, "
        f"alpha={args.alpha}"
    )
    if args.diso_mode and float(args.alpha) == 0.0:
        print(
            "[DISO alpha=0 behavior] steering vectors are still computed/loaded for DISO site selection; "
            "MASK filling is performed without activation addition."
        )
    if args.diso_mode:
        print(
            f"[DISO params] R={args.R}, T={args.T}, "
            f"relatedness_layer={args.relatedness_layer}, "
            f"allow_reselect_positions={args.allow_reselect_positions}"
        )

    try:
        generate_csv_from_references(
            generator=generator,
            reference_csv=args.reference_csv,
            out_csv=args.out_csv,
            seq_col=args.seq_col,
            steps=args.steps,
            mask_ratio=args.mask_ratio,
            temperature=args.temperature,
            strategy=args.strategy,
            top_p=args.top_p,
            base_count_penalty=args.base_count_penalty,
            start_index=args.start_index,
            num_refs=args.num_refs,
            num_samples_per_ref=args.num_samples_per_ref,
            skip_invalid=args.skip_invalid,
            save_mask_history=args.save_mask_history,
            debug=args.debug,
            progress_every=args.progress_every,
            steering_meta=steering_meta,
            stream_generated=args.stream_generated,
            steering_reference_beta_col=args.steer_reference_beta_col,
            steering_reference_beta_min=steering_reference_beta_min,
            reference_sample_seed=args.seed,
            diso_mode=args.diso_mode,
            R=args.R,
            T=args.T,
            relatedness_layer=args.relatedness_layer,
            allow_reselect_positions=args.allow_reselect_positions,
        )
    finally:
        controller.remove_hooks()


if __name__ == "__main__":
    main()
