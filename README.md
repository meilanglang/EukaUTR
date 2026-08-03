# EukaUTR

**EukaUTR: A Eukaryotic 3'UTR Foundation Model for Representation Learning, De Novo Generation, and Sequence Optimization**

EukaUTR is an RNA language model designed for eukaryotic 3' untranslated regions (3'UTRs). The model learns contextual representations of UTR sequences and supports three practical workflows: extracting sequence-level embeddings for downstream prediction tasks, generating 3'UTR sequences from scratch with controllable lengths, and optimizing reference 3'UTRs with activation steering and DISO-style site selection.

The current implementation uses a 12-layer Transformer backbone with 768 hidden dimensions, 12 attention heads, and a maximum sequence length of 1,024 nt. Packaged scripts provide ready-to-run inference utilities for feature extraction, de novo generation, and reference-guided optimization.

---

## Highlights

- **3'UTR-focused language modeling** - Models eukaryotic UTR sequences with an RNA alphabet and masked-token infilling objective.
- **Two-stage checkpoints** - Provides S1 and S2 checkpoints for EukaUTR backbone inference, with S2 used as the default checkpoint in the provided scripts.
- **Sequence representation extraction** - Produces mean-pooled or CLS sequence embeddings for CSV, FASTA, TXT, or single-sequence inputs.
- **De novo 3'UTR generation** - Generates new 3'UTRs with user-specified target lengths, batched decoding, temperature sampling, top-p truncation, and resumable CSV output.
- **Activation-steered optimization** - Optimizes reference 3'UTRs using a precomputed steering vector or task-specific vectors computed from positive and negative sequence sets.
- **DISO site selection** - Supports reference-guided mutation site selection using token-level relatedness to the steering direction.

---

## Installation

We recommend running EukaUTR on Linux with a CUDA-enabled GPU. CPU inference is supported but will be slower for long sequences or large batches.

### From source

```bash
git clone https://github.com/xxx/EukaUTR.git
cd EukaUTR
conda env create -f environment.yaml
conda activate eukautr
```

If this repository is already checked out inside a larger workspace, enter the project directory directly:

```bash
cd EukaUTR
conda env create -f environment.yaml
conda activate eukautr
```

### Key Dependencies

```text
python=3.8.5
pytorch=1.7.1
cudatoolkit=11.0
numpy==1.19.2
pandas==1.3.1
biopython==1.78
pytorch-lightning==1.5.10
rna-fm==0.1.0
```

---

## Model Checkpoints

Place downloaded checkpoints under the following paths. If the checkpoint files are already included in the repository, this step can be skipped.

| Model | Purpose | Local Path | Download |
| --- | --- | --- | --- |
| EukaUTR-S1 | Stage-1 backbone checkpoint | `checkpoints/S1/EukaUTR-S1.ckpt` | Coming soon |
| EukaUTR-S2 | Stage-2 backbone checkpoint; default for inference | `checkpoints/S2/EukaUTR-S2.ckpt` | Coming soon |
| EukaUTR-LoRA | LoRA-adapted checkpoint | `checkpoints/LoRA/EukaUTR-LoRA.ckpt` | Coming soon |
| EukaUTR-DISO steering vector | Default activation-steering vector for optimization | `utr_optimazaition/steering_vector/EukaUTR_DISO.pt` | Included / Coming soon |

Example layout:

```text
EukaUTR/
  checkpoints/
    S1/EukaUTR-S1.ckpt
    S2/EukaUTR-S2.ckpt
    LoRA/EukaUTR-LoRA.ckpt
  utr_optimazaition/
    steering_vector/EukaUTR_DISO.pt
```

---

## Quick Start

### 1. Feature Extraction

Extract a sequence embedding for a single DNA/RNA sequence:

```bash
python extract_embedding.py \
  --sequence AUCGAUCGGAUUCGUUUCCCGGGG \
  --output results/example_embedding.pt
```

Extract embeddings from a FASTA file:

```bash
python extract_embedding.py \
  --input sequences.fasta \
  --input-format fasta \
  --output results/fasta_embeddings.pt \
  --batch-size 8
```

Extract embeddings from a CSV file with `id` and `sequence` columns:

```bash
python extract_embedding.py \
  --input sequences.csv \
  --input-format csv \
  --id-column id \
  --seq-column sequence \
  --output results/csv_embeddings.pt \
  --checkpoint checkpoints/S2/EukaUTR-S2.ckpt
```

The output `.pt` file contains:

```text
ids, sequences, lengths, embeddings, repr_layer, pooling, checkpoint
```

Default extraction uses layer 12 and mean pooling over biological sequence tokens.

### 2. De Novo 3'UTR Generation

Generate 3'UTRs with lengths uniformly sampled from 100 to 1,000 nt:

```bash
python utr_generation.py \
  --target-lengths 100-1000 \
  --num-sequences 10000 \
  --output results/EukaUTR_sequence.csv \
  --checkpoint checkpoints/S2/EukaUTR-S2.ckpt \
  --batch-size 16
```

Generate a small fixed-length test set:

```bash
python utr_generation.py \
  --target-lengths 110,150,250 \
  --output results/EukaUTR_fixed_lengths.csv \
  --overwrite
```

Generate from a target-length CSV:

```bash
python utr_generation.py \
  --target-lengths targets.csv \
  --id-column id \
  --length-column target_length \
  --output results/EukaUTR_from_targets.csv
```

`targets.csv` should contain:

```csv
id,target_length
seq_001,110
seq_002,250
```

The generation output contains `id`, `target_length`, `sequence`, and `generated_length`. Existing valid output rows are resumed/skipped by default; use `--overwrite` to restart.

### 3. Reference-Guided Sequence Optimization

Optimize reference 3'UTRs with the packaged DISO steering vector:

```bash
python utr_optimization.py \
  --reference_csv refs.csv \
  --seq_col sequence \
  --out_csv results/eukautr_optimized_sequences.csv \
  --ckpt_path checkpoints/S2/EukaUTR-S2.ckpt \
  --steering_vector_path utr_optimazaition/steering_vector/EukaUTR_DISO.pt \
  --alpha 1.0 \
  --R 8 \
  --T 4
```

`refs.csv` should contain at least a sequence column:

```csv
id,sequence
ref_001,TTTTTTCTTTAAAAACAAATTAGGATTTTTTTTTTTTTT
```

Input DNA sequences are normalized internally by converting `T` to `U`; optimized outputs include RNA and DNA-style sequence fields plus sequence statistics and mutation metadata.

### 4. Task-Specific Steering Vector

You can compute a steering vector from positive and negative sequence sets, then use it for optimization:

```bash
python utr_optimization.py \
  --reference_csv refs.csv \
  --seq_col sequence \
  --positive_steering_csv utr_optimazaition/streering_data/CCLE_UTR/steering_sets/positive_steering_q5_low_beta.csv \
  --negative_steering_csv utr_optimazaition/streering_data/CCLE_UTR/steering_sets/negative_steering_q5_high_beta.csv \
  --save_steering_vector_path utr_optimazaition/steering_vector/custom_task.pt \
  --out_csv results/eukautr_optimized_custom.csv \
  --ckpt_path checkpoints/S2/EukaUTR-S2.ckpt \
  --alpha 1.0
```

The steering direction is:

```text
mean_positive_activation - mean_negative_activation
```

To steer toward a desired property, put sequences with the desired property in the positive CSV and counterexamples in the negative CSV.

---

## Main Scripts

| Script | Function |
| --- | --- |
| `extract_embedding.py` | Extract pooled EukaUTR embeddings from sequence inputs. |
| `utr_generation.py` | Generate new 3'UTR sequences from target lengths. |
| `utr_optimization.py` | Optimize reference 3'UTRs with activation steering and DISO site selection. |

---

## Citation

If you find EukaUTR useful in your research, please cite:

```bibtex
@article{eukautr2025,
  title={EukaUTR: A Eukaryotic 3'UTR Foundation Model for Representation Learning and Controllable UTR Design},
  author={xxx},
  journal={xxx},
  year={2025}
}
```

---

## License

License terms will be provided with the public release.

---

## Acknowledgments

EukaUTR builds on RNA language-modeling utilities and Transformer components adapted for nucleotide sequence modeling. We thank the developers of open-source RNA and protein language model toolkits for their contributions to the community.
