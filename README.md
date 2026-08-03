# EukaUTR

**EukaUTR: a foundation model for unified functional modelling and design of eukaryotic 3′ UTRs**

<p align="center">
  <img src="fig/Fig1_overview.jpg" alt="EukaUTR model overview" width="100%">
</p>

EukaUTR is a 3′ UTR-specific RNA foundation model for learning transferable regulatory representations from eukaryotic 3′ untranslated region sequences. The framework connects four complementary capabilities: **functional prediction**, **local cis-regulatory analysis**, **de novo 3′ UTR generation**, and **property-directed sequence optimization with EukaUTR-Guide**.

EukaUTR uses a 12-layer Transformer encoder with 12 attention heads, a hidden dimension of 768, approximately 87 million parameters, and a maximum input length of 1,024 tokens including special tokens. The model is pretrained with masked language modelling at single-nucleotide resolution.

---

## Overview

### Species-disjoint two-stage pretraining

EukaUTR was trained using two non-overlapping pretraining stages:

| Stage        | Model                     | Dataset                          | Coverage                                                                                            | Pretraining data                                         |
| ------------ | ------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Stage 1      | **EukaUTR-S1**      | 7,025,323 non-redundant 3′ UTRs | 1,715 eukaryotic genomes spanning vertebrates, non-vertebrate metazoans, plants, fungi and protists | [Stage 1](https://zenodo.org/uploads/21790090)            |
| Stage 2      | **EukaUTR-S2**      | 59,221 curated 3′ UTRs          | Human, mouse, rat and zebrafish; these four species were excluded from Stage 1                      | [Stage 2](https://zenodo.org/uploads/21790090)            |
| Stage 2 LoRA | **EukaUTR-S2-LoRA** | Same Stage 2 corpus              | Parameter-efficient continued pretraining from EukaUTR-S1                                           | Same [Stage 2](https://zenodo.org/uploads/21790090) data |

Together, the two stages contain **7,084,544 non-redundant sequences from 1,719 unique genomes**. Stage 1 provides broad eukaryotic diversity, whereas Stage 2 provides a smaller high-confidence vertebrate corpus for continued pretraining.

### Downstream evaluation

The manuscript evaluates EukaUTR across **13 downstream prediction tasks from nine source datasets**, spanning:

- RBP binding on eCLIP and gene-disjoint CLIP benchmarks
- m6A modification-site prediction
- alternative polyadenylation isoform usage
- zebrafish reporter degradation rate and derived half-life
- human fast-UTR reporter half-life
- breast-cancer SLAM-seq cross-context relative mRNA decay
- full-length human 3′ UTR protein output, RNA abundance and translational efficiency
- viral MPRA RNA abundance and mean ribosome load (MRL)

Across these benchmarks, EukaUTR achieved leading or competitive performance and outperformed the strongest external baseline in **12 of 13 tasks**, with relative gains of up to **27.45%**.

### Local regulatory analysis

Beyond transcript-level prediction, the manuscript evaluates whether EukaUTR captures local cis-regulatory information. EukaUTR localized elevated mutational sensitivity to the **CXCL2 ARE1** region and recognized **AUA-rich sequence contexts associated with RBMS3 binding**.

### De novo 3′ UTR generation

EukaUTR supports template-free iterative masked generation at user-defined target lengths. Generated sequences were evaluated for:

- GC content and sequence-length-matched composition
- 3–6-mer frequency fidelity
- RNAfold-predicted minimum free energy
- canonical PAS, ARE, CPE and PRE motif densities
- human miRNA 8-mer target-site density
- embedding-space similarity to natural 3′ UTRs
- homopolymer and low-complexity properties
- within-set redundancy and similarity to the pretraining corpus

EukaUTR-generated sequences reproduced multiple statistical and regulatory properties of natural 3′ UTRs without detectable high-coverage internal duplication or training-set near-duplicates under the reported search criteria.

### EukaUTR-Guide

**EukaUTR-Guide** is a representation-guided editing framework for property-directed 3′ UTR optimization. It separates:

1. **property-informed site selection**, using a representation direction derived from sequences with contrasting predicted degradation rates; and
2. **context-aware nucleotide decoding**, using the pretrained masked language model.

The representation direction is used to rank candidate editing positions only; it is **not added to hidden representations during nucleotide decoding**. In the manuscript, EukaUTR-Guide reduced the median predicted degradation rate of high-rate reference sequences by **37.2%** and reduced the predicted degradation rate in **89.1%** of high-rate references.

---

## Installation

Linux with a CUDA-enabled GPU is recommended. CPU inference is supported but will be slower for long sequences or large batches.

### From source

```bash
git clone https://github.com/meilanglang/EukaUTR.git
cd EukaUTR
conda env create -f environment.yaml
conda activate eukautr
```

If the repository is already checked out inside a larger workspace:

```bash
cd EukaUTR
conda env create -f environment.yaml
conda activate eukautr
```

### Key dependencies

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

## Model checkpoints

Pretrained model checkpoints are available from [Hugging Face: langmei/EukaUTR](https://huggingface.co/langmei/EukaUTR). Download the required files from the Hugging Face links below; the example commands assume the downloaded files are placed under the same relative paths in this repository.

| Model                             | Purpose                                                                        | Hugging Face link                                                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **EukaUTR-S1**              | Stage 1 broad-eukaryotic checkpoint                                            | [EukaUTR-S1](https://huggingface.co/langmei/EukaUTR/blob/main/checkpoints/S1/EukaUTR-S1.ckpt)                                                             |
| **EukaUTR-S2**              | Full-parameter Stage 2 checkpoint; default for the provided inference examples | [EukaUTR-S2](https://huggingface.co/langmei/EukaUTR/blob/main/checkpoints/S2/EukaUTR-S2.ckpt)                                                             |
| **EukaUTR-S2-LoRA**         | LoRA-based Stage 2 checkpoint                                                  | [EukaUTR-S2-LoRA](https://huggingface.co/langmei/EukaUTR/blob/main/checkpoints/LoRA/EukaUTR-LoRA.ckpt)                                                    |
| **EukaUTR-Guide direction** | Precomputed representation direction for the packaged optimization workflow    | [`utr_optimazaition/guided_vector/EukaUTR_Guide.pt`](https://huggingface.co/langmei/EukaUTR/blob/main/utr_optimazaition/guided_vector/EukaUTR_Guide.pt) |

> **Note:** the packaged EukaUTR-Guide direction is stored as `utr_optimazaition/guided_vector/EukaUTR_Guide.pt`.

Example layout:

```text
EukaUTR/
  checkpoints/
    S1/EukaUTR-S1.ckpt
    S2/EukaUTR-S2.ckpt
    LoRA/EukaUTR-LoRA.ckpt
  utr_optimazaition/
    guided_vector/EukaUTR_Guide.pt
```

---

## Data downloads

Processed pretraining and downstream benchmark datasets are available from [Zenodo upload 21790090](https://zenodo.org/uploads/21790090). The Zenodo record includes:

- processed Stage 1 and Stage 2 pretraining corpora
- downstream task datasets used for benchmark evaluation

Large datasets are not stored directly in this Git repository. Download the Zenodo archive and unpack it into the repository root when reproducing pretraining, downstream fine-tuning or benchmark analyses.

---

## Quick start

### 1. Extract EukaUTR representations

Extract an embedding for a single DNA/RNA sequence:

```bash
python extract_embedding.py \
  --sequence AUCGAUCGGAUUCGUUUCCCGGGG \
  --output results/example_embedding.pt
```

Extract embeddings from FASTA:

```bash
python extract_embedding.py \
  --input sequences.fasta \
  --input-format fasta \
  --output results/fasta_embeddings.pt \
  --batch-size 8
```

Extract embeddings from a CSV containing `id` and `sequence` columns:

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

The extraction utility supports pooled sequence representations. In the manuscript's downstream fine-tuning experiments, the final-layer `<cls>` representation was used as the sequence-level embedding unless otherwise specified.

### 2. De novo 3′ UTR generation

Generate 10,000 sequences with target lengths sampled from 100 to 1,000 nt:

```bash
python utr_generation.py \
  --target-lengths 100-1000 \
  --num-sequences 10000 \
  --output results/EukaUTR_sequence.csv \
  --checkpoint checkpoints/S2/EukaUTR-S2.ckpt \
  --batch-size 16
```

Generate a small fixed-length set:

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

Example `targets.csv`:

```csv
id,target_length
seq_001,110
seq_002,250
```

Generation is template-free and uses iterative masked decoding restricted to canonical RNA nucleotides. The output contains `id`, `target_length`, `sequence`, and `generated_length`. Existing valid output rows are resumed/skipped by default; use `--overwrite` to restart.

### 3. EukaUTR-Guide property-directed optimization

Optimize reference 3′ UTRs using the packaged EukaUTR-Guide representation direction:

```bash
python utr_optimization.py \
  --reference_csv refs.csv \
  --seq_col sequence \
  --out_csv results/eukautr_guide_sequences.csv \
  --ckpt_path checkpoints/S2/EukaUTR-S2.ckpt \
  --steering_vector_path utr_optimazaition/guided_vector/EukaUTR_Guide.pt \
  --alpha 0.0 \
  --R 8 \
  --T 4
```

Example `refs.csv`:

```csv
id,sequence
ref_001,TTTTTTCTTTAAAAACAAATTAGGATTTTTTTTTTTTTT
```

DNA inputs are normalized internally by converting `T` to `U`. In the manuscript configuration, four positions are selected per round (`T = 4`) for eight iterative editing rounds (`R = 8`). The property direction is used for **site selection**, while nucleotide identities are proposed by the masked language model with no activation shift during decoding (`alpha = 0`).

### 4. Build a custom property direction

A task-specific representation direction can be constructed from positive and negative sequence sets:

```bash
python utr_optimization.py \
  --reference_csv refs.csv \
  --seq_col sequence \
  --positive_steering_csv utr_optimazaition/streering_data/CCLE_UTR/steering_sets/positive_steering_q5_low_beta.csv \
  --negative_steering_csv utr_optimazaition/streering_data/CCLE_UTR/steering_sets/negative_steering_q5_high_beta.csv \
  --save_steering_vector_path utr_optimazaition/guided_vector/custom_task.pt \
  --out_csv results/eukautr_guide_custom.csv \
  --ckpt_path checkpoints/S2/EukaUTR-S2.ckpt \
  --alpha 0.0 \
  --R 8 \
  --T 4
```

The representation direction is computed as:

```text
mean_positive_representation - mean_negative_representation
```

For optimization toward a desired property, the positive set should represent the desired state and the negative set the contrasting state. The direction is then used to prioritize candidate nucleotide positions for masked decoding.

---

## Main scripts

| Script                   | Function                                                           |
| ------------------------ | ------------------------------------------------------------------ |
| `extract_embedding.py` | Extract pooled EukaUTR sequence representations.                   |
| `utr_generation.py`    | Generate de novo 3′ UTR sequences from target lengths.            |
| `utr_optimization.py`  | Perform EukaUTR-Guide representation-guided sequence optimization. |

---

## Data and model availability

- **Code:** https://github.com/meilanglang/EukaUTR
- **Pretrained model checkpoints:** https://huggingface.co/langmei/EukaUTR
- **Processed pretraining and downstream benchmark datasets:** https://zenodo.org/uploads/21790090

The Zenodo deposit contains the processed Stage 1 and Stage 2 pretraining corpora, downstream benchmark datasets, and associated metadata used in the study. Original source data remain available from the databases and studies cited in the manuscript.

---

## Scope and interpretation

EukaUTR is primarily a sequence-based model. The current framework does not explicitly model promoter context, coding-region interactions, RNA modifications, trans-factor abundance, binding occupancy or dynamic cellular state.

The de novo generation analyses establish statistical fidelity and sequence-level novelty under the reported computational criteria, but do not by themselves establish biological function. Likewise, EukaUTR-Guide optimization is currently evaluated computationally. In the manuscript, Guided editing was accompanied by increased GC content, and GC enrichment accounted for a substantial component of the predicted optimization effect. Prospective experimental validation is therefore required before interpreting generated or optimized sequences as biologically stabilized 3′ UTRs.

---

## Citation

If you use EukaUTR in your research, please cite the accompanying manuscript:

```bibtex
@misc{lang2026eukautr,
  title  = {EukaUTR: a foundation model for unified functional modelling and design of eukaryotic 3' UTRs},
  author = {Lang, Mei and Fang, Xingyu and Chen, Mingxuan and Wang, Zhen and Cheng, Zhaowen and Zhu, Xiagu and Tam, Kin Yip and Zhang, Junwei and Li, Xiaolin},
  year   = {2026},
  note   = {Manuscript}
}
```

Citation metadata can be updated after formal publication.

---

## License

This repository is released under the **MIT License**.

---

## Acknowledgments

EukaUTR builds on open-source Transformer and RNA language-modeling utilities for nucleotide sequence modelling. We thank the developers and maintainers of the software packages and public datasets used throughout this project.
