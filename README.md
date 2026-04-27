# CS-4063 NLP — Assignment 3: Transformer + RAG Pipeline

**FAST NUCES | Spring 2026**

A three-stage NLP pipeline for Amazon product review analysis, built entirely from scratch in PyTorch — no pretrained models, no `nn.Transformer`, no `nn.MultiheadAttention`.

---

## Pipeline Overview

```
Review Text
    │
    ▼
┌─────────────────────────────────┐
│  Part A: Encoder-Only           │
│  Transformer (4L × 8H × d=256) │
│  → Sentiment (Neg/Neu/Pos)      │
│  → Length (Short/Medium/Long)   │
│  → CLS Embedding (256-d)        │
└────────────┬────────────────────┘
             │ CLS embeddings
             ▼
┌─────────────────────────────────┐
│  Part B: Retrieval Module       │
│  Cosine similarity over 31,500  │
│  stored training embeddings     │
│  → Top-k similar reviews        │
└────────────┬────────────────────┘
             │ retrieved context
             ▼
┌─────────────────────────────────┐
│  Part C: Decoder-Only           │
│  Transformer (4L × 8H × d=256) │
│  Conditioned on: review +       │
│  sentiment + length + context   │
│  → Generated explanation        │
└─────────────────────────────────┘
```

---

## Results

| Task | Metric | Score |
|---|---|---|
| Sentiment Classification | Accuracy | 60.3% |
| Sentiment Classification | Macro F1 | 0.61 |
| Review Length Classification | Accuracy | 90.9% |
| Review Length Classification | Macro F1 | 0.90 |
| RAG Decoder | Test Perplexity | 1.45 |
| Baseline Decoder (no retrieval) | Test Perplexity | 1.51 |
| **RAG Improvement** | **PPL reduction** | **+0.07** |

> Note: Sentiment accuracy of 60.3% is on a **perfectly balanced** 3-class dataset (random chance = 33.3%). All three classes are genuinely learned — Negative F1=0.65, Neutral F1=0.47, Positive F1=0.69.

---

## Dataset

Amazon Reviews dataset (Ni et al., 2019) — 5 product categories:

| Category | Raw Reviews |
|---|---|
| Beauty | 198,475 |
| Cell Phones & Accessories | 194,340 |
| Electronics | 1,688,117 |
| Home & Kitchen | 551,466 |
| Sports & Outdoors | 296,214 |
| **Total loaded** | **2,928,612** |

**Sampled subset:** 15,000 per sentiment class → 45,000 reviews total  
**Split:** 70% train / 15% val / 15% test (stratified)

---

## Repository Structure

```
├── i23XXXX-NLP-Assignment3.ipynb   # Main notebook — run top to bottom
├── README.md
├── models/
│   ├── encoder_best.pt             # Best encoder weights
│   ├── decoder_best.pt             # Best RAG decoder weights
│   └── decoder_baseline.pt         # Baseline decoder (no retrieval)
└── results/
    ├── vocab.json                  # Vocabulary (26,531 tokens)
    ├── train_embeddings.pt         # 31,500 × 256 CLS embeddings
    ├── train_meta.csv              # Metadata for retrieval index
    ├── encoder_metrics.json
    ├── decoder_metrics.json
    ├── ablation_quantitative.json  # RAG vs baseline PPL comparison
    ├── ablation_results.json       # Qualitative ablation examples
    ├── qualitative_examples.json   # 5 generated explanations
    ├── retrieval_examples.json     # Retrieval query examples
    ├── encoder_learning_curves.png
    ├── decoder_learning_curves.png
    ├── encoder_confusion_matrices.png
    ├── retrieval_k_analysis.png
    ├── attention_layer1.png
    └── attention_layer4.png
```

---

## How to Run

### On Kaggle (recommended — GPU available)

1. Upload the notebook to Kaggle
2. Add the dataset: `ishmalfaheem/amazon-revs`
3. Enable GPU: **Settings → Accelerator → GPU T4**
4. Click **Run All**

The data path is pre-configured:
```
/kaggle/input/datasets/ishmalfaheem/amazon-revs/
```

### On Google Colab

1. Upload the notebook
2. Upload the JSON files (`Beauty_5.json`, `Cell_Phones_and_Accessories_5.json`, etc.) to `/content/`
3. Enable GPU: **Runtime → Change runtime type → T4 GPU**
4. Run All

### Locally

```bash
pip install torch torchvision numpy pandas matplotlib seaborn tqdm scikit-learn
```

Place JSON files in the same directory as the notebook, then run all cells.

---

## Model Architecture

### Encoder (Part A)
- **Type:** Encoder-only Transformer (BERT-style)
- **Layers:** 4 | **Heads:** 8 | **d_model:** 256 | **FF dim:** 512
- **Positional encoding:** Sinusoidal (fixed)
- **Classification heads:** Two independent 3-layer MLPs on [CLS] token
- **Tasks:** Sentiment (3-class) + Review Length (3-class)
- **Parameters:** ~3.2M

### Retrieval (Part B)
- **Index:** L2-normalised CLS embeddings for all 31,500 training reviews
- **Similarity:** Cosine (dot product on normalised vectors)
- **Top-k:** 3 neighbours per query
- **Batch retrieval:** Single matrix multiply (B×256)·(256×N)

### Decoder (Part C)
- **Type:** Decoder-only Transformer with causal masking
- **Layers:** 4 | **Heads:** 8 | **d_model:** 256 | **FF dim:** 512
- **Input:** Concatenated prompt (review + sentiment + length + retrieved context)
- **Sequence budget:** 160 tokens (130 prompt + 30 explanation reserved)
- **Parameters:** ~15.7M

---

## Hyperparameters

```python
# Encoder
EMBED_DIM           = 256
NUM_HEADS           = 8
FF_DIM              = 512
NUM_ENCODER_LAYERS  = 4
DROPOUT             = 0.1
ENCODER_LR          = 1e-4
NUM_EPOCHS          = 35
BATCH_SIZE          = 128
GRAD_ACCUM_STEPS    = 2      # effective batch = 256
WARMUP_STEPS        = 200

# Retrieval
TOP_K               = 3

# Decoder
DECODER_EMBED_DIM   = 256
DECODER_HEADS       = 8
DECODER_FF_DIM      = 512
NUM_DECODER_LAYERS  = 4
DECODER_SEQ_LEN     = 160
DECODER_LR          = 2e-4
DECODER_EPOCHS      = 20
```

---

## Key Design Decisions

**Why class-weighted CE instead of WeightedRandomSampler?**  
Using both simultaneously double-corrects for class imbalance, causing training collapse (observed: accuracy dropped to 21.6%). Class-weighted CE alone is sufficient and more stable.

**Why no Supervised Contrastive Loss?**  
SupCon with temperature=0.07 produces gradient overflow under FP16 AMP, degrading sentiment accuracy from 67% to 61%. Would require FP32 training or temperature ≥ 0.5 to stabilise.

**Why prompt truncation instead of sequence truncation?**  
The original approach truncated the full sequence, which often cut off all explanation tokens — the decoder's loss was effectively zero and it never trained. The fix truncates only the prompt (max 130 tokens), reserving 30 tokens for the explanation.

**Why a separate baseline decoder for ablation?**  
Using the same decoder weights for RAG and no-retrieval comparison produces 0.00 PPL difference trivially. Two separately trained decoders give a genuine measurement of retrieval's contribution.

---

## Requirements

```
torch >= 2.0
numpy
pandas
matplotlib
seaborn
tqdm
scikit-learn
```

> **Note:** AMP (Automatic Mixed Precision) is disabled in this notebook due to a Kaggle GPU driver compatibility issue (`cudaErrorNoKernelImageForDevice`). Training runs in FP32 and is ~20% slower but fully stable.

---

## References

- Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using distantly-labeled reviews. *EMNLP*.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*.
