# Machine Learning Experiments

A collection of machine learning and deep learning experiments I did focused on answering practical questions through controlled experiments.

Each experiment contains the problem/question, experimental setup, methodology, results, analysis, and conclusions.

---

## Experiments

### 2. Positional Embeddings in GPT-2
**Question:** What do GPT-2's learned positional embeddings encode, and how does the model use them?
To investigate the geometry, Fourier structure, and causal role of GPT-2's learned position vectors, and how attention turns absolute positions into relative ones.
**Folder:** [`position_embeddings`](./experiments/arithmetic_with_tokens/)
[→ View Experiment](./experiments/position_embeddings/gpt2_positional_embeddings.ipynb)

### 1. Aritmetic with Tokens
**Question:** Does GPT-2's embedding space encode arithmetic relationships?

To investigate whether arithmetic structure is directly reflected in the geometry of GPT-2's learned token embedding space.

**Folder:** [`arithmetic_with_tokens`](./experiments/arithmetic_with_tokens/)

[→ View Experiment](./experiments/arithmetic_with_tokens/arithmetic_with_tokens_report.md)

### 2. External Data Enrichment for ML
**Question:** Does enriching seed/customer data with external data improve model performance?

Investigates whether adding external information such as **weather, holidays, and road-network data** provides meaningful improvement over the original dataset and standard feature engineering.

**Folder:** [`dataset_enrichment`](./experiments/dataset_enrichment/)

[→ View Experiment](./experiments/dataset_enrichment/experiment_01/delivery_enrichment_v0_poc.md)

---

## Repository Structure

```text
ML Experiments/
│
├── experiments/
│   ├── external-data-enrichment/
│   │
│   └── ...
│
└── README.md
