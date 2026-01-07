# Lie Detection POC

This directory contains the Basic Proof of Concept implementation for characterizing deceptive behavior in the "It Takes Two to Lie" Diplomacy dataset.

## Overview

This POC implements a 5-step analysis pipeline:
1. **Data Ingestion & Sanitation**: Load ConvoKit corpus and move data
2. **Baseline Behavioral Metrics**: Calculate deception rates and statistics
3. **Training the Initial Probe**: Train message-only lie detection probe
4. **Training the Enriched Probe**: Train probe with move context
5. **Cosine Similarity Analysis**: Analyze deception concepts in embedding space

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Execution

Run scripts sequentially:
```bash
python poc/01_data_ingestion.py
python poc/02_baseline_metrics.py
python poc/03_train_probe.py
python poc/04_train_enriched_probe.py
python poc/05_cosine_similarity_analysis.py
```

## Output Structure

- `data/`: Intermediate data files (cleaned messages, embeddings)
- `results/`: Analysis outputs (metrics, visualizations)
- `models/`: Trained probe models

## Key Features

- Uses ConvoKit for easy data access
- BERT embeddings with [CLS] token and mean pooling
- Linear probe classifiers
- Enriched embeddings incorporating move context
- Cosine similarity analysis for deception concepts

