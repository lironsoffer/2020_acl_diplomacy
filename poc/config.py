"""Configuration settings for the Lie Detection POC."""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent
POC_DIR = Path(__file__).parent

# Data paths
DATA_DIR = POC_DIR / "data"
RESULTS_DIR = POC_DIR / "results"
MODELS_DIR = POC_DIR / "models"
MOVES_DIR = BASE_DIR / "moves"

# Output file paths
CLEANED_MESSAGES_CSV = DATA_DIR / "cleaned_messages.csv"
MOVES_DATA_JSON = DATA_DIR / "moves_data.json"

# Embedding file paths
MESSAGE_EMBEDDINGS = DATA_DIR / "message_embeddings.npy"
MESSAGE_EMBEDDINGS_CLS = DATA_DIR / "message_embeddings_cls.npy"
MESSAGE_EMBEDDINGS_MEAN = DATA_DIR / "message_embeddings_mean.npy"
ENRICHED_MESSAGE_EMBEDDINGS = DATA_DIR / "enriched_message_embeddings.npy"

# Model paths
PROBE_MESSAGE_ONLY = MODELS_DIR / "lie_detector_probe_message_only.pkl"
PROBE_CLS_ONLY = MODELS_DIR / "lie_detector_probe_cls_only.pkl"
PROBE_MEAN_ONLY = MODELS_DIR / "lie_detector_probe_mean_only.pkl"
PROBE_ENRICHED = MODELS_DIR / "lie_detector_probe_enriched.pkl"

# Results paths
BASELINE_METRICS_JSON = RESULTS_DIR / "baseline_metrics.json"
BASELINE_METRICS_TXT = RESULTS_DIR / "baseline_metrics.txt"
PROBE_METRICS_MESSAGE_ONLY = RESULTS_DIR / "probe_metrics_message_only.json"
PROBE_METRICS_ENRICHED = RESULTS_DIR / "probe_metrics_enriched.json"
PROBE_COMPARISON_JSON = RESULTS_DIR / "probe_comparison.json"
PROBE_COMPARISON_PNG = RESULTS_DIR / "probe_comparison.png"
CLS_VS_MEAN_COMPARISON_JSON = RESULTS_DIR / "cls_vs_mean_comparison.json"
CLS_VS_MEAN_COMPARISON_PNG = RESULTS_DIR / "cls_vs_mean_comparison.png"
CONFUSION_MATRIX_MESSAGE_ONLY = RESULTS_DIR / "confusion_matrix_message_only.png"
CONFUSION_MATRIX_ENRICHED = RESULTS_DIR / "confusion_matrix_enriched.png"
COSINE_SIMILARITY_ANALYSIS = RESULTS_DIR / "cosine_similarity_analysis.json"
COSINE_SIMILARITY_HISTOGRAM = RESULTS_DIR / "cosine_similarity_histogram.png"
COSINE_SIMILARITY_SCATTER = RESULTS_DIR / "cosine_similarity_scatter.png"
ENRICHED_VS_ORIGINAL_COMPARISON = RESULTS_DIR / "enriched_vs_original_comparison.png"

# Model configuration
BERT_MODEL_NAME = "bert-base-uncased"
EMBEDDING_DIM = 768  # BERT base dimension
CONCATENATED_DIM = 1536  # [CLS] + mean pooling

# Hyperparameter tuning
C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]

# Prompt template
ENRICHED_PROMPT_TEMPLATE = """Message: {message_text}

Player Actions: {move_summary}

Game State: Sender has {game_score} supply centers. Power difference: {game_score_delta}

Context: This message was sent during {year} {season} turn."""

