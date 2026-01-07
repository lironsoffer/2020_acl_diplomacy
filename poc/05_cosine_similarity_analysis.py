"""Step 5: Unsupervised Embedding Analysis (Cosine Similarity)

Extract deception/truth concept vectors, calculate cosine similarities,
analyze distributions, and create visualizations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    CLEANED_MESSAGES_CSV,
    ENRICHED_MESSAGE_EMBEDDINGS,
    MESSAGE_EMBEDDINGS,
    COSINE_SIMILARITY_ANALYSIS,
    COSINE_SIMILARITY_HISTOGRAM,
    COSINE_SIMILARITY_SCATTER,
    ENRICHED_VS_ORIGINAL_COMPARISON,
    RESULTS_DIR
)
from utils import load_csv, load_numpy, save_json, logger

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def extract_deception_concept(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute mean embedding for deceptive messages."""
    lie_indices = np.where(labels == 1)[0]
    if len(lie_indices) == 0:
        logger.warning("No lies found in labels")
        return np.zeros(embeddings.shape[1])
    
    deception_concept = np.mean(embeddings[lie_indices], axis=0)
    logger.info(f"Extracted deception concept vector: shape {deception_concept.shape}, "
                f"from {len(lie_indices)} deceptive messages")
    return deception_concept


def extract_truth_concept(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute mean embedding for truthful messages."""
    truth_indices = np.where(labels == 0)[0]
    if len(truth_indices) == 0:
        logger.warning("No truths found in labels")
        return np.zeros(embeddings.shape[1])
    
    truth_concept = np.mean(embeddings[truth_indices], axis=0)
    logger.info(f"Extracted truth concept vector: shape {truth_concept.shape}, "
                f"from {len(truth_indices)} truthful messages")
    return truth_concept


def calculate_cosine_similarities(embeddings: np.ndarray, concept_vector: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between each embedding and concept vector."""
    # Normalize concept vector
    concept_norm = concept_vector / (np.linalg.norm(concept_vector) + 1e-9)
    
    # Normalize embeddings
    embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (embedding_norms + 1e-9)
    
    # Calculate cosine similarities
    similarities = np.dot(embeddings_norm, concept_norm)
    
    return similarities


def plot_similarity_histogram(similarities_deception: np.ndarray,
                              similarities_truth: np.ndarray,
                              labels: np.ndarray,
                              save_path: Path):
    """Plot histogram of cosine similarities to deception concept."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Separate by label
    lie_similarities = similarities_deception[labels == 1]
    truth_similarities = similarities_deception[labels == 0]
    
    # Histogram
    axes[0].hist(truth_similarities, bins=50, alpha=0.7, label='Truths', color='blue', density=True)
    axes[0].hist(lie_similarities, bins=50, alpha=0.7, label='Lies', color='red', density=True)
    axes[0].set_xlabel('Cosine Similarity to Deception Concept')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of Similarities to Deception Concept')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [truth_similarities, lie_similarities]
    axes[1].boxplot(data_to_plot, labels=['Truths', 'Lies'])
    axes[1].set_ylabel('Cosine Similarity to Deception Concept')
    axes[1].set_title('Box Plot: Similarities to Deception Concept')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved similarity histogram to {save_path}")


def plot_similarity_scatter(similarities_deception: np.ndarray,
                           similarities_truth: np.ndarray,
                           labels: np.ndarray,
                           save_path: Path):
    """Plot scatter plot: similarity to deception vs similarity to truth."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate by label
    lie_mask = labels == 1
    truth_mask = labels == 0
    
    ax.scatter(similarities_truth[truth_mask], similarities_deception[truth_mask],
              alpha=0.5, label='Truths', s=20, color='blue')
    ax.scatter(similarities_truth[lie_mask], similarities_deception[lie_mask],
              alpha=0.5, label='Lies', s=20, color='red')
    
    ax.set_xlabel('Cosine Similarity to Truth Concept')
    ax.set_ylabel('Cosine Similarity to Deception Concept')
    ax.set_title('Similarity to Truth vs Similarity to Deception')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved similarity scatter plot to {save_path}")


def plot_tsne_visualization(embeddings: np.ndarray, labels: np.ndarray,
                           similarities_deception: np.ndarray,
                           save_path: Path, n_samples: int = 2000):
    """Plot t-SNE visualization colored by cosine similarity to deception concept."""
    logger.info("Computing t-SNE (this may take a while)...")
    
    # Sample if too many points
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_sample = embeddings[indices]
        labels_sample = labels[indices]
        similarities_sample = similarities_deception[indices]
    else:
        embeddings_sample = embeddings
        labels_sample = labels
        similarities_sample = similarities_deception
    
    # Apply PCA first for speed
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings_sample)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_tsne = tsne.fit_transform(embeddings_pca)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                        c=similarities_sample, cmap='RdYlBu_r',
                        s=20, alpha=0.6, edgecolors='none')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cosine Similarity to Deception Concept')
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Visualization (Colored by Similarity to Deception Concept)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved t-SNE visualization to {save_path}")


def compare_enriched_vs_original(enriched_similarities: np.ndarray,
                                original_similarities: np.ndarray,
                                labels: np.ndarray,
                                save_path: Path):
    """Compare enriched vs original embedding similarities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Separate by label
    lie_mask = labels == 1
    truth_mask = labels == 0
    
    # Plot for lies
    axes[0].scatter(original_similarities[lie_mask], enriched_similarities[lie_mask],
                   alpha=0.5, s=20, color='red', label='Lies')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
    axes[0].set_xlabel('Original Embeddings Similarity')
    axes[0].set_ylabel('Enriched Embeddings Similarity')
    axes[0].set_title('Enriched vs Original: Lies')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot for truths
    axes[1].scatter(original_similarities[truth_mask], enriched_similarities[truth_mask],
                   alpha=0.5, s=20, color='blue', label='Truths')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
    axes[1].set_xlabel('Original Embeddings Similarity')
    axes[1].set_ylabel('Enriched Embeddings Similarity')
    axes[1].set_title('Enriched vs Original: Truths')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved enriched vs original comparison to {save_path}")


def main():
    """Main function for cosine similarity analysis."""
    logger.info("Starting Step 5: Cosine Similarity Analysis")
    
    # Load data
    logger.info("Loading enriched embeddings and labels...")
    enriched_embeddings = load_numpy(ENRICHED_MESSAGE_EMBEDDINGS)
    df = load_csv(CLEANED_MESSAGES_CSV)
    
    # Get labels for all messages
    labels = df['label'].values
    
    logger.info(f"Loaded {len(enriched_embeddings)} enriched embeddings")
    logger.info(f"Lies: {np.sum(labels == 1)}, Truths: {np.sum(labels == 0)}")
    
    # Extract concept vectors
    logger.info("Extracting concept vectors...")
    deception_concept = extract_deception_concept(enriched_embeddings, labels)
    truth_concept = extract_truth_concept(enriched_embeddings, labels)
    
    # Calculate cosine similarities
    logger.info("Calculating cosine similarities to deception concept...")
    similarities_deception = calculate_cosine_similarities(enriched_embeddings, deception_concept)
    
    logger.info("Calculating cosine similarities to truth concept...")
    similarities_truth = calculate_cosine_similarities(enriched_embeddings, truth_concept)
    
    # Calculate statistics
    lie_mask = labels == 1
    truth_mask = labels == 0
    
    stats = {
        'deception_concept': {
            'mean_similarity_lies': float(np.mean(similarities_deception[lie_mask])),
            'mean_similarity_truths': float(np.mean(similarities_deception[truth_mask])),
            'median_similarity_lies': float(np.median(similarities_deception[lie_mask])),
            'median_similarity_truths': float(np.median(similarities_deception[truth_mask])),
            'std_similarity_lies': float(np.std(similarities_deception[lie_mask])),
            'std_similarity_truths': float(np.std(similarities_deception[truth_mask]))
        },
        'truth_concept': {
            'mean_similarity_lies': float(np.mean(similarities_truth[lie_mask])),
            'mean_similarity_truths': float(np.mean(similarities_truth[truth_mask])),
            'median_similarity_lies': float(np.median(similarities_truth[lie_mask])),
            'median_similarity_truths': float(np.median(similarities_truth[truth_mask])),
            'std_similarity_lies': float(np.std(similarities_truth[lie_mask])),
            'std_similarity_truths': float(np.std(similarities_truth[truth_mask]))
        }
    }
    
    # Find threshold (where deception similarity indicates a lie)
    # Use median as a simple threshold
    threshold = np.median(similarities_deception)
    stats['threshold'] = float(threshold)
    
    # Calculate accuracy at threshold
    predictions = (similarities_deception > threshold).astype(int)
    threshold_accuracy = float(np.mean(predictions == labels))
    stats['threshold_accuracy'] = threshold_accuracy
    
    logger.info(f"Threshold: {threshold:.4f}, Accuracy at threshold: {threshold_accuracy:.4f}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    plot_similarity_histogram(similarities_deception, similarities_truth, labels,
                            COSINE_SIMILARITY_HISTOGRAM)
    plot_similarity_scatter(similarities_deception, similarities_truth, labels,
                          COSINE_SIMILARITY_SCATTER)
    
    # Compare with original embeddings if available
    if MESSAGE_EMBEDDINGS.exists():
        logger.info("Loading original embeddings for comparison...")
        original_embeddings = load_numpy(MESSAGE_EMBEDDINGS)
        original_deception_concept = extract_deception_concept(original_embeddings, labels)
        original_similarities = calculate_cosine_similarities(original_embeddings, original_deception_concept)
        
        compare_enriched_vs_original(similarities_deception, original_similarities, labels,
                                    ENRICHED_VS_ORIGINAL_COMPARISON)
        
        stats['enriched_vs_original'] = {
            'mean_similarity_enriched_lies': float(np.mean(similarities_deception[lie_mask])),
            'mean_similarity_original_lies': float(np.mean(original_similarities[lie_mask])),
            'mean_similarity_enriched_truths': float(np.mean(similarities_deception[truth_mask])),
            'mean_similarity_original_truths': float(np.mean(original_similarities[truth_mask]))
        }
    
    # Save analysis
    save_json(stats, COSINE_SIMILARITY_ANALYSIS)
    
    logger.info("\n" + "="*70)
    logger.info("Step 5 Summary")
    logger.info("="*70)
    logger.info(f"Mean similarity to deception concept - Lies: {stats['deception_concept']['mean_similarity_lies']:.4f}, "
                f"Truths: {stats['deception_concept']['mean_similarity_truths']:.4f}")
    logger.info(f"Mean similarity to truth concept - Lies: {stats['truth_concept']['mean_similarity_lies']:.4f}, "
                f"Truths: {stats['truth_concept']['mean_similarity_truths']:.4f}")
    logger.info(f"Threshold accuracy: {threshold_accuracy:.4f}")
    logger.info("="*70)
    logger.info("Step 5 completed successfully!")


if __name__ == "__main__":
    main()

