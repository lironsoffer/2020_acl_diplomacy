"""Step 3: Training the Initial Probe (Message-Only)

Generate BERT embeddings, tune hyperparameters, train linear classifier probes,
and compare [CLS] vs mean pooling strategies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, Tuple

import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    CLEANED_MESSAGES_CSV,
    MESSAGE_EMBEDDINGS,
    MESSAGE_EMBEDDINGS_CLS,
    MESSAGE_EMBEDDINGS_MEAN,
    PROBE_MESSAGE_ONLY,
    PROBE_CLS_ONLY,
    PROBE_MEAN_ONLY,
    PROBE_METRICS_MESSAGE_ONLY,
    CONFUSION_MATRIX_MESSAGE_ONLY,
    CLS_VS_MEAN_COMPARISON_JSON,
    CLS_VS_MEAN_COMPARISON_PNG,
    RESULTS_DIR,
    BERT_MODEL_NAME,
    EMBEDDING_DIM,
    CONCATENATED_DIM,
    C_VALUES
)
from utils import load_csv, save_numpy, load_numpy, save_json, save_csv, logger

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


def generate_embeddings(df: pd.DataFrame, model, tokenizer) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate BERT embeddings using both [CLS] and mean pooling strategies."""
    logger.info("Generating BERT embeddings...")
    
    model.eval()
    cls_embeddings = []
    mean_embeddings = []
    
    batch_size = 32
    total = len(df)
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_messages = df['message'].iloc[i:i+batch_size].tolist()
            
            # Tokenize
            encoded = tokenizer(
                batch_messages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            # Get BERT outputs
            outputs = model(**encoded)
            last_hidden_state = outputs.last_hidden_state
            
            # Extract [CLS] token embeddings (first token)
            cls_emb = last_hidden_state[:, 0, :].cpu().numpy()
            cls_embeddings.append(cls_emb)
            
            # Mean pooling (excluding padding tokens)
            attention_mask = encoded['attention_mask'].cpu().numpy()
            # Expand attention mask for broadcasting
            attention_mask_expanded = np.expand_dims(attention_mask, axis=-1)
            # Sum embeddings, weighted by attention mask
            sum_embeddings = np.sum(last_hidden_state.cpu().numpy() * attention_mask_expanded, axis=1)
            sum_mask = np.sum(attention_mask_expanded, axis=1)
            # Avoid division by zero
            mean_emb = sum_embeddings / np.maximum(sum_mask, 1e-9)
            mean_embeddings.append(mean_emb)
            
            if (i + batch_size) % 1000 == 0:
                logger.info(f"Processed {min(i + batch_size, total)}/{total} messages")
    
    # Concatenate batches
    cls_embeddings = np.vstack(cls_embeddings)
    mean_embeddings = np.vstack(mean_embeddings)
    
    # Concatenate [CLS] and mean pooling
    concatenated_embeddings = np.hstack([cls_embeddings, mean_embeddings])
    
    logger.info(f"Generated embeddings: CLS shape {cls_embeddings.shape}, "
                f"Mean shape {mean_embeddings.shape}, "
                f"Concatenated shape {concatenated_embeddings.shape}")
    
    return cls_embeddings, mean_embeddings, concatenated_embeddings


def tune_hyperparameter(X_train, y_train, X_val, y_val, C_values) -> float:
    """Tune C hyperparameter using validation set."""
    logger.info("Tuning hyperparameter C...")
    best_C = C_values[0]
    best_auc = 0
    
    for C in C_values:
        model = LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predict probabilities for validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        
        logger.info(f"C={C}: AUC-ROC = {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_C = C
    
    logger.info(f"Best C: {best_C} (AUC-ROC: {best_auc:.4f})")
    return best_C


def train_and_evaluate_probe(X_train, y_train, X_val, y_val, X_test, y_test, 
                             best_C: float, probe_name: str) -> Dict:
    """Train probe and evaluate on all splits."""
    logger.info(f"Training {probe_name} probe...")
    
    # Train on train set
    model = LogisticRegression(
        C=best_C,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate on all splits
    results = {}
    
    for split_name, X, y in [('train', X_train, y_train), 
                              ('validation', X_val, y_val),
                              ('test', X_test, y_test)]:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        cm = confusion_matrix(y, y_pred)
        
        results[split_name] = {
            'accuracy': float(accuracy),
            'auc_roc': float(auc),
            'confusion_matrix': cm.tolist(),
            'true_positives': int(cm[1, 1]),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0])
        }
        
        # Calculate precision and recall
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results[split_name]['precision'] = float(precision)
        results[split_name]['recall'] = float(recall)
        results[split_name]['f1_score'] = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        logger.info(f"{split_name.capitalize()} - Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f}")
    
    return model, results


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Truth', 'Lie'],
                yticklabels=['Truth', 'Lie'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def main():
    """Main function for probe training."""
    logger.info("Starting Step 3: Training the Initial Probe")
    
    # Load cleaned messages
    logger.info(f"Loading cleaned messages from {CLEANED_MESSAGES_CSV}")
    df = load_csv(CLEANED_MESSAGES_CSV)
    logger.info(f"Loaded {len(df)} messages")
    
    # Split data
    # If validation/test are empty, create splits from train data
    train_df = df[df['split'] == 'Train'].copy()
    val_df = df[df['split'] == 'Validation'].copy()
    test_df = df[df['split'] == 'Test'].copy()
    
    # If no validation/test splits, create them from train data (80/10/10 split)
    if len(val_df) == 0 and len(test_df) == 0:
        logger.info("No validation/test splits found, creating 80/10/10 split from train data")
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(
            train_df, test_size=0.2, random_state=42, stratify=train_df['label']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
        )
        # Reset indices for proper alignment
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
    
    logger.info(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Load BERT model and tokenizer
    logger.info(f"Loading BERT model: {BERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
    
    # Generate embeddings for all messages (concatenate splits in order)
    logger.info("Generating embeddings for all messages...")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    cls_emb, mean_emb, concat_emb = generate_embeddings(all_df, model, tokenizer)
    
    # Save embeddings
    logger.info("Saving embeddings...")
    save_numpy(cls_emb, MESSAGE_EMBEDDINGS_CLS)
    save_numpy(mean_emb, MESSAGE_EMBEDDINGS_MEAN)
    save_numpy(concat_emb, MESSAGE_EMBEDDINGS)
    
    # Split embeddings by position (since we concatenated in order)
    train_size = len(train_df)
    val_size = len(val_df)
    
    X_train_cls = cls_emb[:train_size]
    X_val_cls = cls_emb[train_size:train_size+val_size]
    X_test_cls = cls_emb[train_size+val_size:]
    
    X_train_mean = mean_emb[:train_size]
    X_val_mean = mean_emb[train_size:train_size+val_size]
    X_test_mean = mean_emb[train_size+val_size:]
    
    X_train_concat = concat_emb[:train_size]
    X_val_concat = concat_emb[train_size:train_size+val_size]
    X_test_concat = concat_emb[train_size+val_size:]
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Tune hyperparameter on concatenated embeddings
    logger.info("Tuning hyperparameter on concatenated embeddings...")
    best_C = tune_hyperparameter(X_train_concat, y_train, X_val_concat, y_val, C_VALUES)
    
    # Train concatenated probe
    logger.info("Training concatenated probe...")
    concat_model, concat_results = train_and_evaluate_probe(
        X_train_concat, y_train, X_val_concat, y_val, X_test_concat, y_test,
        best_C, "concatenated"
    )
    
    # Save concatenated probe
    joblib.dump(concat_model, PROBE_MESSAGE_ONLY)
    logger.info(f"Saved concatenated probe to {PROBE_MESSAGE_ONLY}")
    
    # Train [CLS]-only probe
    logger.info("Training [CLS]-only probe...")
    best_C_cls = tune_hyperparameter(X_train_cls, y_train, X_val_cls, y_val, C_VALUES)
    cls_model, cls_results = train_and_evaluate_probe(
        X_train_cls, y_train, X_val_cls, y_val, X_test_cls, y_test,
        best_C_cls, "[CLS]-only"
    )
    joblib.dump(cls_model, PROBE_CLS_ONLY)
    
    # Train mean-only probe
    logger.info("Training mean-only probe...")
    best_C_mean = tune_hyperparameter(X_train_mean, y_train, X_val_mean, y_val, C_VALUES)
    mean_model, mean_results = train_and_evaluate_probe(
        X_train_mean, y_train, X_val_mean, y_val, X_test_mean, y_test,
        best_C_mean, "mean-only"
    )
    joblib.dump(mean_model, PROBE_MEAN_ONLY)
    
    # Compare [CLS] vs mean pooling
    logger.info("Comparing [CLS] vs mean pooling...")
    comparison = {
        'cls_only': {
            'test_accuracy': cls_results['test']['accuracy'],
            'test_auc_roc': cls_results['test']['auc_roc'],
            'best_C': best_C_cls
        },
        'mean_only': {
            'test_accuracy': mean_results['test']['accuracy'],
            'test_auc_roc': mean_results['test']['auc_roc'],
            'best_C': best_C_mean
        },
        'concatenated': {
            'test_accuracy': concat_results['test']['accuracy'],
            'test_auc_roc': concat_results['test']['auc_roc'],
            'best_C': best_C
        }
    }
    
    save_json(comparison, CLS_VS_MEAN_COMPARISON_JSON)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    strategies = ['[CLS] only', 'Mean only', 'Concatenated']
    accuracies = [comparison['cls_only']['test_accuracy'],
                  comparison['mean_only']['test_accuracy'],
                  comparison['concatenated']['test_accuracy']]
    aucs = [comparison['cls_only']['test_auc_roc'],
            comparison['mean_only']['test_auc_roc'],
            comparison['concatenated']['test_auc_roc']]
    
    ax1.bar(strategies, accuracies)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_ylim([0, 1])
    
    ax2.bar(strategies, aucs)
    ax2.set_ylabel('AUC-ROC')
    ax2.set_title('Test AUC-ROC Comparison')
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(CLS_VS_MEAN_COMPARISON_PNG)
    plt.close()
    logger.info(f"Saved comparison plot to {CLS_VS_MEAN_COMPARISON_PNG}")
    
    # Plot confusion matrix for concatenated probe
    y_test_pred = concat_model.predict(X_test_concat)
    plot_confusion_matrix(
        y_test, y_test_pred,
        "Confusion Matrix - Message-Only Probe (Test Set)",
        CONFUSION_MATRIX_MESSAGE_ONLY
    )
    
    # Save metrics
    all_metrics = {
        'concatenated': concat_results,
        'cls_only': cls_results,
        'mean_only': mean_results,
        'hyperparameters': {
            'concatenated_C': best_C,
            'cls_only_C': best_C_cls,
            'mean_only_C': best_C_mean
        }
    }
    save_json(all_metrics, PROBE_METRICS_MESSAGE_ONLY)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_test_pred,
        'predicted_proba': concat_model.predict_proba(X_test_concat)[:, 1]
    })
    save_csv(predictions_df, RESULTS_DIR / "probe_predictions_message_only.csv")
    
    logger.info("\n" + "="*70)
    logger.info("Step 3 Summary")
    logger.info("="*70)
    logger.info(f"Concatenated probe - Test Accuracy: {concat_results['test']['accuracy']:.4f}, "
                f"AUC-ROC: {concat_results['test']['auc_roc']:.4f}")
    logger.info(f"[CLS]-only probe - Test Accuracy: {cls_results['test']['accuracy']:.4f}, "
                f"AUC-ROC: {cls_results['test']['auc_roc']:.4f}")
    logger.info(f"Mean-only probe - Test Accuracy: {mean_results['test']['accuracy']:.4f}, "
                f"AUC-ROC: {mean_results['test']['auc_roc']:.4f}")
    logger.info("="*70)
    logger.info("Step 3 completed successfully!")


if __name__ == "__main__":
    main()

