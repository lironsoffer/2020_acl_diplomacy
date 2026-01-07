"""Step 4: Training the Enriched Probe (With Move Context)

Construct enriched prompts, generate embeddings, train probe, and compare with message-only probe.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Tuple

import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    CLEANED_MESSAGES_CSV,
    MOVES_DATA_JSON,
    ENRICHED_MESSAGE_EMBEDDINGS,
    PROBE_ENRICHED,
    PROBE_METRICS_ENRICHED,
    PROBE_COMPARISON_JSON,
    PROBE_COMPARISON_PNG,
    CONFUSION_MATRIX_ENRICHED,
    PROBE_METRICS_MESSAGE_ONLY,
    RESULTS_DIR,
    BERT_MODEL_NAME,
    ENRICHED_PROMPT_TEMPLATE,
    C_VALUES
)
from utils import load_csv, load_json, save_numpy, load_numpy, save_json, save_csv, logger

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


def load_move_data(moves_file: Path) -> Tuple[Dict, Dict]:
    """Load processed move data from Step 1."""
    logger.info(f"Loading move data from {moves_file}")
    data = load_json(moves_file)
    
    moves_data = data.get('moves_data', {})
    move_index = data.get('move_index', {})
    
    # Convert string keys back to tuples for move_index
    move_index_parsed = {}
    for key_str, value in move_index.items():
        # Parse key like "(1, '1901', 'spring', 'italy')"
        try:
            key = eval(key_str)  # Convert string representation to tuple
            move_index_parsed[key] = value
        except:
            logger.warning(f"Could not parse move index key: {key_str}")
    
    logger.info(f"Loaded {len(moves_data)} move files and {len(move_index_parsed)} move index entries")
    return moves_data, move_index_parsed


def find_moves_for_message(message_row: pd.Series, move_index: Dict) -> Optional[Dict]:
    """Find corresponding moves for a message."""
    game_id = message_row.get('game_id')
    year = message_row.get('year')
    sender = message_row.get('sender')
    
    # Extract game_id from sender if not available (format: "italy-Game1")
    if pd.isna(game_id) and sender:
        if '-Game' in str(sender):
            try:
                game_id = int(str(sender).split('-Game')[1])
            except:
                pass
    
    # Extract country name from sender (format: "italy-Game1" -> "italy")
    if sender:
        country = str(sender).split('-')[0].lower()
    else:
        country = None
    
    if pd.isna(game_id) or pd.isna(year) or not country:
        return None
    
    # Try to find moves for this game/year/season/sender
    # We need to try different seasons since we might not have season info
    seasons = ['spring', 'fall', 'winter']
    
    for season in seasons:
        key = (int(game_id), str(year), season, country)
        if key in move_index:
            return move_index[key]
    
    return None


def format_move_summary(moves: Dict) -> str:
    """Format moves into readable summary for prompt."""
    if not moves:
        return "No moves available"
    
    move_descriptions = []
    for unit, move_info in moves.items():
        move_type = move_info.get('type', 'UNKNOWN')
        result = move_info.get('result', 'UNKNOWN')
        
        if move_type == 'MOVE':
            to_location = move_info.get('to', 'UNKNOWN')
            move_descriptions.append(f"Moved {unit} to {to_location} ({result})")
        elif move_type == 'HOLD':
            move_descriptions.append(f"Held {unit} ({result})")
        elif move_type == 'SUPPORT':
            supported_unit = move_info.get('supported_unit', 'UNKNOWN')
            move_descriptions.append(f"Supported {supported_unit} ({result})")
        else:
            move_descriptions.append(f"{unit}: {move_type} ({result})")
    
    return "; ".join(move_descriptions) if move_descriptions else "No moves available"


def construct_enriched_prompt(message_row: pd.Series, moves: Optional[Dict], 
                             game_state: Dict) -> str:
    """Create prompt combining message + moves + context."""
    message_text = message_row.get('message', '')
    year = message_row.get('year', 'Unknown')
    game_score = message_row.get('game_score', 'Unknown')
    game_score_delta = message_row.get('game_score_delta', 'Unknown')
    
    # Format move summary
    if moves:
        move_summary = format_move_summary(moves)
    else:
        move_summary = "No moves available for this turn"
    
    # Infer season (we'll use a placeholder since we don't have exact season info)
    season = "Unknown"  # Could be improved by parsing from absolute_message_index or other metadata
    
    # Construct prompt
    prompt = ENRICHED_PROMPT_TEMPLATE.format(
        message_text=message_text,
        move_summary=move_summary,
        game_score=game_score,
        game_score_delta=game_score_delta,
        year=year,
        season=season
    )
    
    return prompt


def generate_enriched_embeddings(df: pd.DataFrame, move_index: Dict, 
                                 model, tokenizer) -> np.ndarray:
    """Generate embeddings for enriched prompts."""
    logger.info("Generating enriched embeddings...")
    
    model.eval()
    cls_embeddings = []
    mean_embeddings = []
    
    batch_size = 32
    total = len(df)
    missing_moves_count = 0
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_prompts = []
            batch_indices = list(range(i, min(i + batch_size, total)))
            
            for idx in batch_indices:
                message_row = df.iloc[idx]
                moves = find_moves_for_message(message_row, move_index)
                
                if moves is None:
                    missing_moves_count += 1
                
                game_state = {
                    'game_score': message_row.get('game_score'),
                    'game_score_delta': message_row.get('game_score_delta')
                }
                
                prompt = construct_enriched_prompt(message_row, moves, game_state)
                batch_prompts.append(prompt)
            
            # Tokenize
            encoded = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            # Get BERT outputs
            outputs = model(**encoded)
            last_hidden_state = outputs.last_hidden_state
            
            # Extract [CLS] token embeddings
            cls_emb = last_hidden_state[:, 0, :].cpu().numpy()
            cls_embeddings.append(cls_emb)
            
            # Mean pooling
            attention_mask = encoded['attention_mask'].cpu().numpy()
            attention_mask_expanded = np.expand_dims(attention_mask, axis=-1)
            sum_embeddings = np.sum(last_hidden_state.cpu().numpy() * attention_mask_expanded, axis=1)
            sum_mask = np.sum(attention_mask_expanded, axis=1)
            mean_emb = sum_embeddings / np.maximum(sum_mask, 1e-9)
            mean_embeddings.append(mean_emb)
            
            if (i + batch_size) % 1000 == 0:
                logger.info(f"Processed {min(i + batch_size, total)}/{total} messages")
    
    logger.info(f"Missing moves for {missing_moves_count}/{total} messages")
    
    # Concatenate batches
    cls_embeddings = np.vstack(cls_embeddings)
    mean_embeddings = np.vstack(mean_embeddings)
    
    # Concatenate [CLS] and mean pooling
    concatenated_embeddings = np.hstack([cls_embeddings, mean_embeddings])
    
    logger.info(f"Generated enriched embeddings: shape {concatenated_embeddings.shape}")
    
    return concatenated_embeddings


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
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        
        logger.info(f"C={C}: AUC-ROC = {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_C = C
    
    logger.info(f"Best C: {best_C} (AUC-ROC: {best_auc:.4f})")
    return best_C


def train_and_evaluate_probe(X_train, y_train, X_val, y_val, X_test, y_test, 
                             best_C: float) -> Tuple[object, Dict]:
    """Train probe and evaluate on all splits."""
    logger.info("Training enriched probe...")
    
    model = LogisticRegression(
        C=best_C,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    results = {}
    
    for split_name, X, y in [('train', X_train, y_train), 
                              ('validation', X_val, y_val),
                              ('test', X_test, y_test)]:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        cm = confusion_matrix(y, y_pred)
        
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results[split_name] = {
            'accuracy': float(accuracy),
            'auc_roc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0,
            'confusion_matrix': cm.tolist()
        }
        
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


def compare_probes(message_only_metrics: Dict, enriched_metrics: Dict) -> Dict:
    """Create side-by-side comparison of both probes.
    
    Args:
        message_only_metrics: Dict with keys like 'accuracy', 'auc_roc', etc. (test set results)
        enriched_metrics: Dict with 'test' key containing test set results
    """
    # Handle both structures - message_only_metrics might be the test dict directly
    if 'test' in message_only_metrics:
        msg_test = message_only_metrics['test']
    else:
        msg_test = message_only_metrics
    
    enriched_test = enriched_metrics['test']
    
    comparison = {
        'test_set': {
            'message_only': {
                'accuracy': msg_test['accuracy'],
                'auc_roc': msg_test['auc_roc'],
                'precision': msg_test['precision'],
                'recall': msg_test['recall'],
                'f1_score': msg_test['f1_score']
            },
            'enriched': {
                'accuracy': enriched_test['accuracy'],
                'auc_roc': enriched_test['auc_roc'],
                'precision': enriched_test['precision'],
                'recall': enriched_test['recall'],
                'f1_score': enriched_test['f1_score']
            }
        },
        'improvement': {
            'accuracy': enriched_test['accuracy'] - msg_test['accuracy'],
            'auc_roc': enriched_test['auc_roc'] - msg_test['auc_roc'],
            'precision': enriched_test['precision'] - msg_test['precision'],
            'recall': enriched_test['recall'] - msg_test['recall'],
            'f1_score': enriched_test['f1_score'] - msg_test['f1_score']
        }
    }
    
    return comparison


def plot_probe_comparison(comparison: Dict, save_path: Path):
    """Create side-by-side comparison plot."""
    metrics = ['accuracy', 'auc_roc', 'precision', 'recall', 'f1_score']
    message_only_values = [comparison['test_set']['message_only'][m] for m in metrics]
    enriched_values = [comparison['test_set']['enriched'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, message_only_values, width, label='Message-Only', alpha=0.8)
    bars2 = ax.bar(x + width/2, enriched_values, width, label='Enriched', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Probe Comparison: Message-Only vs Enriched (Test Set)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    """Main function for enriched probe training."""
    logger.info("Starting Step 4: Training the Enriched Probe")
    
    # Load data
    logger.info("Loading data...")
    df = load_csv(CLEANED_MESSAGES_CSV)
    moves_data, move_index = load_move_data(MOVES_DATA_JSON)
    
    # Split data
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
    
    # Load BERT model
    logger.info(f"Loading BERT model: {BERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
    
    # Generate enriched embeddings for all messages (concatenate splits in order)
    logger.info("Generating enriched embeddings...")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    enriched_embeddings = generate_enriched_embeddings(all_df, move_index, model, tokenizer)
    
    # Save enriched embeddings
    save_numpy(enriched_embeddings, ENRICHED_MESSAGE_EMBEDDINGS)
    
    # Split embeddings by position (since we concatenated in order)
    train_size = len(train_df)
    val_size = len(val_df)
    
    X_train = enriched_embeddings[:train_size]
    X_val = enriched_embeddings[train_size:train_size+val_size]
    X_test = enriched_embeddings[train_size+val_size:]
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Tune hyperparameter
    best_C = tune_hyperparameter(X_train, y_train, X_val, y_val, C_VALUES)
    
    # Train and evaluate enriched probe
    enriched_model, enriched_results = train_and_evaluate_probe(
        X_train, y_train, X_val, y_val, X_test, y_test, best_C
    )
    
    # Save enriched probe
    joblib.dump(enriched_model, PROBE_ENRICHED)
    save_json(enriched_results, PROBE_METRICS_ENRICHED)
    
    # Plot confusion matrix
    y_test_pred = enriched_model.predict(X_test)
    plot_confusion_matrix(
        y_test, y_test_pred,
        "Confusion Matrix - Enriched Probe (Test Set)",
        CONFUSION_MATRIX_ENRICHED
    )
    
    # Load message-only metrics for comparison
    message_only_metrics = load_json(PROBE_METRICS_MESSAGE_ONLY)
    message_only_test = message_only_metrics['concatenated']['test']
    
    # Compare probes
    comparison = compare_probes(message_only_test, enriched_results)
    save_json(comparison, PROBE_COMPARISON_JSON)
    plot_probe_comparison(comparison, PROBE_COMPARISON_PNG)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_test_pred,
        'predicted_proba': enriched_model.predict_proba(X_test)[:, 1]
    })
    save_csv(predictions_df, RESULTS_DIR / "probe_predictions_enriched.csv")
    
    logger.info("\n" + "="*70)
    logger.info("Step 4 Summary")
    logger.info("="*70)
    logger.info(f"Enriched probe - Test Accuracy: {enriched_results['test']['accuracy']:.4f}, "
                f"AUC-ROC: {enriched_results['test']['auc_roc']:.4f}")
    logger.info(f"Message-only probe - Test Accuracy: {message_only_test['accuracy']:.4f}, "
                f"AUC-ROC: {message_only_test['auc_roc']:.4f}")
    logger.info(f"Improvement - Accuracy: {comparison['improvement']['accuracy']:.4f}, "
                f"AUC-ROC: {comparison['improvement']['auc_roc']:.4f}")
    logger.info("="*70)
    logger.info("Step 4 completed successfully!")


if __name__ == "__main__":
    main()

