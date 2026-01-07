"""Step 2: Baseline Behavioral Metrics

Calculate deception rates, per-game statistics, temporal patterns, and deception quadrant distribution.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict
import logging

from config import (
    CLEANED_MESSAGES_CSV,
    BASELINE_METRICS_JSON,
    BASELINE_METRICS_TXT,
    RESULTS_DIR
)
from utils import load_csv, save_json, logger

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_deception_rate(df: pd.DataFrame) -> Dict:
    """Calculate overall deception rate."""
    total = len(df)
    lies = len(df[df['label'] == 1])
    truths = len(df[df['label'] == 0])
    
    deception_rate = (lies / total * 100) if total > 0 else 0
    
    return {
        'total_messages': total,
        'lies': lies,
        'truths': truths,
        'deception_rate_percent': deception_rate,
        'truth_rate_percent': 100 - deception_rate
    }


def calculate_per_game_rates(df: pd.DataFrame) -> Dict:
    """Calculate per-game deception rates."""
    if 'game_id' not in df.columns or df['game_id'].isna().all():
        logger.warning("game_id not available, skipping per-game analysis")
        return {}
    
    game_stats = []
    for game_id in sorted(df['game_id'].dropna().unique()):
        game_df = df[df['game_id'] == game_id]
        total = len(game_df)
        lies = len(game_df[game_df['label'] == 1])
        deception_rate = (lies / total * 100) if total > 0 else 0
        
        game_stats.append({
            'game_id': int(game_id),
            'total_messages': total,
            'lies': lies,
            'truths': total - lies,
            'deception_rate_percent': deception_rate
        })
    
    return {'per_game_stats': game_stats}


def calculate_sender_receiver_stats(df: pd.DataFrame) -> Dict:
    """Analyze deception by sender/receiver pairs."""
    if 'sender' not in df.columns or 'receiver' not in df.columns:
        logger.warning("sender/receiver not available, skipping pair analysis")
        return {}
    
    # Group by sender-receiver pairs
    pair_stats = []
    for (sender, receiver), group_df in df.groupby(['sender', 'receiver']):
        total = len(group_df)
        lies = len(group_df[group_df['label'] == 1])
        deception_rate = (lies / total * 100) if total > 0 else 0
        
        pair_stats.append({
            'sender': sender,
            'receiver': receiver,
            'total_messages': total,
            'lies': lies,
            'truths': total - lies,
            'deception_rate_percent': deception_rate
        })
    
    # Sort by total messages descending
    pair_stats.sort(key=lambda x: x['total_messages'], reverse=True)
    
    return {'sender_receiver_pairs': pair_stats[:20]}  # Top 20 pairs


def calculate_temporal_patterns(df: pd.DataFrame) -> Dict:
    """Calculate deception rate by year."""
    if 'year' not in df.columns or df['year'].isna().all():
        logger.warning("year not available, skipping temporal analysis")
        return {}
    
    year_stats = []
    for year in sorted(df['year'].dropna().unique()):
        year_df = df[df['year'] == year]
        total = len(year_df)
        lies = len(year_df[year_df['label'] == 1])
        deception_rate = (lies / total * 100) if total > 0 else 0
        
        year_stats.append({
            'year': str(year),
            'total_messages': total,
            'lies': lies,
            'truths': total - lies,
            'deception_rate_percent': deception_rate
        })
    
    return {'temporal_patterns': year_stats}


def calculate_deception_quadrant_distribution(df: pd.DataFrame) -> Dict:
    """Analyze distribution across deception quadrants."""
    if 'deception_quadrant' not in df.columns:
        logger.warning("deception_quadrant not available, skipping quadrant analysis")
        return {}
    
    quadrant_counts = df['deception_quadrant'].value_counts().to_dict()
    quadrant_stats = {}
    
    for quadrant, count in quadrant_counts.items():
        quadrant_df = df[df['deception_quadrant'] == quadrant]
        total = len(quadrant_df)
        lies = len(quadrant_df[quadrant_df['label'] == 1])
        deception_rate = (lies / total * 100) if total > 0 else 0
        
        quadrant_stats[quadrant] = {
            'count': count,
            'lies': lies,
            'truths': total - lies,
            'deception_rate_percent': deception_rate
        }
    
    return {'deception_quadrant_distribution': quadrant_stats}


def calculate_class_distribution(df: pd.DataFrame) -> Dict:
    """Calculate class distribution statistics."""
    total = len(df)
    lies = len(df[df['label'] == 1])
    truths = len(df[df['label'] == 0])
    
    return {
        'class_distribution': {
            'total': total,
            'lies': lies,
            'truths': truths,
            'lie_percentage': (lies / total * 100) if total > 0 else 0,
            'truth_percentage': (truths / total * 100) if total > 0 else 0,
            'imbalance_ratio': (truths / lies) if lies > 0 else None
        }
    }


def generate_text_report(metrics: Dict) -> str:
    """Generate human-readable text report."""
    report = []
    report.append("="*70)
    report.append("BASELINE BEHAVIORAL METRICS")
    report.append("="*70)
    report.append("")
    
    # Overall deception rate
    overall = metrics.get('overall_deception_rate', {})
    report.append("OVERALL DECEPTION RATE")
    report.append("-"*70)
    report.append(f"Total messages: {overall.get('total_messages', 0):,}")
    report.append(f"Lies: {overall.get('lies', 0):,} ({overall.get('deception_rate_percent', 0):.2f}%)")
    report.append(f"Truths: {overall.get('truths', 0):,} ({overall.get('truth_rate_percent', 0):.2f}%)")
    report.append("")
    
    # Class distribution
    class_dist = metrics.get('class_distribution', {}).get('class_distribution', {})
    if class_dist:
        report.append("CLASS DISTRIBUTION")
        report.append("-"*70)
        report.append(f"Imbalance ratio (truth:lie): {class_dist.get('imbalance_ratio', 'N/A')}")
        report.append("")
    
    # Per-game stats
    per_game = metrics.get('per_game_rates', {}).get('per_game_stats', [])
    if per_game:
        report.append("PER-GAME DECEPTION RATES")
        report.append("-"*70)
        for game_stat in per_game[:10]:  # Top 10 games
            report.append(
                f"Game {game_stat['game_id']}: "
                f"{game_stat['deception_rate_percent']:.2f}% "
                f"({game_stat['lies']}/{game_stat['total_messages']} lies)"
            )
        report.append("")
    
    # Temporal patterns
    temporal = metrics.get('temporal_patterns', {}).get('temporal_patterns', [])
    if temporal:
        report.append("TEMPORAL PATTERNS (BY YEAR)")
        report.append("-"*70)
        for year_stat in temporal:
            report.append(
                f"Year {year_stat['year']}: "
                f"{year_stat['deception_rate_percent']:.2f}% "
                f"({year_stat['lies']}/{year_stat['total_messages']} lies)"
            )
        report.append("")
    
    # Deception quadrant
    quadrant = metrics.get('deception_quadrant_distribution', {}).get('deception_quadrant_distribution', {})
    if quadrant:
        report.append("DECEPTION QUADRANT DISTRIBUTION")
        report.append("-"*70)
        for quad, stats in quadrant.items():
            report.append(
                f"{quad}: {stats['count']} messages "
                f"({stats['deception_rate_percent']:.2f}% lies)"
            )
        report.append("")
    
    # Sender-receiver pairs
    pairs = metrics.get('sender_receiver_stats', {}).get('sender_receiver_pairs', [])
    if pairs:
        report.append("TOP SENDER-RECEIVER PAIRS (by message count)")
        report.append("-"*70)
        for pair in pairs[:10]:  # Top 10 pairs
            report.append(
                f"{pair['sender']} -> {pair['receiver']}: "
                f"{pair['total_messages']} messages, "
                f"{pair['deception_rate_percent']:.2f}% lies"
            )
        report.append("")
    
    report.append("="*70)
    return "\n".join(report)


def main():
    """Main function for baseline metrics calculation."""
    logger.info("Starting Step 2: Baseline Behavioral Metrics")
    
    # Load cleaned messages
    logger.info(f"Loading cleaned messages from {CLEANED_MESSAGES_CSV}")
    df = load_csv(CLEANED_MESSAGES_CSV)
    logger.info(f"Loaded {len(df)} messages")
    
    # Calculate metrics
    metrics = {}
    
    logger.info("Calculating overall deception rate...")
    metrics['overall_deception_rate'] = calculate_deception_rate(df)
    
    logger.info("Calculating per-game rates...")
    metrics['per_game_rates'] = calculate_per_game_rates(df)
    
    logger.info("Calculating sender-receiver statistics...")
    metrics['sender_receiver_stats'] = calculate_sender_receiver_stats(df)
    
    logger.info("Calculating temporal patterns...")
    metrics['temporal_patterns'] = calculate_temporal_patterns(df)
    
    logger.info("Calculating deception quadrant distribution...")
    metrics['deception_quadrant_distribution'] = calculate_deception_quadrant_distribution(df)
    
    logger.info("Calculating class distribution...")
    metrics['class_distribution'] = calculate_class_distribution(df)
    
    # Save JSON metrics
    logger.info(f"Saving metrics to {BASELINE_METRICS_JSON}")
    save_json(metrics, BASELINE_METRICS_JSON)
    
    # Generate and save text report
    logger.info(f"Generating text report...")
    text_report = generate_text_report(metrics)
    logger.info(f"Saving text report to {BASELINE_METRICS_TXT}")
    with open(BASELINE_METRICS_TXT, 'w') as f:
        f.write(text_report)
    
    # Print summary
    print("\n" + text_report)
    logger.info("Step 2 completed successfully!")


if __name__ == "__main__":
    main()

