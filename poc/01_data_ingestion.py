"""Step 1: Data Ingestion & Sanitation

Load ConvoKit corpus, extract utterance-level data, load move data, and save cleaned data.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from convokit import Corpus, download

from config import (
    CLEANED_MESSAGES_CSV,
    MOVES_DATA_JSON,
    MOVES_DIR,
    DATA_DIR
)
from utils import (
    normalize_speaker_intention,
    save_json,
    save_csv,
    logger
)

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_convokit_corpus() -> Corpus:
    """Download and load ConvoKit corpus."""
    logger.info("Downloading and loading ConvoKit corpus...")
    corpus = Corpus(filename=download("diplomacy-corpus"))
    logger.info(f"Loaded corpus with {len(corpus.get_conversation_ids())} conversations")
    return corpus


def filter_by_split(corpus: Corpus, split_name: str) -> Corpus:
    """Filter conversations by acl2020_fold."""
    filtered_corpus = corpus.filter_conversations_by(
        lambda convo: convo.meta.get('acl2020_fold') == split_name
    )
    logger.info(f"Filtered to {split_name}: {len(filtered_corpus.get_conversation_ids())} conversations")
    return filtered_corpus


def get_receiver_from_conversation(conversation, speaker_id: str, corpus) -> Optional[str]:
    """Extract receiver from conversation participants."""
    # Get all speakers in the conversation by looking at utterances
    speakers = set()
    for utterance_id in conversation.get_utterance_ids():
        utterance = corpus.get_utterance(utterance_id)
        speakers.add(utterance.speaker.id)
    
    # The receiver is the other participant (not the speaker)
    for participant in speakers:
        if participant != speaker_id:
            return participant
    return None


def extract_utterance_data(utterance, conversation, corpus) -> Optional[Dict[str, Any]]:
    """Extract message and metadata from utterance."""
    try:
        # Get basic utterance data
        text = utterance.text
        speaker_id = utterance.speaker.id
        conversation_id = utterance.conversation_id
        
        # Get receiver
        receiver = get_receiver_from_conversation(conversation, speaker_id, corpus)
        
        # Get metadata
        meta = utterance.meta
        
        # Extract speaker_intention
        speaker_intention = meta.get('speaker_intention')
        if speaker_intention is None:
            return None  # Skip utterances without speaker_intention
        
        # Normalize label
        label = normalize_speaker_intention(speaker_intention)
        if label is None:
            return None
        
        # Extract other metadata
        receiver_perception = meta.get('receiver_perception')
        deception_quadrant = meta.get('deception_quadrant')
        relative_message_index = meta.get('relative_message_index')
        absolute_message_index = meta.get('absolute_message_index')
        year = meta.get('year')
        game_score = meta.get('game_score')
        game_score_delta = meta.get('game_score_delta')
        
        # Convert game_score_delta to int if it's a string
        if game_score_delta is not None:
            try:
                game_score_delta = int(game_score_delta)
            except (ValueError, TypeError):
                game_score_delta = None
        
        # Get game_id from conversation or speaker metadata
        # Try to extract from conversation metadata or speaker metadata
        game_id = None
        if conversation.meta:
            # Check if game_id is in conversation metadata
            game_id = conversation.meta.get('game_id')
        
        # If not found, try to extract from speaker metadata
        if game_id is None and utterance.speaker.meta:
            game_id = utterance.speaker.meta.get('game_id')
        
        return {
            'message': text,
            'sender': speaker_id,
            'receiver': receiver,
            'label': label,
            'conversation_id': conversation_id,
            'absolute_message_index': absolute_message_index,
            'relative_message_index': relative_message_index,
            'year': year,
            'game_score': game_score,
            'game_score_delta': game_score_delta,
            'deception_quadrant': deception_quadrant,
            'receiver_perception': receiver_perception,
            'game_id': game_id
        }
    except Exception as e:
        logger.warning(f"Error extracting data from utterance {utterance.id}: {e}")
        return None


def load_move_data(moves_dir: Path) -> Dict[str, Any]:
    """Load and parse move JSON files."""
    logger.info(f"Loading move data from {moves_dir}...")
    moves_data = {}
    
    if not moves_dir.exists():
        logger.warning(f"Moves directory {moves_dir} does not exist")
        return moves_data
    
    # Find all JSON files in moves directory
    move_files = list(moves_dir.glob("DiplomacyGame*.json"))
    logger.info(f"Found {len(move_files)} move files")
    
    for move_file in move_files:
        try:
            # Parse filename: DiplomacyGame{ID}_{YEAR}_{SEASON}.json
            filename = move_file.stem
            parts = filename.split('_')
            if len(parts) >= 3:
                game_id = int(parts[0].replace('DiplomacyGame', ''))
                year = parts[1]
                season = parts[2]
                
                # Load move data
                with open(move_file, 'r') as f:
                    move_content = json.load(f)
                
                # Create key for indexing
                key = (game_id, year, season)
                moves_data[key] = move_content
        except Exception as e:
            logger.warning(f"Error loading move file {move_file}: {e}")
    
    logger.info(f"Loaded moves for {len(moves_data)} game/year/season combinations")
    return moves_data


def create_move_index(moves_data: Dict[str, Any]) -> Dict[tuple, Dict]:
    """Create (game_id, year, season, player) -> moves mapping."""
    move_index = {}
    
    for (game_id, year, season), move_content in moves_data.items():
        # Extract orders from move content
        orders = move_content.get('orders', {})
        
        # Index by player
        for player, player_orders in orders.items():
            key = (game_id, year, season, player.lower())
            move_index[key] = player_orders
    
    logger.info(f"Created move index with {len(move_index)} entries")
    return move_index


def main():
    """Main function for data ingestion."""
    logger.info("Starting Step 1: Data Ingestion & Sanitation")
    
    # Load ConvoKit corpus
    corpus = load_convokit_corpus()
    
    # Process each split
    all_messages = []
    splits = ['Train', 'Validation', 'Test']
    
    for split_name in splits:
        logger.info(f"Processing {split_name} split...")
        split_corpus = filter_by_split(corpus, split_name)
        
        # Extract utterance data
        for conversation_id in split_corpus.get_conversation_ids():
            conversation = split_corpus.get_conversation(conversation_id)
            
            for utterance_id in conversation.get_utterance_ids():
                utterance = split_corpus.get_utterance(utterance_id)
                utterance_data = extract_utterance_data(utterance, conversation, split_corpus)
                
                if utterance_data is not None:
                    utterance_data['split'] = split_name
                    all_messages.append(utterance_data)
    
    logger.info(f"Extracted {len(all_messages)} messages total")
    
    # Create DataFrame
    df = pd.DataFrame(all_messages)
    
    # Reorder columns
    column_order = [
        'message', 'sender', 'receiver', 'label', 'conversation_id',
        'absolute_message_index', 'relative_message_index', 'year',
        'game_score', 'game_score_delta', 'deception_quadrant', 'split',
        'receiver_perception', 'game_id'
    ]
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # Save cleaned messages
    logger.info(f"Saving cleaned messages to {CLEANED_MESSAGES_CSV}")
    save_csv(df, CLEANED_MESSAGES_CSV)
    
    # Load move data
    moves_data = load_move_data(MOVES_DIR)
    move_index = create_move_index(moves_data)
    
    # Save move data
    logger.info(f"Saving move data to {MOVES_DATA_JSON}")
    # Convert tuple keys to strings for JSON serialization
    moves_data_str = {}
    for k, v in moves_data.items():
        key_str = str(k)  # Convert tuple to string
        moves_data_str[key_str] = v
    
    move_index_str = {}
    for k, v in move_index.items():
        # Convert tuple key to string representation
        key_str = str(k)  # This will be like "(1, '1901', 'spring', 'italy')"
        move_index_str[key_str] = v
    
    save_json({
        'moves_data': moves_data_str,
        'move_index': move_index_str
    }, MOVES_DATA_JSON)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Data Ingestion Summary")
    logger.info("="*50)
    logger.info(f"Total messages: {len(df)}")
    logger.info(f"Train: {len(df[df['split'] == 'Train'])}")
    logger.info(f"Validation: {len(df[df['split'] == 'Validation'])}")
    logger.info(f"Test: {len(df[df['split'] == 'Test'])}")
    logger.info(f"Lies (label=1): {len(df[df['label'] == 1])}")
    logger.info(f"Truths (label=0): {len(df[df['label'] == 0])}")
    logger.info(f"Move data entries: {len(moves_data)}")
    logger.info("="*50)
    logger.info("Step 1 completed successfully!")


if __name__ == "__main__":
    main()

