# Diplomacy Dataset Documentation

## Overview

Each file is a JSONL (JSON Lines) file containing an entire game dialog/conversation per line. Each line represents a conversation between two players from a Diplomacy game, with all messages, labels, and metadata for that conversation.

An alternate representation of the data, by message not by conversation, is available as part of ConvoKit: https://convokit.cornell.edu/documentation/diplomacy.html

## Data Structure

### File Format

- **Format**: JSONL (one JSON object per line)
- **Encoding**: UTF-8
- **Files**: `train.jsonl`, `validation.jsonl`, `test.jsonl`

### JSON Object Structure

Each line in the JSONL file contains a single JSON object representing one conversation between two players. All fields are arrays of equal length, where each index corresponds to a single message in the conversation.

```json
{
  "messages": ["message1", "message2", ...],
  "sender_labels": [true, false, ...],
  "receiver_labels": [true, false, "NOANNOTATION", ...],
  "speakers": ["italy", "germany", ...],
  "receivers": ["germany", "italy", ...],
  "game_score": ["3", "3", ...],
  "game_score_delta": ["0", "1", ...],
  "absolute_message_index": [74, 76, ...],
  "relative_message_index": [0, 1, ...],
  "seasons": ["Spring", "Spring", ...],
  "years": ["1901", "1901", ...],
  "players": ["italy", "germany"],
  "game_id": 1
}
```

### Field Descriptions

#### Core Message Fields

- **`messages`** (array of strings)
  - The raw message text sent between players
  - Length: 1 word to 294 words (mean: 20.79 words, median: 14 words)
  - Contains natural language diplomatic communication

- **`speakers`** (array of strings)
  - The sender of each message
  - Values: `"italy"`, `"germany"`, `"austria"`, `"russia"`, `"england"`, `"turkey"`, `"france"` (lowercase)
  - Length matches `messages` array

- **`receivers`** (array of strings)
  - The receiver of each message
  - Values: same as speakers (lowercase country names)
  - Length matches `messages` array

#### Label Fields

- **`sender_labels`** (array of boolean/string)
  - Ground truth: whether the sender intentionally sent a deceptive message
  - Values: `true` (truthful) or `false` (deceptive)
  - Can be boolean or string format (`"true"`/`"false"`)
  - Used for **ACTUAL_LIE** detection task
  - Length matches `messages` array

- **`receiver_labels`** (array of string)
  - Perception: whether the receiver perceived the message as deceptive
  - Values: `"true"` (perceived truthful), `"false"` (perceived deceptive), or `"NOANNOTATION"` (missing annotation)
  - Used for **SUSPECTED_LIE** detection task
  - ~8.4% of labels are `"NOANNOTATION"` (missing annotations)
  - Length matches `messages` array

#### Game State Fields

- **`game_score`** (array of strings)
  - Current supply centers (game score) of the sender at the time of each message
  - Values: `"0"` to `"18"` (string format)
  - Represents player power/position in the game
  - Length matches `messages` array

- **`game_score_delta`** (array of strings)
  - Power imbalance: sender's score minus receiver's score
  - Values: `"-18"` to `"18"` (string format)
  - Negative: receiver is more powerful
  - Positive: sender is more powerful
  - Mean: ~0.07 (roughly balanced overall)
  - Used as a feature in models with power information
  - Length matches `messages` array

#### Index Fields

- **`absolute_message_index`** (array of integers)
  - The index of the message in the entire game, across all conversations
  - Unique across all conversations in the same game
  - Used to understand message ordering across different player pairs
  - Length matches `messages` array

- **`relative_message_index`** (array of integers)
  - The index of the message within the current conversation
  - Starts at 0 for the first message in each conversation
  - Used to understand message ordering within a single conversation
  - Length matches `messages` array

#### Temporal Fields

- **`seasons`** (array of strings)
  - The season in Diplomacy when the message was sent
  - Values: `"Spring"`, `"Fall"`, `"Winter"`
  - Length matches `messages` array

- **`years`** (array of strings)
  - The year in Diplomacy when the message was sent
  - Values: `"1901"` through `"1918"` (string format)
  - Length matches `messages` array

#### Conversation Metadata

- **`players`** (array of strings, length 2)
  - The two players involved in this conversation
  - Values: lowercase country names (e.g., `["italy", "germany"]`)
  - Same two players throughout the conversation
  - Not an array per message, but a single array for the entire conversation

- **`game_id`** (integer)
  - Which of the 12 games this conversation comes from
  - Values: 1-12 (note: not all game IDs are present in train set)
  - Same for all conversations from the same game

### Array Relationships

**Critical**: All message-level fields (`messages`, `sender_labels`, `receiver_labels`, `speakers`, `receivers`, `game_score`, `game_score_delta`, `absolute_message_index`, `relative_message_index`, `seasons`, `years`) must have the same length, as each index corresponds to a single message.

Example: `messages[0]`, `sender_labels[0]`, `speakers[0]`, `receivers[0]`, etc. all refer to the same message.

## Dataset Statistics (train.jsonl)

### Basic Statistics

- **Total conversations**: 189
- **Total messages**: 13,132
- **Games represented**: 9 unique games (IDs: 1, 2, 3, 5, 6, 7, 8, 9, 10)
- **Empty conversations**: 5 (conversations with 0 messages)

### Conversation Characteristics

- **Messages per conversation**: 
  - Minimum: 0
  - Maximum: 675
  - Mean: 69.48 messages
- **Conversations per game**: 21 (all player pairs: 7 choose 2 = 21)

### Label Distribution

#### Sender Labels (Actual Lie Detection)
- **True (truthful)**: 12,541 messages (95.5%)
- **False (deceptive)**: 591 messages (4.5%)
- **Class imbalance ratio**: ~21:1 (truthful:deceptive)

#### Receiver Labels (Suspected Lie Detection)
- **True (perceived truthful)**: 11,459 messages (87.3%)
- **False (perceived deceptive)**: 566 messages (4.3%)
- **NOANNOTATION (missing)**: 1,107 messages (8.4%)

#### Label Agreement
- **Agreement**: 11,044 cases (91.8% of annotated messages)
- **Disagreement**: 981 cases (8.2% of annotated messages)
- Agreement calculated only for messages where receiver annotation exists (excluding NOANNOTATION)

### Message Characteristics

- **Word count**:
  - Minimum: 1 word
  - Maximum: 294 words
  - Mean: 20.79 words
  - Median: 14 words

### Power Dynamics

- **Score delta range**: -14 to +14 supply centers
- **Mean score delta**: 0.07 (roughly balanced overall)
- **Distribution**: Relatively balanced, with slight variations by game phase

### Player Activity

**Most active speakers** (by message count):
1. Germany: 2,687 messages
2. Italy: 2,493 messages
3. England: 2,304 messages
4. Austria: 1,827 messages
5. France: 1,279 messages
6. Russia: 1,317 messages
7. Turkey: 1,225 messages

**Most active receivers**: Similar distribution, with Germany receiving the most (2,851 messages)

### Temporal Coverage

- **Years**: 1901-1910 (10 years of game time)
- **Seasons**: Spring, Fall, Winter

### Per-Game Statistics

| Game ID | Conversations | Messages | Lies (sender) | Lie Rate |
|---------|--------------|----------|---------------|----------|
| 1       | 21           | 2,618    | 219           | 8.4%     |
| 2       | 21           | 3,302    | 78            | 2.4%     |
| 3       | 21           | 1,914    | 45            | 2.4%     |
| 5       | 21           | 427      | 41            | 9.6%     |
| 6       | 21           | 521      | 50            | 9.6%     |
| 7       | 21           | 1,003    | 45            | 4.5%     |
| 8       | 21           | 905      | 27            | 3.0%     |
| 9       | 21           | 1,611    | 56            | 3.5%     |
| 10      | 21           | 831      | 30            | 3.6%     |

**Note**: Lie rates vary significantly across games, ranging from 2.4% to 9.6%, indicating different game dynamics and player behaviors.

## Key Insights

1. **Strong Class Imbalance**: Deceptive messages are rare (~4.5%), requiring careful handling in model training (e.g., weighted loss functions, class balancing techniques).

2. **High Label Agreement**: Senders and receivers agree 91.8% of the time when annotations exist, suggesting that most deception is either transparent or well-detected.

3. **Variable Game Dynamics**: Lie rates vary significantly across games (2.4% to 9.6%), indicating that game context and player strategies significantly influence deception frequency.

4. **Balanced Power Distribution**: Mean score_delta near 0 suggests conversations occur across various power relationships, providing diverse training scenarios.

5. **Rich Contextual Information**: Conversations average 69 messages, providing substantial context for hierarchical models that capture conversation dynamics.

6. **Missing Annotations**: 8.4% of receiver labels are missing, which must be handled appropriately during model training (typically by filtering out these examples for suspected lie detection).

7. **Data Consistency**: All conversations have consistent array lengths, ensuring data integrity.

## Usage Notes

- When processing the data, ensure all message-level arrays are aligned by index
- Filter out messages with `"NOANNOTATION"` when training suspected lie detection models
- Consider class imbalance when training models (lies are rare)
- Use `absolute_message_index` to understand cross-conversation message ordering
- Use `relative_message_index` for within-conversation message ordering
- The `players` field identifies the two participants in each conversation

*UPDATE*: We additionally have all game data (moves) available in the Moves folder.
