# Project Summary: "It Takes Two to Lie: One to Lie, and One to Listen"

## Overview

This repository contains the code and data for the ACL 2020 paper **"It Takes Two to Lie: One to Lie, and One to Listen"** by Peskov et al. The project focuses on detecting deception in strategic communication, specifically using the game of Diplomacy as a naturalistic setting where both actual lies (sender's intent) and suspected lies (receiver's perception) can be studied.

## Research Contribution

The key insight of this work is that deception detection involves two distinct perspectives:
- **Actual Lie (Sender Perspective)**: Whether the sender intentionally sent a deceptive message
- **Suspected Lie (Receiver Perspective)**: Whether the receiver perceived the message as deceptive

The paper demonstrates that these two tasks are fundamentally different and require different modeling approaches, with the receiver's perception being more challenging to predict.

## Dataset

The dataset consists of **12 complete Diplomacy games** with annotated messages:

- **Format**: JSONL files (train.jsonl, validation.jsonl, test.jsonl)
- **Content**: Each line contains an entire game dialog/conversation
- **Annotations**:
  - `sender_labels`: Ground truth from sender (True/False for truthful/deceptive)
  - `receiver_labels`: Perception from receiver (True/False/"NOANNOTATION")
  - `score_delta`: Power imbalance (difference in supply centers between sender and receiver)
  - Additional metadata: speakers, receivers, seasons, years, message indices

- **Data Statistics**: 
  - Messages range from single words to paragraphs
  - <10% of receiver annotations are missing ("NOANNOTATION")
  - Power imbalance ranges from -18 to +18 supply centers

## Task Definitions

### 1. Actual Lie Detection (Sender Task)
- **Goal**: Predict if the sender intentionally sent a deceptive message
- **Label Source**: `sender_labels` (self-reported by sender)
- **Use Case**: Understanding deceptive intent in strategic communication

### 2. Suspected Lie Detection (Receiver Task)
- **Goal**: Predict if the receiver perceived the message as deceptive
- **Label Source**: `receiver_labels` (receiver's perception)
- **Use Case**: Understanding how deception is perceived and detected by others
- **Challenge**: More difficult than actual lie detection, as it requires modeling receiver psychology

## Model Architectures

The project implements a comprehensive suite of models, from simple baselines to state-of-the-art neural architectures:

### Baselines
1. **Human Baseline**: Measures agreement between sender and receiver labels (how well receivers detect actual lies)
2. **Random Baseline**: Random predictions for comparison
3. **Majority Class Baseline**: Always predicts the majority class

### Traditional Machine Learning Models

#### 1. Harbringers Model (`harbringers.py`)
- **Approach**: Logistic regression with linguistic features from a Diplomacy-specific lexicon
- **Features**: Binary indicators for presence of specific linguistic markers (e.g., "but", country names, etc.)
- **Power Features**: Optional binary features for severe power imbalances (>4 or <-4 supply centers)
- **Implementation**: Uses scikit-learn with balanced class weights

#### 2. Bag of Words Model (`bagofwords.py`)
- **Approach**: Logistic regression with bag-of-words features
- **Features**: Count vectorization of message text (with stop word removal)
- **Power Features**: Optional binary power imbalance features
- **Tokenization**: Uses spaCy for tokenization, normalizes numbers to `_NUM_`

### Neural Models (AllenNLP-based)

All neural models support two configurations:
- **Without Power**: Text-only models
- **With Power**: Models incorporating `score_delta` as an additional feature

#### 1. LSTM (`lstm.jsonnet`)
- **Architecture**: Single LSTM encoder for individual messages
- **Reader**: `MessageReader` (processes messages independently)
- **Use Case**: Baseline neural model without conversation context

#### 2. Context LSTM (`contextlstm.jsonnet`)
- **Architecture**: Hierarchical LSTM
  - Message-level encoder (LSTM or pooled RNN)
  - Conversation-level encoder (LSTM over message representations)
- **Reader**: `DiplomacyReader` (processes full conversations)
- **Key Innovation**: Captures conversation context and message sequence

#### 3. BERT + Context (`bert+context.jsonnet`)
- **Architecture**: Hierarchical model with BERT
  - Message encoder: BERT (bert-base-uncased) for individual messages
  - Conversation encoder: LSTM over BERT-encoded messages
- **Reader**: `DiplomacyReader`
- **State-of-the-art**: Best performing model, leveraging pre-trained BERT embeddings

#### Model Components:
- **LieDetector**: Simple message-level classifier (for LSTM models)
- **HierarchicalLSTM**: Conversation-aware classifier (for context models)
- **PooledRNN**: Multi-pooling encoder (max, mean, last hidden states)
- **Loss Function**: Weighted cross-entropy to handle class imbalance (lies are rare)

## Technical Architecture

### Data Processing
- **Readers**: Two AllenNLP dataset readers
  - `MessageReader`: For single-message models (LSTM)
  - `DiplomacyReader`: For conversation-level models (context-aware)
- **Preprocessing**: `singlemessage_format.py` converts conversation format for LSTM models

### Training Configuration
- **Framework**: AllenNLP 0.9.0
- **Optimization**: Adam optimizer with learning rate 0.0003
- **Training**: 15 epochs with early stopping (patience=10)
- **Evaluation**: Macro F1 score as primary metric
- **Class Imbalance**: Handled via weighted loss functions (pos_weight ~15 for lies)

### Model Variants
Each model type has configurations for:
- **Actual Lie**: `configs/actual_lie/`
- **Suspected Lie**: `configs/suspected_lie/`
- **With/Without Power**: Separate configs (e.g., `bert+context.jsonnet` vs `bert+context+power.jsonnet`)

## Key Findings (Inferred from Code Structure)

1. **Context Matters**: Hierarchical models (contextlstm, bert+context) outperform single-message models
2. **Power Imbalance**: Including power features (`score_delta`) improves performance
3. **BERT Advantage**: BERT-based models achieve best results
4. **Task Difficulty**: Suspected lie detection is harder than actual lie detection
5. **Class Imbalance**: Lies are rare (~6% based on pos_weight of 15), requiring careful handling

## Usage

### Setup
```bash
conda create --name diplomacy python=3.7
conda activate diplomacy
pip install -e .
```

### Running Models

**Single Model**:
```bash
allennlp train -f --include-package diplomacy -s logdir configs/actual_lie/bert+context.jsonnet
```

**All Models** (via script):
```bash
bash run_models.sh
```

**Individual Baselines**:
```bash
python diplomacy/models/human_baseline.py
python diplomacy/models/random_and_majority_baselines.py
python diplomacy/models/harbringers.py s n  # actual lie, no power
python diplomacy/models/bagofwords.py r y   # suspected lie, with power
```

## Dependencies

- **allennlp** ~= 0.9.0 (neural models)
- **scikit-learn** (traditional ML models)
- **spaCy** (tokenization)
- **jsonlines** (data format)
- **torch** (via allennlp)

## Additional Resources

- **ConvoKit**: Alternative data representation available at https://convokit.cornell.edu/documentation/diplomacy.html
- **Game Moves**: Full game data (without messages) available in `moves/` folder
- **License**: Creative Commons Attribution 4.0 International

## Citation

```bibtex
@inproceedings{Peskov:Cheng:Elgohary:Barrow:Danescu-Niculescu-Mizil:Boyd-Graber-2020,
    Title = {It Takes Two to Lie: One to Lie and One to Listen},
    Author = {Denis Peskov and Benny Cheng and Ahmed Elgohary and Joe Barrow and Cristian Danescu-Niculescu-Mizil and Jordan Boyd-Graber},
    Booktitle = {Association for Computational Linguistics},
    Year = {2020},
    Location = {The Cyberverse Simulacrum of Seattle},
}
```

## Project Structure

```
2020_acl_diplomacy/
├── configs/              # Model configurations (actual_lie/, suspected_lie/)
├── data/                 # Dataset (train/validation/test.jsonl)
├── diplomacy/
│   ├── models/           # Model implementations
│   │   ├── lie_detector.py      # Simple message classifier
│   │   ├── hlstm.py             # Hierarchical LSTM
│   │   ├── pooled_rnn.py        # Multi-pooling encoder
│   │   ├── harbringers.py       # Linguistic features + LR
│   │   ├── bagofwords.py        # BoW + LR
│   │   ├── human_baseline.py    # Human agreement baseline
│   │   └── random_and_majority_baselines.py
│   └── readers/          # Data readers
│       ├── game_reader.py        # Conversation-level reader
│       └── message_reader.py    # Message-level reader
├── moves/                # Full game move data
├── utils/                # Utilities (lexicon, formatting)
└── run_models.sh         # Script to run all models
```

