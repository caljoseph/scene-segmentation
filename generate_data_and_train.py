import os
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from models import SimpleClassifier, LSTMSequenceClassifier, CNNSequenceClassifier
from embedder import Embedder


# ==================== CONFIGURATION ====================
# Data Configuration - Choose ONE mode below:

# ============================================================
# MODE 1: RANDOM SPLIT (recommended for general training)
# ============================================================
# Pool all datasets together and randomly split train/val/test
# Set RANDOM_SPLIT = True and list your datasets in DATASET_DIRS

RANDOM_SPLIT = True

DATASET_DIRS = [
    "data/datasets/german_scene/train_full",
    "data/datasets/german_scene/test_full",
    "data/datasets/story_lab",
]

# ============================================================
# MODE 2: FORCED SPLITS (for cross-lingual transfer, etc.)
# ============================================================
# Explicitly assign datasets to train/val/test splits
# Set RANDOM_SPLIT = False and configure TRAIN/VAL/TEST_SOURCES
#
# RANDOM_SPLIT = False
#
# TRAIN_SOURCES = [
#     "data/datasets/german_scene/train_full",
# ]
#
# VAL_SOURCES = [
#     "data/datasets/german_scene/test_full",
# ]
#
# TEST_SOURCES = [
#     "data/datasets/story_lab",
# ]
# ============================================================

CHUNK_SIZE = 7
MIN_CONTEXT = 1

# Model
CLASSIFIER_TYPE = "llm-finetune"  # Choose: "lstm", "cnn", "simple", or "llm-finetune"
EMBEDDING_MODEL = "BAAI/bge-m3"  # For lstm/cnn/simple only
EMBEDDING_DIM = 1024  # For lstm/cnn/simple only
HIDDEN_DIM = 512  # For LSTM/CNN (ignored for simple/llm-finetune)
DROPOUT = 0.3

# LLM Fine-tuning (only used when CLASSIFIER_TYPE = "llm-finetune")
LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B"  # or "meta-llama/Llama-3.2-1B" for faster experiments
LLM_MAX_LENGTH = 512  # Max tokens for concatenated 7 sentences
USE_LORA = True  # Use LoRA for parameter-efficient fine-tuning
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0.1  # LoRA dropout
USE_4BIT = True  # Use 4-bit quantization (QLoRA) - reduces memory

# Training
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 200
EARLY_STOP_PATIENCE = 50

# Splits (only used when RANDOM_SPLIT = True)
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42

# Default empty lists for forced mode (don't change these here, change above)
if 'TRAIN_SOURCES' not in locals():
    TRAIN_SOURCES = []
if 'VAL_SOURCES' not in locals():
    VAL_SOURCES = []
if 'TEST_SOURCES' not in locals():
    TEST_SOURCES = []

# Validation and output naming
if RANDOM_SPLIT:
    # Validate random split mode
    assert DATASET_DIRS, "RANDOM_SPLIT=True but DATASET_DIRS is empty! Add datasets to DATASET_DIRS."
    assert abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) < 0.01, \
        f"TRAIN_SIZE + VAL_SIZE + TEST_SIZE must equal 1.0 (got {TRAIN_SIZE + VAL_SIZE + TEST_SIZE})"

    # Build dataset name from DATASET_DIRS
    _dataset_parts = [Path(d).name for d in DATASET_DIRS]
    DATASET_NAME = "+".join(_dataset_parts) if len(_dataset_parts) <= 3 else f"{len(_dataset_parts)}_datasets"
else:
    # Validate forced split mode
    assert TRAIN_SOURCES or VAL_SOURCES or TEST_SOURCES, \
        "RANDOM_SPLIT=False but no sources specified! Set TRAIN_SOURCES, VAL_SOURCES, or TEST_SOURCES."
    assert TRAIN_SOURCES, \
        "RANDOM_SPLIT=False but TRAIN_SOURCES is empty! You must have training data."

    # Build dataset name from forced sources
    _all_sources = TRAIN_SOURCES + VAL_SOURCES + TEST_SOURCES
    _dataset_parts = [Path(d).name for d in _all_sources]
    DATASET_NAME = "+".join(_dataset_parts) if len(_dataset_parts) <= 3 else f"{len(_dataset_parts)}_datasets"

# Output paths
DATASET_OUTPUT_DIR = f"data/generated/{DATASET_NAME}"
if CLASSIFIER_TYPE == "llm-finetune":
    LLM_MODEL_SHORT = LLM_MODEL_NAME.split('/')[-1]  # e.g., "Llama-3.2-3B"
    CLASSIFIER_OUTPUT = f"data/models/llm-finetune_{LLM_MODEL_SHORT}_{DATASET_NAME}"
    PLOT_OUTPUT = f"plots/llm-finetune_{LLM_MODEL_SHORT}_{DATASET_NAME}_training.png"
else:
    MODEL_NAME_SHORT = EMBEDDING_MODEL.replace('/', '-')
    CLASSIFIER_OUTPUT = f"data/models/{CLASSIFIER_TYPE}_{MODEL_NAME_SHORT}_{DATASET_NAME}_{HIDDEN_DIM}h.pth"
    PLOT_OUTPUT = f"plots/{CLASSIFIER_TYPE}_{DATASET_NAME}_training.png"
# =======================================================


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_annotated_csv(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
        df[0] = pd.to_numeric(df[0], errors='coerce').fillna(0).astype(int)

        if 1 not in df[0].values:
            return None

        first_scene_transition_index = df[df[0] == 1].index[0]
        df[1] = df[1].astype(str).apply(lambda x: x.replace('\n', ' '))
        df[0] = df[0].fillna(0)
        df = df.dropna(subset=[1])
        df = df[df[1] != 0]
        df = df[df[1] != '0']
        df = df.iloc[first_scene_transition_index:].reset_index(drop=True)

        return df

    except Exception as e:
        print(f"  ✗ Error reading {file_path.name}: {e}")
        return None


def load_all_stories(random_split, dataset_dirs=None, train_sources=None, val_sources=None, test_sources=None):
    """Load stories from directories with optional split assignments."""
    all_stories = {}
    source_assignments = {}  # Track which split each story belongs to

    if random_split:
        # MODE 1: Load all datasets, mark as "all" for random splitting
        for source_dir in dataset_dirs:
            print(f"\n✓ Loading from: {source_dir} (random split)")
            input_path = Path(source_dir)
            csv_files = sorted(input_path.glob("*.csv"))

            if not csv_files:
                print(f"  ⚠️  No CSV files found, skipping...")
                continue

            for csv_file in csv_files:
                df = load_annotated_csv(csv_file)
                if df is not None:
                    source_name = Path(source_dir).name
                    story_key = f"{source_name}__{csv_file.stem}"
                    all_stories[story_key] = df
                    source_assignments[story_key] = "all"
                    print(f"  ✓ {csv_file.stem}: {len(df)} sentences, {(df[0] == 1).sum()} transitions")
    else:
        # MODE 2: Load from train/val/test sources separately
        for source_dir in (train_sources or []):
            print(f"\n✓ Loading from: {source_dir} → TRAIN")
            input_path = Path(source_dir)
            for csv_file in sorted(input_path.glob("*.csv")):
                df = load_annotated_csv(csv_file)
                if df is not None:
                    source_name = Path(source_dir).name
                    story_key = f"{source_name}__{csv_file.stem}"
                    all_stories[story_key] = df
                    source_assignments[story_key] = "train"
                    print(f"  ✓ {csv_file.stem}: {len(df)} sentences, {(df[0] == 1).sum()} transitions")

        for source_dir in (val_sources or []):
            print(f"\n✓ Loading from: {source_dir} → VAL")
            input_path = Path(source_dir)
            for csv_file in sorted(input_path.glob("*.csv")):
                df = load_annotated_csv(csv_file)
                if df is not None:
                    source_name = Path(source_dir).name
                    story_key = f"{source_name}__{csv_file.stem}"
                    all_stories[story_key] = df
                    source_assignments[story_key] = "val"
                    print(f"  ✓ {csv_file.stem}: {len(df)} sentences, {(df[0] == 1).sum()} transitions")

        for source_dir in (test_sources or []):
            print(f"\n✓ Loading from: {source_dir} → TEST")
            input_path = Path(source_dir)
            for csv_file in sorted(input_path.glob("*.csv")):
                df = load_annotated_csv(csv_file)
                if df is not None:
                    source_name = Path(source_dir).name
                    story_key = f"{source_name}__{csv_file.stem}"
                    all_stories[story_key] = df
                    source_assignments[story_key] = "test"
                    print(f"  ✓ {csv_file.stem}: {len(df)} sentences, {(df[0] == 1).sum()} transitions")

    if not all_stories:
        raise ValueError(f"No stories loaded from any source!")

    num_sources = len(dataset_dirs) if random_split else len((train_sources or []) + (val_sources or []) + (test_sources or []))
    print(f"\n✓ Loaded {len(all_stories)} total stories from {num_sources} sources")
    return all_stories, source_assignments


def extract_transitions(stories):
    transitions = []
    for story_name, df in stories.items():
        transition_indices = df.index[df[0] == 1].tolist()
        for idx in transition_indices:
            transitions.append({
                'story_name': story_name,
                'story_df': df,
                'idx': idx,
                'id': f"{story_name}_{idx}"
            })
    return transitions


def split_transitions(transitions, source_assignments, train_size, val_size, test_size, seed):
    """
    Split transitions respecting source assignments.

    If source_assignment is:
    - "all": use normal train/val/test splits
    - "train"/"val"/"test": force into that split
    """
    if abs(train_size + val_size + test_size - 1.0) > 0.01:
        raise ValueError("Split sizes must sum to 1.0")

    # Separate transitions by assignment
    forced_train = []
    forced_val = []
    forced_test = []
    auto_split = []

    for trans in transitions:
        story_name = trans['story_name']
        assignment = source_assignments.get(story_name, "all")

        if assignment == "train":
            forced_train.append(trans)
        elif assignment == "val":
            forced_val.append(trans)
        elif assignment == "test":
            forced_test.append(trans)
        else:  # "all" or None
            auto_split.append(trans)

    # Split the "all" transitions normally
    if auto_split:
        indices = np.arange(len(auto_split))

        if val_size == 0.0:
            train_indices, test_indices = train_test_split(
                indices, test_size=test_size, random_state=seed
            )
            val_indices = np.array([])
        else:
            train_indices, temp_indices = train_test_split(
                indices, test_size=(val_size + test_size), random_state=seed
            )
            val_indices, test_indices = train_test_split(
                temp_indices, test_size=test_size/(val_size + test_size), random_state=seed
            )

        auto_train = [auto_split[i] for i in train_indices]
        auto_val = [auto_split[i] for i in val_indices]
        auto_test = [auto_split[i] for i in test_indices]
    else:
        auto_train, auto_val, auto_test = [], [], []

    # Combine forced and auto splits
    train_transitions = forced_train + auto_train
    val_transitions = forced_val + auto_val
    test_transitions = forced_test + auto_test

    return train_transitions, val_transitions, test_transitions


def generate_transition_chunks(transitions):
    chunks = []

    for trans in transitions:
        df = trans['story_df']
        idx = trans['idx']

        min_offset = MIN_CONTEXT
        max_offset = CHUNK_SIZE - 1 - MIN_CONTEXT

        available_before = idx
        max_offset = min(max_offset, available_before)

        available_after = len(df) - 1 - idx
        min_offset = max(min_offset, CHUNK_SIZE - 1 - available_after)

        if idx == 0:
            min_offset = 0

        if min_offset > max_offset:
            continue

        for offset in range(min_offset, max_offset + 1):
            start_idx = idx - offset
            end_idx = idx + (CHUNK_SIZE - 1 - offset)

            if start_idx < 0 or end_idx >= len(df):
                continue

            chunk = df.iloc[start_idx:end_idx + 1]

            if len(chunk) == CHUNK_SIZE and (chunk[0] == 0).any():
                sentences = chunk[1].tolist()
                chunks.append(sentences)  # Keep as list of 7 strings

    return chunks


def generate_control_chunks(stories, transitions_in_split, target_count, seed):
    random.seed(seed)

    transition_indices_by_story = {}
    for trans in transitions_in_split:
        story_name = trans['story_name']
        if story_name not in transition_indices_by_story:
            transition_indices_by_story[story_name] = set()
        transition_indices_by_story[story_name].add(trans['idx'])

    chunks = []
    story_names = list(stories.keys())
    attempts = 0
    max_attempts = target_count * 100

    while len(chunks) < target_count and attempts < max_attempts:
        attempts += 1

        story_name = random.choice(story_names)
        df = stories[story_name]

        if len(df) < CHUNK_SIZE:
            continue

        idx = random.randint(0, len(df) - CHUNK_SIZE)
        chunk_indices = set(range(idx, idx + CHUNK_SIZE))

        transition_indices = transition_indices_by_story.get(story_name, set())
        if chunk_indices.intersection(transition_indices):
            continue

        chunk = df.iloc[idx:idx + CHUNK_SIZE]

        if len(chunk) == CHUNK_SIZE and (chunk[0] == 0).all():
            sentences = chunk[1].tolist()
            chunks.append(sentences)  # Keep as list of 7 strings

    if len(chunks) < target_count:
        print(f"  ⚠️  Warning: Only generated {len(chunks)}/{target_count} control chunks")

    return chunks


def generate_or_load_embeddings(chunks, labels, embedder, cache_file_base, device):
    """
    Generate embeddings for sentence sequences using per-sentence caching.

    Args:
        chunks: List of lists, each containing 7 sentences
        labels: numpy array of labels
        embedder: Embedder instance (with per-sentence cache)
        cache_file_base: Base path for batch cache files
        device: torch device

    Returns:
        embeddings: numpy array of shape (num_chunks, 7, embedding_dim)
        labels: numpy array of labels
    """
    embeddings_file = f"{cache_file_base}_embeddings.npy"
    labels_file = f"{cache_file_base}_labels.npy"

    # Check if we have a cached batch file
    if os.path.exists(embeddings_file) and os.path.exists(labels_file):
        embeddings = np.load(embeddings_file)
        loaded_labels = np.load(labels_file)
        print(f"  ✓ Loaded {len(embeddings)} embeddings from cache")
        return embeddings, loaded_labels

    print(f"  Generating embeddings for {len(chunks)} chunks (7 sentences each)...")

    # Flatten all sentences to embed in batch (per-sentence cache used internally)
    all_sentences = []
    for chunk in chunks:
        all_sentences.extend(chunk)

    print(f"    Total sentences: {len(all_sentences)} (checking per-sentence cache...)")

    # Use embedder's per-sentence cache
    all_embeddings = embedder.generate_embeddings(all_sentences)

    # Reshape into (num_chunks, 7, embedding_dim)
    embedding_dim = all_embeddings.shape[1]
    embeddings = all_embeddings.reshape(len(chunks), 7, embedding_dim)

    # Save batch cache
    Path(cache_file_base).parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file, embeddings)
    np.save(labels_file, labels)
    print(f"  ✓ Generated and saved {len(embeddings)} embeddings")

    return embeddings, labels


def save_dataset_to_csv(chunks, labels, output_file):
    """Save dataset with separate sentence columns."""
    rows = []
    for chunk, label in zip(chunks, labels):
        row = {'isTransition': int(label)}
        for i, sent in enumerate(chunk):
            row[f'sent{i+1}'] = sent
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved dataset to: {output_file}")


def create_data_loader(embeddings, labels, batch_size, shuffle=True):
    dataset = TensorDataset(
        torch.tensor(embeddings, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(classifier, optimizer, criterion, data_loader, device):
    classifier.train()
    total_loss = 0

    for embeddings, labels in data_loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = classifier(embeddings)
        loss = criterion(predictions.view(-1), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate_accuracy(classifier, data_loader, device):
    if len(data_loader) == 0:
        return 0.0

    classifier.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            predictions = classifier(embeddings)
            predicted_labels = predictions.view(-1) > 0.0
            correct += (predicted_labels == labels.byte()).sum().item()
            total += labels.size(0)

    return correct / total


def evaluate_metrics(classifier, data_loader, device):
    if len(data_loader) == 0:
        return 0.0, 0.0, 0.0, 0.0

    classifier.eval()
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            predictions = classifier(embeddings)
            predicted_labels = (predictions.view(-1) > 0.0).byte()
            labels = labels.byte()

            true_positives += ((predicted_labels == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted_labels == 1) & (labels == 0)).sum().item()
            true_negatives += ((predicted_labels == 0) & (labels == 0)).sum().item()
            false_negatives += ((predicted_labels == 0) & (labels == 1)).sum().item()

    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1


def evaluate_loss(classifier, data_loader, criterion, device):
    if len(data_loader) == 0:
        return 0.0

    classifier.eval()
    total_loss = 0

    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            predictions = classifier(embeddings)
            loss = criterion(predictions.view(-1), labels)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def main():
    print("=" * 60)
    print("Scene Transition Classifier: Data Generation + Training")
    print("=" * 60)

    set_random_seeds(RANDOM_SEED)
    device = get_device()
    print(f"\n✓ Random seed: {RANDOM_SEED}")
    print(f"✓ Device: {device}")
    print(f"✓ Dataset: {DATASET_NAME}")
    print(f"✓ Mode: {'Random Split' if RANDOM_SPLIT else 'Forced Splits'}")

    stories, source_assignments = load_all_stories(
        random_split=RANDOM_SPLIT,
        dataset_dirs=DATASET_DIRS if RANDOM_SPLIT else None,
        train_sources=TRAIN_SOURCES if not RANDOM_SPLIT else None,
        val_sources=VAL_SOURCES if not RANDOM_SPLIT else None,
        test_sources=TEST_SOURCES if not RANDOM_SPLIT else None
    )

    print("\n" + "=" * 60)
    print("Extracting and Splitting Transitions")
    print("=" * 60)

    all_transitions = extract_transitions(stories)
    print(f"\n✓ Extracted {len(all_transitions)} total transitions")

    train_transitions, val_transitions, test_transitions = split_transitions(
        all_transitions, source_assignments, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_SEED
    )

    print(f"✓ Split: {len(train_transitions)} train, {len(val_transitions)} val, {len(test_transitions)} test")

    print("\n" + "=" * 60)
    print("Generating Chunks")
    print("=" * 60)

    print("\nTrain set:")
    train_trans_chunks = generate_transition_chunks(train_transitions)
    print(f"  ✓ Generated {len(train_trans_chunks)} transition chunks (all positions)")

    train_ctrl_chunks = generate_control_chunks(stories, train_transitions, len(train_trans_chunks), RANDOM_SEED)
    print(f"  ✓ Generated {len(train_ctrl_chunks)} control chunks (random sampling)")

    train_texts = train_trans_chunks + train_ctrl_chunks
    train_labels = np.array([1] * len(train_trans_chunks) + [0] * len(train_ctrl_chunks))
    print(f"  ✓ Total train chunks: {len(train_texts)} ({len(train_trans_chunks)} transitions + {len(train_ctrl_chunks)} controls)")

    shuffle_indices = np.random.permutation(len(train_texts))
    train_texts = [train_texts[i] for i in shuffle_indices]
    train_labels = train_labels[shuffle_indices]

    save_dataset_to_csv(train_texts, train_labels, f"{DATASET_OUTPUT_DIR}/{DATASET_NAME}_train.csv")

    if len(val_transitions) > 0:
        print("\nValidation set:")
        val_trans_chunks = generate_transition_chunks(val_transitions)
        print(f"  ✓ Generated {len(val_trans_chunks)} transition chunks")

        val_ctrl_chunks = generate_control_chunks(stories, val_transitions, len(val_trans_chunks), RANDOM_SEED + 1)
        print(f"  ✓ Generated {len(val_ctrl_chunks)} control chunks")

        val_texts = val_trans_chunks + val_ctrl_chunks
        val_labels = np.array([1] * len(val_trans_chunks) + [0] * len(val_ctrl_chunks))
        print(f"  ✓ Total val chunks: {len(val_texts)} ({len(val_trans_chunks)} transitions + {len(val_ctrl_chunks)} controls)")

        shuffle_indices = np.random.permutation(len(val_texts))
        val_texts = [val_texts[i] for i in shuffle_indices]
        val_labels = val_labels[shuffle_indices]

        save_dataset_to_csv(val_texts, val_labels, f"{DATASET_OUTPUT_DIR}/{DATASET_NAME}_val.csv")
    else:
        val_texts, val_labels = [], np.array([])

    print("\nTest set:")
    test_trans_chunks = generate_transition_chunks(test_transitions)
    print(f"  ✓ Generated {len(test_trans_chunks)} transition chunks")

    test_ctrl_chunks = generate_control_chunks(stories, test_transitions, len(test_trans_chunks), RANDOM_SEED + 2)
    print(f"  ✓ Generated {len(test_ctrl_chunks)} control chunks")

    test_texts = test_trans_chunks + test_ctrl_chunks
    test_labels = np.array([1] * len(test_trans_chunks) + [0] * len(test_ctrl_chunks))
    print(f"  ✓ Total test chunks: {len(test_texts)} ({len(test_trans_chunks)} transitions + {len(test_ctrl_chunks)} controls)")

    shuffle_indices = np.random.permutation(len(test_texts))
    test_texts = [test_texts[i] for i in shuffle_indices]
    test_labels = test_labels[shuffle_indices]

    save_dataset_to_csv(test_texts, test_labels, f"{DATASET_OUTPUT_DIR}/{DATASET_NAME}_test.csv")

    # ============================================================
    # BRANCH: LLM Fine-Tuning vs Embedding-Based Classifiers
    # ============================================================

    if CLASSIFIER_TYPE == "llm-finetune":
        # LLM fine-tuning path - no embeddings needed
        print("\n" + "=" * 60)
        print("Preparing LLM Fine-Tuning")
        print("=" * 60)

        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            BitsAndBytesConfig,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding
        )
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from torch.utils.data import Dataset as TorchDataset

        # Check if 4-bit quantization is available (requires CUDA)
        can_use_4bit = USE_4BIT and torch.cuda.is_available()
        if USE_4BIT and not can_use_4bit:
            print("⚠️  4-bit quantization requires CUDA, disabling (will use fp32 on MPS/CPU)")

        # Device diagnostics
        print(f"\n🔍 Device Diagnostics:")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   MPS available: {torch.backends.mps.is_available()}")
        print(f"   MPS built: {torch.backends.mps.is_built()}")
        print(f"   Selected device: {device}")

        print(f"\n✓ Model: {LLM_MODEL_NAME}")
        print(f"✓ Max length: {LLM_MAX_LENGTH} tokens")
        print(f"✓ LoRA enabled: {USE_LORA} (r={LORA_R}, alpha={LORA_ALPHA})")
        print(f"✓ 4-bit quantization: {can_use_4bit}")

        # Load tokenizer and configure padding
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

        # Llama models don't have a pad token by default - set it to eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # For decoder models, pad on the left so the last token is always the final real token
        tokenizer.padding_side = 'left'

        # Create dataset class for HuggingFace Trainer
        class SceneDataset(TorchDataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                # Concatenate 7 sentences with space separator
                self.texts = [' '.join(chunk) for chunk in texts]
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }

        train_dataset = SceneDataset(train_texts, train_labels, tokenizer, LLM_MAX_LENGTH)
        val_dataset = SceneDataset(val_texts, val_labels, tokenizer, LLM_MAX_LENGTH) if len(val_texts) > 0 else None
        test_dataset = SceneDataset(test_texts, test_labels, tokenizer, LLM_MAX_LENGTH)

        print(f"\n✓ Created datasets: {len(train_dataset)} train, {len(val_dataset) if val_dataset else 0} val, {len(test_dataset)} test")

        # Load model with quantization (only on CUDA)
        print(f"\n✓ Loading model: {LLM_MODEL_NAME}...")
        if can_use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                LLM_MODEL_NAME,
                num_labels=2,
                quantization_config=bnb_config,
                device_map="auto"
            )
            if USE_LORA:
                model = prepare_model_for_kbit_training(model)
        else:
            # Load in full precision (fp32 or fp16)
            model = AutoModelForSequenceClassification.from_pretrained(
                LLM_MODEL_NAME,
                num_labels=2,
                torch_dtype=torch.float16 if device.type == 'mps' else torch.float32,
                pad_token_id=tokenizer.pad_token_id
            ).to(device)

        # Verify model device placement
        print(f"\n🔍 Model Device Check:")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Model dtype: {next(model.parameters()).dtype}")

        # Apply LoRA
        if USE_LORA:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Apply to all attention projections
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            print(f"   Model device after LoRA: {next(model.parameters()).device}")

        # Training arguments
        # Note: MPS support in Trainer requires setting these flags carefully
        training_args = TrainingArguments(
            output_dir=CLASSIFIER_OUTPUT,
            num_train_epochs=5,  # LLMs converge faster
            per_device_train_batch_size=2,  # Small batch for 3B model on MPS
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,  # Effective batch size = 2*2 = 4
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=100,
            logging_dir=f"{CLASSIFIER_OUTPUT}/logs",
            logging_steps=50,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_accuracy" if val_dataset else None,
            save_total_limit=3,
            report_to="none",  # Disable wandb/tensorboard
            fp16=False,  # MPS doesn't support fp16 training well
            bf16=False,  # MPS doesn't support bf16
            use_cpu=False,
            # Explicitly set device to avoid auto-detection issues
            no_cuda=True,  # Disable CUDA detection (we're on MPS)
            dataloader_pin_memory=False,  # MPS doesn't support pinned memory
        )

        # Define compute_metrics for evaluation
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = (predictions == labels).mean()
            return {"accuracy": accuracy}

        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )

        print("\n" + "=" * 60)
        print("Training LLM Classifier")
        print("=" * 60)

        # Train
        trainer.train()

        # Evaluate on test set
        print("\n✓ Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        print(f"✓ Test accuracy: {test_results['eval_accuracy']*100:.2f}%")

        # Save final model
        print(f"\n✓ Saving model to: {CLASSIFIER_OUTPUT}")
        trainer.save_model(CLASSIFIER_OUTPUT)
        tokenizer.save_pretrained(CLASSIFIER_OUTPUT)

        print("=" * 60)
        print("✓ LLM fine-tuning complete!")
        print(f"✓ Model saved to: {CLASSIFIER_OUTPUT}")
        print("=" * 60)

        return  # Exit early for LLM path

    # ============================================================
    # Embedding-based classifiers (LSTM, CNN, Simple)
    # ============================================================

    print("\n" + "=" * 60)
    print("Generating Embeddings")
    print("=" * 60)

    print(f"\n✓ Initializing embedder: {EMBEDDING_MODEL}")
    embedder = Embedder(model_name=EMBEDDING_MODEL, cache_dir="data/cache")

    print("\nTrain embeddings:")
    train_embeddings, train_labels = generate_or_load_embeddings(
        train_texts, train_labels, embedder,
        f"{DATASET_OUTPUT_DIR}/embeddings/train", device
    )

    if len(val_texts) > 0:
        print("\nValidation embeddings:")
        val_embeddings, val_labels = generate_or_load_embeddings(
            val_texts, val_labels, embedder,
            f"{DATASET_OUTPUT_DIR}/embeddings/val", device
        )
    else:
        val_embeddings, val_labels = np.array([]).reshape(0, 7, EMBEDDING_DIM), np.array([])

    print("\nTest embeddings:")
    test_embeddings, test_labels = generate_or_load_embeddings(
        test_texts, test_labels, embedder,
        f"{DATASET_OUTPUT_DIR}/embeddings/test", device
    )

    train_loader = create_data_loader(train_embeddings, train_labels, BATCH_SIZE, shuffle=True)
    val_loader = create_data_loader(val_embeddings, val_labels, BATCH_SIZE, shuffle=False)
    test_loader = create_data_loader(test_embeddings, test_labels, BATCH_SIZE, shuffle=False)

    print("\n" + "=" * 60)
    print("Training Classifier")
    print("=" * 60)

    # Initialize classifier based on type
    if CLASSIFIER_TYPE == "lstm":
        classifier = LSTMSequenceClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            dropout=DROPOUT
        ).to(device)
        print(f"\n✓ Model: LSTM (input: {EMBEDDING_DIM}, hidden: {HIDDEN_DIM}, layers: 2, bidirectional)")
    elif CLASSIFIER_TYPE == "cnn":
        classifier = CNNSequenceClassifier(
            embedding_dim=EMBEDDING_DIM,
            num_filters=256,
            kernel_sizes=[2, 3, 4],
            dropout=DROPOUT
        ).to(device)
        print(f"\n✓ Model: CNN (input: {EMBEDDING_DIM}, filters: 256, kernels: [2,3,4])")
    elif CLASSIFIER_TYPE == "simple":
        # For simple classifier, we need to average the 7 embeddings
        print(f"\n⚠️  Note: SimpleClassifier expects concatenated embeddings, averaging 7 sentence embeddings")
        train_embeddings = train_embeddings.mean(axis=1)  # (N, 7, D) → (N, D)
        if len(val_embeddings) > 0:
            val_embeddings = val_embeddings.mean(axis=1)
        test_embeddings = test_embeddings.mean(axis=1)

        # Recreate dataloaders with averaged embeddings
        train_loader = create_data_loader(train_embeddings, train_labels, BATCH_SIZE, shuffle=True)
        val_loader = create_data_loader(val_embeddings, val_labels, BATCH_SIZE, shuffle=False)
        test_loader = create_data_loader(test_embeddings, test_labels, BATCH_SIZE, shuffle=False)

        classifier = SimpleClassifier(
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=1,
            dropout_prob=DROPOUT
        ).to(device)
        print(f"\n✓ Model: SimpleClassifier ({EMBEDDING_DIM} → {HIDDEN_DIM} → 1)")
    else:
        raise ValueError(f"Unknown CLASSIFIER_TYPE: {CLASSIFIER_TYPE}. Choose 'lstm', 'cnn', or 'simple'")

    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    print(f"✓ Optimizer: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"✓ Scheduler: ReduceLROnPlateau")
    print(f"✓ Training for {EPOCHS} epochs (patience={EARLY_STOP_PATIENCE})")

    Path("classifiers").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    max_accuracy = 0
    stopper_count = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("\n" + "=" * 60)
    for epoch in range(EPOCHS):
        train_loss = train_epoch(classifier, optimizer, criterion, train_loader, device)
        train_losses.append(train_loss)

        train_accuracy = evaluate_accuracy(classifier, train_loader, device)
        val_loss = evaluate_loss(classifier, val_loader, criterion, device)
        val_accuracy = evaluate_accuracy(classifier, val_loader, device)
        test_accuracy = evaluate_accuracy(classifier, test_loader, device)

        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step(val_loss if len(val_loader) > 0 else train_loss)

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy*100:.2f}% | "
              f"Test Acc: {test_accuracy*100:.2f}%")

        if test_accuracy > max_accuracy:
            stopper_count = 0
            max_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'embedding_dim': EMBEDDING_DIM,
                'embedder_name': EMBEDDING_MODEL,
                'test_accuracy': max_accuracy,
                'hidden_dim': HIDDEN_DIM,
                'dropout': DROPOUT
            }, CLASSIFIER_OUTPUT)
            print(f"  ✓ Saved new best model (test acc: {test_accuracy*100:.2f}%)")
        else:
            stopper_count += 1

        if stopper_count >= EARLY_STOP_PATIENCE:
            print(f"\n✓ Early stopping triggered after {EARLY_STOP_PATIENCE} epochs without improvement")
            break

    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=150, bbox_inches='tight')

    final_test_accuracy, final_precision, final_recall, final_f1 = evaluate_metrics(classifier, test_loader, device)

    print(f"\n✓ Best test accuracy: {max_accuracy*100:.2f}%")
    print(f"✓ Final test accuracy: {final_test_accuracy*100:.2f}%")
    print(f"✓ Final precision: {final_precision*100:.2f}%")
    print(f"✓ Final recall: {final_recall*100:.2f}%")
    print(f"✓ Final F1 score: {final_f1*100:.2f}%")
    print(f"✓ Model saved to: {CLASSIFIER_OUTPUT}")
    print(f"✓ Plots saved to: {PLOT_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
