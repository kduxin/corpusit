#!/usr/bin/env python3
"""
Example demonstrating the new Corpusit API without Vocab and SkipGramDataset classes.

This example shows how to use the simplified API for SkipGram training.
"""

import corpusit
import numpy as np
from collections import Counter

def create_vocabulary_and_counts(text_sequences):
    """
    Create word counts and word-to-id mapping from text sequences.
    This replaces the old Vocab.build() functionality.
    """
    # Flatten all sequences and count words
    all_words = []
    for seq in text_sequences:
        all_words.extend(seq.split())
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Create word-to-id mapping
    unique_words = list(word_counts.keys())
    word_to_id = {word: idx for idx, word in enumerate(unique_words)}
    
    # Create word counts mapping (word_id -> count)
    id_to_count = {word_to_id[word]: count for word, count in word_counts.items()}
    
    return word_to_id, id_to_count

def example_positive_sampling():
    """Example of positive sampling with the new API."""
    print("=== Positive Sampling Example ===")
    
    # Sample text sequences
    text_sequences = [
        "hello world python programming",
        "world python machine learning",
        "python programming is fun",
        "machine learning with python"
    ]
    
    # Create vocabulary and counts
    word_to_id, word_counts = create_vocabulary_and_counts(text_sequences)
    print(f"Vocabulary: {word_to_id}")
    print(f"Word counts: {word_counts}")
    
    # Create SkipGram configuration
    config = corpusit.SkipGramConfig(
        word_counts=word_counts,
        win_size=3,
        subsample=1e-3,
        power=0.75,
        n_neg=1
    )
    
    # Create positive sampler
    sampler = config.positive_sampler(seed=42)
    
    # Process sequences
    for i, text_seq in enumerate(text_sequences):
        # Convert text to word IDs
        word_ids = [word_to_id[word] for word in text_seq.split()]
        
        # Generate positive pairs
        pairs = sampler.process_sequence(word_ids)
        
        print(f"\nSequence {i+1}: '{text_seq}'")
        print(f"Word IDs: {word_ids}")
        print(f"Generated {len(pairs)} positive pairs")
        print(f"First few pairs: {pairs[:3] if len(pairs) > 0 else 'No pairs'}")
        
        # Show word pairs
        if len(pairs) > 0:
            id_to_word = {v: k for k, v in word_to_id.items()}
            print("Word pairs:")
            for pair in pairs[:5]:  # Show first 5 pairs
                word1 = id_to_word.get(pair[0], f"ID_{pair[0]}")
                word2 = id_to_word.get(pair[1], f"ID_{pair[1]}")
                print(f"  ({word1}, {word2})")

def example_negative_sampling():
    """Example of negative sampling with the new API."""
    print("\n=== Negative Sampling Example ===")
    
    # Sample text sequences
    text_sequences = [
        "hello world python programming",
        "world python machine learning",
        "python programming is fun"
    ]
    
    # Create vocabulary and counts
    word_to_id, word_counts = create_vocabulary_and_counts(text_sequences)
    
    # Create SkipGram configuration
    config = corpusit.SkipGramConfig(
        word_counts=word_counts,
        win_size=2,
        subsample=1e-3,
        power=0.75,
        n_neg=2  # Generate 2 negative samples per positive
    )
    
    # Create sampler with negative sampling
    sampler = config.sampler(seed=42, num_threads=2)
    
    # Process sequences
    for i, text_seq in enumerate(text_sequences):
        # Convert text to word IDs
        word_ids = [word_to_id[word] for word in text_seq.split()]
        
        # Generate positive and negative pairs
        pairs, labels = sampler.process_sequence(word_ids)
        
        print(f"\nSequence {i+1}: '{text_seq}'")
        print(f"Word IDs: {word_ids}")
        print(f"Generated {len(pairs)} samples")
        print(f"Positive samples: {np.sum(labels)}")
        print(f"Negative samples: {np.sum(~labels)}")
        
        # Show some samples
        if len(pairs) > 0:
            id_to_word = {v: k for k, v in word_to_id.items()}
            print("Sample pairs (first 6):")
            for j, (pair, label) in enumerate(zip(pairs[:6], labels[:6])):
                word1 = id_to_word.get(pair[0], f"ID_{pair[0]}")
                word2 = id_to_word.get(pair[1], f"ID_{pair[1]}")
                label_str = "positive" if label else "negative"
                print(f"  ({word1}, {word2}) - {label_str}")

def example_with_tokenization():
    """Example using the tokenization-enabled API."""
    print("\n=== Tokenization Example ===")
    
    # Sample text sequences
    text_sequences = [
        "hello world python programming",
        "world python machine learning",
        "python programming is fun"
    ]
    
    # Create vocabulary and counts
    word_to_id, word_counts = create_vocabulary_and_counts(text_sequences)
    
    # Create configuration with tokenization support
    config = corpusit.SkipGramConfigWithTokenization(
        word_counts=word_counts,
        word_to_id=word_to_id,
        separator=" ",
        win_size=2,
        subsample=1e-3,
        power=0.75,
        n_neg=1
    )
    
    # Create sampler
    sampler = config.sampler(seed=42, num_threads=2)
    
    # Process raw text sequences directly
    pairs, labels = sampler.process_string_sequences(text_sequences)
    
    print(f"Processed {len(text_sequences)} text sequences")
    print(f"Generated {len(pairs)} samples")
    print(f"Positive samples: {np.sum(labels)}")
    print(f"Negative samples: {np.sum(~labels)}")
    
    # Show some samples
    if len(pairs) > 0:
        id_to_word = {v: k for k, v in word_to_id.items()}
        print("Sample pairs (first 6):")
        for j, (pair, label) in enumerate(zip(pairs[:6], labels[:6])):
            word1 = id_to_word.get(pair[0], f"ID_{pair[0]}")
            word2 = id_to_word.get(pair[1], f"ID_{pair[1]}")
            label_str = "positive" if label else "negative"
            print(f"  ({word1}, {word2}) - {label_str}")

if __name__ == "__main__":
    print("Corpusit New API Examples")
    print("=" * 50)
    
    try:
        example_positive_sampling()
        example_negative_sampling()
        example_with_tokenization()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
