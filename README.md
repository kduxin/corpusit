# Corpusit
`corpusit` provides easy-to-use dataset iterators for natural language modeling
tasks, such as SkipGram.

It is written in rust to enable fast multi-threading random sampling with
deterministic results. So you dont have to worry about the speed /
reproducibility.

Corpusit does not provide tokenization functionalities. So please use `corpusit`
on tokenized corpus files (plain texts).

# Environment

Python >= 3.6

# Installation

```bash
$ pip install corpusit
```

## On Windows and MacOS

Please install [rust](https://www.rust-lang.org/tools/install) compiler before
executing `pip install corpusit`. 

# Usage

## SkipGram with Positive Sampling

Process tokenized sequences to generate positive SkipGram pairs:

```python
import corpusit
import numpy as np

# Create word counts mapping (word_id -> count)
word_counts = {0: 100, 1: 50, 2: 200, 3: 75, 4: 150}

# Create SkipGram configuration
config = corpusit.SkipGramConfig(
    word_counts=word_counts,
    win_size=5,
    subsample=1e-3,
    power=0.75,
    n_neg=1
)

# Create positive sampler
sampler = config.positive_sampler(seed=0)

# Process a sequence of word IDs
sequence = [0, 1, 2, 3, 4, 1, 2, 0]
pairs = sampler.process_sequence(sequence)
print(f'Generated {len(pairs)} positive pairs')
print(f'Shape: {pairs.shape}')
print(f'First few pairs: {pairs[:3]}')
```

## SkipGram with Negative Sampling

Generate both positive and negative samples with labels:

```python
# Create sampler with negative sampling
sampler = config.sampler(seed=0, num_threads=4)

# Process sequences
sequences = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 0]]
pairs, labels = sampler.process_sequences(sequences)

print(f'Generated {len(pairs)} samples')
print(f'Pairs shape: {pairs.shape}')
print(f'Labels shape: {labels.shape}')
print(f'Positive samples: {np.sum(labels)}')
print(f'Negative samples: {np.sum(~labels)}')
```

## SkipGram with Tokenization

Process raw text sequences with automatic tokenization:

```python
# Create configuration with tokenization support
word_counts = {0: 100, 1: 50, 2: 200, 3: 75, 4: 150}
word_to_id = {"hello": 0, "world": 1, "python": 2, "rust": 3, "fast": 4}

config = corpusit.SkipGramConfigWithTokenization(
    word_counts=word_counts,
    word_to_id=word_to_id,
    separator=" ",
    win_size=5,
    subsample=1e-3,
    power=0.75,
    n_neg=1
)

# Create sampler
sampler = config.sampler(seed=0, num_threads=4)

# Process raw text
text_sequences = ["hello world python", "world python rust", "python rust fast"]
pairs, labels = sampler.process_string_sequences(text_sequences)

print(f'Generated {len(pairs)} samples from text')
print(f'First few pairs: {pairs[:3]}')
print(f'Labels: {labels[:3]}')
```

# Roadmap
- GloVe


# License
MIT