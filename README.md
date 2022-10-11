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

## SkipGram

Each line in the corpus file is a document, and the tokens should be separated by whitespace.

```python
import corpusit

corpus_path = 'corpusit/data/corpus.txt'
vocab = corpusit.Vocab.build(corpus_path, min_count=1, unk='<unk>')

dataset = corpusit.SkipGramDataset(
    path_to_corpus=corpus_path,
    vocab=vocab,
    win_size=10,
    sep=" ",
    mode="onepass",       # onepass | repeat | shuffle
    subsample=1e-3,
    power=0.75,
    n_neg=1,
)

it = dataset.positive_sampler(batch_size=100, seed=0, num_threads=4)

for i, pair in enumerate(it):
    print(f'Iter {i:>4d}, shape={pair.shape}. First pair: '
          f'{pair[0,0]:>5} ({vocab.i2s[pair[0,0]]:>10}), '
          f'{pair[0,1]:>5} ({vocab.i2s[pair[0,1]]:>10})')

# Return:
# Iter    0, shape=(100, 2). First pair:    14 (        is),    10 ( anarchism)
# Iter    1, shape=(100, 2). First pair:     8 (        to),   540 (      and/)
# Iter    2, shape=(100, 2). First pair:   775 (constitutes),    34 (anarchists)
# Iter    3, shape=(100, 2). First pair:    72 (     other),   214 (  criteria)
# Iter    4, shape=(100, 2). First pair:   650 (  defining),   487 ( companion)
# ...
```


## SkipGram with negative sampling
```python
it = dataset.sampler(100, seed=0, num_threads=4)

for i, res in enumerate(it):
    pair, label = res
    print(f'Iter {i:>4d}, shape={pair.shape}. First pair: '
          f'{pair[0,0]:>5} ({vocab.i2s[pair[0,0]]:>10}), '
          f'{pair[0,1]:>5} ({vocab.i2s[pair[0,1]]:>10}), '
          f'label = {label[0]}')

# Returns:
# Iter    0, shape=(200, 2). First pair:    15 (        is),    10 ( anarchism), label = True
# Iter    1, shape=(200, 2). First pair:     9 (        to),   722 (      and/), label = True
# Iter    2, shape=(200, 2). First pair:   389 (constitutes),    34 (anarchists), label = True
# Iter    3, shape=(200, 2). First pair:    73 (     other),   212 (  criteria), label = True
# Iter    4, shape=(200, 2). First pair:   445 (  defining),   793 ( companion), label = True
# ...
```

# Roadmap
- GloVe


# License
MIT