from typing import List, Literal, Union, Mapping, NewType

uint = NewType("UnsignedInt", int)

class Vocab:
    """
    A vocabulary that contains indexed words and their counts in a corpus.
    """

    """mapping from word string to index"""
    s2i: Mapping[str, int]
    """ mapping from word index to string"""
    i2s: Mapping[int, str]
    """ mapping from word index to count"""
    i2count: Mapping[int, int]
    """ mapping from word string to count"""
    counts: Mapping[str, int]

    def __init__(
        self,
        i2s: Mapping[int, str],
        i2count: Mapping[int, int],
        unk: str = None,
        other_special_name2str: Mapping[str, str] = None,
    ) -> Vocab:
        """Create a vocabulary.

        Args:
            i2s (Mapping[int, str]): mapping from word index to string
            i2count (Mapping[int, int]): mapping from word index to count
            unk (str, optional): The `unknown` word. Defaults to None.
            other_special_name2str (Mapping[str, str], optional):
                Special words in addition to `unk`, a mapping from
                name (`eos`, etc.) to word string (e.g., `<eos>`).
                Defaults to None.

        Returns:
            Vocab
        """
    @staticmethod
    def from_json(
        path_to_json, min_count: int = None, max_size: int = None, unk: str = None
    ) -> Vocab:
        """
        Read a Vocab stored in a json file at `path_to_json`

        Parameters
            - min_count: set a new count threshold. All words with smaller
                    counts are truncated, and viewed as {unk}.
            - max_size: set a new vocabulary size limit.
            - unk: set / reset the {unk} token.
        """
    def to_json(path_to_json):
        """Save a Vocab as a JSON file"""
    @staticmethod
    def from_bin(
        path_to_bin, min_count: int = None, max_size: int = None, unk: str = None
    ) -> Vocab:
        """
        Read a Vocab stored in a binary file at `path_to_bin`

        Parameters
            - min_count: set a new count threshold. All words with smaller
                    counts are truncated, and viewed as {unk}.
            - max_size: set a new vocabulary size limit.
            - unk: set / reset the {unk} token.
        """
    def to_bin(path_to_bin):
        """Save a Vocab as a binary file"""
    @staticmethod
    def build(
        path_to_corpus,
        min_count: int = 5,
        max_size: int = None,
        unk: str = "<unk>",
        path_to_save_json: str = None,
        path_to_save_bin: str = None,
    ) -> Vocab:
        """
        Build a Vocab by with a corpus at `path_to_corpus`

        Args:
            - min_count: set a new count threshold. All words with smaller
                    counts are truncated, and viewed as {unk} (or discarded).
            - max_size: set a new vocabulary size limit.
            - unk: set the {unk} token.
            - path_to_save_json: if not specified, will save at
                    ${path_to_corpus}.vocab.json
            - path_to_save_bin: if not specified, will save at
                    ${path_to_corpus}.vocab.bin
        """
    def keys(self) -> List[str]:
        """
        Get the list of words.
        """
    def i2s_dict(self) -> Mapping[int, str]:
        """
        Get a dict from word index to word string.
        Faster than vocab.s2i[..] if there are many inqueries.
        """
    def s2i_dict(self) -> Mapping[str, int]:
        """
        Get a dict from word string to word index.
        Faster than vocab.s2i[..] if there are many inqueries.
        """
    def i2count_dict(self) -> Mapping[int, int]:
        """
        Get a dict from word index to word counts.
        Faster than vocab.i2count[..] if there are many inqueries.
        """
    def counts_dict(self) -> Mapping[str, int]:
        """
        Get a dict from word string to word counts.
        Faster than vocab.counts[..] if there are many inqueries.
        """
    @property
    def unk(self) -> Union[int, None]:
        """
        Get the {unk} word.
        Returns None if {unk} was not set.
        """
    @property
    def unk_id(self) -> Union[int, None]:
        """
        Get the index of the {unk} word.
        Returns None if {unk} was not set.
        """

class SkipGramDataset:
    """
    A iterable dataset provides sampling functionalities
    for SkipGram training algorithm, proposed in:

    - Mikolov, Tomas, et al. "Distributed representations of
    words and phrases and their compositionality." Advances
    in neural information processing systems 26 (2013).
    """

    def __init__(
        self,
        path_to_corpus,
        vocab: Vocab,
        win_size: int = 10,
        sep=" ",
        mode: Literal["shuffle", "onepass", "repeat"] = "shuffle",
        subsample: float = 1e-5,
        power: float = 0.75,
        n_neg: int = 1,
    ):
        """Create a SkipGramDataset from a tokenized corpus file (plain text).

        Args:
            - path_to_corpus (str): the corpus should have been tokenized:
                each line is a document of tokens separated with `sep`
            - vocab (Vocab): a `corpusit.Vocab` object.
            - win_size (int, optional): words occurring in [-win_size, win_size)
                are regarded as the "neighborhood" of the center word.
                Defaults to 10.
            - sep (str, optional): separator of a tokenized document. Defaults to " ".
            - mode (Literal["shuffle";, "onepass";, "repeat"], optional):
                how to reading the corpus.
                    - shuffle: read randomly (with some buffer),
                    - onepass: read from the beginning through the end, one time.
                    - repeat: read from the beginning through the end, and repeat. \\
                Defaults to "shuffle".
            - subsample (float, optional):
                Reduce the probability of sampling frequent words.
                A smaller value indicates a stronger reduction in the probability,
                which also slows down the "positive" sampling process.
                Defaults to 1e-5.
            - power (float, optional):
                A parameter used in negative sampling controlling the probability
                of sampling rare words. Defaults to 0.75.
            - n_neg (int, optional): how many negative samples for one positive sample.
                Defaults to 1.
        """
    def positive_sampler(self, batch_size: int, seed: uint = 0, num_threads: uint = 4):
        """Create an iterable (maybe multi-thread) sampler for generating "positive" samples.
        The sampler returns a numpy 2-d array of shape (batch_size, 2) in each iteration,
        where each row is a pair of word indices.

        Args:
            - batch_size (int): number of pairs in each iteration
            - seed (uint, optional): random seed. Defaults to 0.
            - num_threads (uint, optional): If larger than 1, use multiple threads. Defaults to 4.
        """
    def sampler(self, batch_size: int, seed: uint = 0, num_threads: uint = 4):
        """Create an iterable (maybe multi-thread) sampler for generating "positive" and "negative" samples.
        The sampler returns `(pairs, labels)` in each iteration,
        - `pairs` (numpy 2-d int array): shape (batch_size * (1 + n_neg), 2),
        - `labels` (numpy 1-d bool array): shape (batch_size * (1 + n_neg), )

        Args:
            - batch_size (int): number of "positive" pairs in each iteration
            - seed (uint, optional): random seed. Defaults to 0.
            - num_threads (uint, optional): If larger than 1, use multiple threads. Defaults to 4.
        """

class SkipGramPosIter:
    """
    Returns "positive" SkipGram samples.
    """

class SkipGramIter:
    """
    Return "positive" and "negative" samples, with binary labels indicating whether a sample
    is "positive" or "negative".
    """

class S2I:
    """A Mapping from word string to index."""

    def get(self, s: str, default) -> int:
        pass

class I2S:
    """A Mapping from word index to string."""

    def get(self, i: int, default) -> str:
        pass

class I2Count:
    """A Mapping from word index to count."""

    def get(self, i: int, default) -> int:
        pass

class Counts:
    """A Mapping from word str to count."""

    def get(self, s: str, default) -> int:
        pass
