use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::{Add, AddAssign};
use tqdm::Iter;

type WordCounts = HashMap<String, u64>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VocabBuilder {
    corpus_path: String,
    min_count: u64,
    unk: String,
}

impl Default for VocabBuilder {
    fn default() -> Self {
        Self {
            corpus_path: String::new(),
            min_count: 5,
            unk: "<unk>".to_string(),
        }
    }
}

impl VocabBuilder {
    pub fn new(corpus_path: &str) -> Self {
        Self {
            corpus_path: corpus_path.to_string(),
            ..Default::default()
        }
    }

    #[must_use]
    pub fn infrequent_filtering_(mut self, min_count: u64, unk: &str) -> Self {
        self.min_count = min_count;
        self.unk = unk.to_string();
        self
    }

    pub fn build(self) -> Vocab {
        let f = File::open(self.corpus_path).unwrap();
        let f = BufReader::new(f);

        let mut wordcounts: WordCounts = HashMap::new();
        for large_chunk in f
            .lines()
            .map(Result::unwrap)
            .tqdm()
            .into_iter()
            .chunks(1000000)
            .into_iter()
        {
            let large_chunk: Vec<String> = large_chunk.collect();
            let chunk_counts = large_chunk
                .chunks(10000)
                .par_bridge()
                .map(|chunk| -> HashMap<String, u64> {
                    let mut counts: HashMap<String, u64> = HashMap::new();
                    for line in chunk {
                        for token in line.split(&[' ', '\t']).filter(|x| x.len() > 0) {
                            counts
                                .entry(token.to_string())
                                .and_modify(|e| *e += 1)
                                .or_insert(1);
                        }
                    }
                    counts
                })
                .reduce(
                    || HashMap::<String, u64>::new(),
                    _merge_counts,
                );
            wordcounts = _merge_counts(wordcounts, chunk_counts);
        }

        let wordcounts = {
            let mut freqwordcounts: HashMap<String, u64> = HashMap::new();
            for (w, c) in wordcounts.iter() {
                let w = match *c >= self.min_count {
                    true => w,
                    false => &self.unk,
                };
                freqwordcounts
                    .entry(w.to_string())
                    .and_modify(|e| *e += *c)
                    .or_insert(*c);
            }
            freqwordcounts
        };

        let mut vocab = Vocab::default();
        vocab.append_wordcounts(wordcounts);
        vocab
    }
}

#[derive(Default)]
pub struct VocabReader {
    vocab_path: String,
    special_tokens: HashMap<String, String>,
}

impl VocabReader {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn vocab_path_(mut self, vocab_path: String) -> Self {
        self.vocab_path = vocab_path;
        self
    }

    pub fn read(self) -> Vocab {
        let f = File::open(self.vocab_path).unwrap();
        let reader = BufReader::new(f);
        let mut wordcounts: HashMap<String, u64> = HashMap::new();
        for line in reader.lines() {
            let segs = line
                .unwrap()
                .split(&[' ', '\t'])
                .filter(|word| word.len() > 0)
                .map(|x| x.to_string())
                .collect::<Vec<String>>();
            let w = &segs[0];
            let count = &segs[1].parse::<u64>().unwrap();
            wordcounts.insert(w.to_string(), *count);
        }

        let mut vocab = Vocab::default();
        vocab.append_special_tokens(self.special_tokens);
        vocab.append_wordcounts(wordcounts);
        vocab
    }
}

#[derive(Default, Debug, Clone, Deserialize, Serialize)]
pub struct Vocab {
    pub i2s: HashMap<usize, String>,
    pub s2i: HashMap<String, usize>,
    pub i2count: HashMap<usize, u64>,
    pub totalcount: u64,
    pub special_name2i: HashMap<String, usize>,
}

impl Vocab {
    pub fn unk(&self) -> Option<(&str, usize)> {
        self.special_name2i
            .get("unk")
            .and_then(|id| Some((self.i2s.get(id).unwrap().as_str(), *id)))
    }
    pub fn unk_str(&self) -> Option<&str> {
        self.special_name2i
            .get("unk")
            .and_then(|id| Some(self.i2s.get(id).unwrap().as_ref()))
    }
    pub fn unk_id(&self) -> Option<&usize> {
        self.special_name2i.get("unk")
    }

    pub fn unk_(&mut self, unk_str: &str) {
        let id = match self.s2i.get(unk_str) {
            Some(id) => *id,
            None => {
                let mut i = 0;
                while self.i2s.contains_key(&i) {
                    i += 1;
                }
                self.i2s.insert(i, unk_str.to_string());
                self.s2i.insert(unk_str.to_string(), i);
                self.i2count.insert(i, 0);
                i
            }
        };
        self.special_name2i
            .entry("unk".to_string())
            .and_modify(|i| *i = id)
            .or_insert(id);
    }

    #[inline]
    pub fn get_id_by_str(&self, s: &str) -> Option<&usize> {
        self.s2i.get(s)
    }

    #[inline]
    pub fn get_str_by_id(&self, id: &usize) -> Option<&str> {
        self.i2s.get(id).and_then(|s| Some(s.as_str()))
    }

    fn append_special_tokens(&mut self, special_tokens: HashMap<String, String>) {
        let counts: WordCounts = special_tokens
            .iter()
            .map(|(_, s)| (s.to_string(), 0))
            .collect();
        self.append_wordcounts(counts);
        special_tokens.into_iter().for_each(|(name, s)| {
            self.special_name2i
                .entry(name)
                .and_modify(|i| *i = self.s2i[&s])
                .or_insert(self.s2i[&s]);
        });
    }

    fn append_wordcounts(&mut self, wordcounts: WordCounts) {
        let mut id = 0;
        let mut sorted_wordcounts: Vec<(String, u64)> = wordcounts.clone().into_iter().collect();
        sorted_wordcounts.sort_by(|(_, c1), (_, c2)| (u64::MAX - *c1).cmp(&(u64::MAX - *c2)));
        for (word, count) in sorted_wordcounts.iter() {
            match self.s2i.get(word) {
                Some(i) => {
                    self.i2count.entry(*i).and_modify(|c| *c += count);
                    continue;
                }
                None => {
                    while self.i2s.contains_key(&id) {
                        id += 1;
                    }
                    self.i2s.insert(id.clone(), word.to_string());
                    self.s2i.insert(word.to_string(), id.clone());
                    self.i2count.insert(id.clone(), count.clone());
                    self.totalcount += count;
                }
            }

            if self.s2i.contains_key(word) {
                let id = self.s2i[word];
                self.i2count.entry(id).and_modify(|c| *c += count);
                continue;
            }
        }
    }

    pub fn truncate(&mut self, min_count: u64, max_size: usize) {

        let mut trun_i2count: HashMap<usize, u64> = HashMap::new();
        self.special_name2i.iter().for_each(|(_, id)| {
            trun_i2count.insert(*id, self.i2count[id]);
        });

        let special_ids: HashSet<usize> = self.special_name2i.values().map(|i| *i).collect();
        let unk_id = self.unk_id();

        let mut sorted_i2count: Vec<(usize, u64)> = self.i2count.clone().into_iter().collect();
        sorted_i2count.sort_by(|(_, c1), (_, c2)| (u64::MAX - *c1).cmp(&(u64::MAX - *c2)));
        for (id, c) in sorted_i2count.iter() {

            if special_ids.contains(id) {
                continue;
            }

            if *c >= min_count && trun_i2count.len() < max_size {
                trun_i2count.insert(*id, *c);
            } else if let Some(unk_id) = unk_id {
                trun_i2count.entry(*unk_id).and_modify(|e| *e += *c);
            }
        }

        self.i2count = trun_i2count.iter().map(|(id, c)| (*id, *c)).collect();
        self.i2s = trun_i2count
            .iter()
            .map(|(id, _)| (*id, self.i2s[id].to_string()))
            .collect();
        self.s2i = trun_i2count
            .iter()
            .map(|(id, _)| (self.i2s[id].to_string(), *id))
            .collect();
    }
}

fn _merge_counts<T>(mut c1: HashMap<String, T>, c2: HashMap<String, T>) -> HashMap<String, T>
where
    T: Add + AddAssign + Copy,
{
    c2.iter().for_each(|(key, count)| {
        c1.entry(key.to_string())
            .and_modify(|e| *e += *count)
            .or_insert(*count);
    });
    c1
}