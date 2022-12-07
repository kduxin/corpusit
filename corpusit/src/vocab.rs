use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
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
                .reduce(|| HashMap::<String, u64>::new(), _merge_counts);
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
    pub fn new(
        i2s: HashMap<usize, String>,
        i2count: HashMap<usize, u64>,
        special_name2str: HashMap<String, String>,
    ) -> Self {
        let s2i: HashMap<String, usize> = i2s.iter().map(|(id, s)| (s.to_string(), *id)).collect();
        let totalcount: u64 = i2count.iter().map(|(_, c)| *c).sum();
        let special_name2i: HashMap<String, usize> = special_name2str
            .iter()
            .map(|(name, s)| {
                (
                    name.to_string(),
                    match s2i.get(s) {
                        Some(id) => *id,
                        None => panic!("Special token `{}` not found in vocabulary.", s),
                    },
                )
            })
            .collect();
        Self {
            i2s: i2s,
            s2i: s2i,
            i2count: i2count,
            totalcount: totalcount,
            special_name2i: special_name2i,
        }
    }

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
        self.s2i.get(s).or_else(|| self.unk_id())
    }

    #[inline]
    pub fn get_str_by_id(&self, id: &usize) -> Option<&str> {
        self.i2s
            .get(id)
            .and_then(|s| Some(s.as_str()))
            .or_else(|| self.unk_str())
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

    pub fn update(&mut self, min_count: Option<u64>, max_size: Option<usize>, unk: Option<&str>) {
        if let Some(unk) = unk {
            self.unk_(&unk);
        }
        let min_count = min_count.unwrap_or(1);
        let max_size = max_size.unwrap_or(usize::MAX);
        self.truncate(min_count, max_size);
    }

    pub fn from_json(
        path_to_json: &str,
        min_count: Option<u64>,
        max_size: Option<usize>,
        unk: Option<&str>,
    ) -> Self {
        let vocab_f = File::open(path_to_json).unwrap();
        let vocab_f = BufReader::new(vocab_f);
        let mut vocab: Self = serde_json::from_reader(vocab_f).unwrap();
        vocab.update(min_count, max_size, unk);
        vocab
    }

    pub fn from_bin(
        path_to_bin: &str,
        min_count: Option<u64>,
        max_size: Option<usize>,
        unk: Option<&str>,
    ) -> Self {
        let vocab_f = File::open(path_to_bin).unwrap();
        let vocab_f = BufReader::new(vocab_f);
        let mut vocab: Vocab = bincode::deserialize_from(vocab_f).unwrap();
        vocab.update(min_count, max_size, unk);
        vocab
    }

    pub fn to_json(&self, path: &str) {
        let vocab_f = File::create(path).unwrap();
        let vocab_f = BufWriter::new(vocab_f);
        serde_json::to_writer_pretty(vocab_f, self).unwrap();
    }

    pub fn to_bin(&self, path: &str) {
        let vocab_f = File::create(path).unwrap();
        let vocab_f = BufWriter::new(vocab_f);
        bincode::serialize_into(vocab_f, self).unwrap();
    }

    pub fn len(&self) -> usize {
        self.i2s.len()
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

#[cfg(test)]
mod tests {
    use crate::vocab::Vocab;
    use std::collections::HashMap;

    #[test]
    fn basics() {
        let mut vocab = Vocab::default();
        let special_tokens: HashMap<String, String> = [("unk", "<unk>")]
            .into_iter()
            .map(|(s1, s2)| (s1.to_string(), s2.to_string()))
            .collect();
        let wordcounts: HashMap<String, u64> =
            [("apple", 100), ("pear", 50), ("bank", 10), ("infreq", 3)]
                .into_iter()
                .map(|(s1, s2)| (s1.to_string(), s2))
                .collect();

        // No unk
        vocab.append_wordcounts(wordcounts);
        assert!(vocab.get_id_by_str("none").is_none());
        assert!(vocab.get_str_by_id(&999).is_none());

        // With unk
        vocab.append_special_tokens(special_tokens);
        assert_eq!(vocab.i2count[&vocab.s2i["apple"]], 100);
        assert_eq!(vocab.i2count[&vocab.s2i["pear"]], 50);
        assert_eq!(vocab.i2count[&vocab.s2i["bank"]], 10);
        assert_eq!(vocab.i2count[&vocab.s2i["infreq"]], 3);

        // Remove `infreq`
        vocab.truncate(5, usize::MAX);
        assert!(!vocab.s2i.contains_key("infreq"));
        assert_eq!(vocab.i2count[&vocab.s2i["<unk>"]], 3);

        // Remove `bank`
        vocab.truncate(1, 3);
        assert!(!vocab.s2i.contains_key("bank"));
        assert!(vocab.get_id_by_str("none").is_some());
        assert_eq!(*vocab.get_id_by_str("bank").unwrap(), vocab.s2i["<unk>"]);
        assert!(vocab.get_str_by_id(&999).is_some());
        assert_eq!(vocab.get_str_by_id(&999).unwrap(), "<unk>");
        assert_eq!(vocab.i2count[&vocab.s2i["<unk>"]], 13);
    }
}
