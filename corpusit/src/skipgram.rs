use itertools::Itertools;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::{
    collections::HashMap,
    collections::HashSet,
    sync::{Arc, RwLock},
    thread,
};

pub type WordIdxPair = [usize; 2];
type Sep = String;

use crate::fileread::{get_file_reader, FileIterateMode, ResetableFileReader};
use crate::vocab::{Vocab, VocabBuilder};

// ----------------------- ReadBuffer ------------------------

type CategoricalSampler = CategoricalSamplerBSearch;

pub struct ReadBuffer {
    reader: Box<dyn ResetableFileReader + Send>,
    buffer: Vec<WordIdxPair>,
    buffer_progress: usize,
    rng: ChaCha20Rng,
}

impl ReadBuffer {
    fn reset(&mut self) {
        self.reader.reset();
        self.buffer.clear();
        self.buffer_progress = 0;
        self.rng = ChaCha20Rng::from_seed(self.rng.get_seed());
    }

    pub fn seed(&mut self, seed: u64) {
        self.rng = ChaCha20Rng::seed_from_u64(seed);
    }
}

// ----------------------- SGProbs ------------------------

#[derive(Debug)]
pub struct SGProbs {
    id2posprobs: HashMap<usize, f64>,
    id2negprobs: HashMap<usize, f64>,
}

#[derive(Clone)]
pub struct CategoricalSamplerBSearch {
    ids: Arc<Vec<usize>>,
    cumprobs: Arc<Vec<f64>>,
    num_threads: usize,
    rng: ChaCha20Rng,
    buffer: Vec<usize>,
}

/// Binary search
impl CategoricalSamplerBSearch {
    fn new(id2probs: &HashMap<usize, f64>, rng: &ChaCha20Rng, num_threads: usize) -> Self {
        assert!(!id2probs.is_empty());
        let mut ids: Vec<usize> = Vec::new();
        let mut cumprobs: Vec<f64> = Vec::new();
        let mut totalprob = 0.;
        id2probs.iter().for_each(|(id, p)| {
            ids.push(*id);
            totalprob += *p;
            cumprobs.push(totalprob)
        });
        if (totalprob - 1.0).abs() > 1e-5 {
            cumprobs = cumprobs.into_iter().map(|p| p / totalprob).collect();
        }
        *cumprobs.last_mut().unwrap() = 1.0f64;

        Self {
            ids: Arc::new(ids),
            cumprobs: Arc::new(cumprobs),
            num_threads: num_threads,
            rng: rng.clone(),
            buffer: Vec::with_capacity(1000000),
        }
    }

    pub fn sample(&mut self) -> usize {
        self.buffer.pop().unwrap_or_else(|| {
            self.update_buffer_parallel_deterministic();
            self.buffer.pop().unwrap()
        })
    }

    pub fn sample_n(&mut self, n: usize) -> Vec<usize> {
        let mut res: Vec<usize> = vec![];
        while res.len() < n {
            let remained = n - res.len();
            let at = if self.buffer.len() > remained {
                self.buffer.len() - remained
            } else {
                0
            };
            res.append(&mut self.buffer.split_off(at));
            if at == 0 {
                self.update_buffer_parallel_deterministic();
            }
        }
        res
    }

    fn update_buffer_parallel_deterministic(&mut self) {
        self.buffer.clear();
        let n_per_job = 1000000 / self.num_threads + 1;
        thread::scope(|s| {
            let mut handlers = Vec::new();

            for _ in 0..self.num_threads {
                let ids = Arc::clone(&self.ids);
                let cumprobs = Arc::clone(&self.cumprobs);
                let mut rng = ChaCha20Rng::seed_from_u64(self.rng.gen::<u64>());
                let handler = s.spawn(move || {
                    let mut res = Vec::new();
                    for _ in 0..n_per_job {
                        res.push(Self::_sample(&ids, &cumprobs, &mut rng));
                    }
                    res
                });
                handlers.push(handler);
            }

            for handler in handlers.into_iter() {
                self.buffer.append(&mut handler.join().unwrap());
            }

        });
    }

    pub fn _sample(ids: &Arc<Vec<usize>>, cumprobs: &Arc<Vec<f64>>, rng: &mut ChaCha20Rng) -> usize {
        let p = rng.gen_range(0.0..1.0);
        let mut left = 0;
        let mut right = cumprobs.len();
        while left < right {
            let mid = left + (right - left) / 2;
            if p <= cumprobs[mid] {
                right = mid;
            } else {
                left = mid + 1
            }
        }
        left = left.min(cumprobs.len() - 1);
        ids[left]
    }

}

#[derive(Clone)]
pub struct CategoricalSamplerTable {
    table: Vec<usize>,
    rng: ChaCha20Rng,
}

static TABLE_SIZE: usize = 10000000;

impl CategoricalSamplerTable {
    fn new(id2probs: &HashMap<usize, f64>, rng: &ChaCha20Rng, table_size: Option<usize>) -> Self {
        assert!(!id2probs.is_empty());

        let table_size = table_size.unwrap_or(TABLE_SIZE);
        let mut table = vec![0usize; table_size];

        let totalprob: f64 = id2probs.iter().map(|(_, p)| *p).sum();
        let id2probs = id2probs
            .into_iter()
            .map(|(x, y)| (*x, *y / totalprob))
            .collect_vec();
        let mut cumprob: f64 = 0.;
        let mut j: usize = 0;
        for i in 0..id2probs.len() {
            cumprob += id2probs[i].1;
            while ((j as f64) / (table_size as f64) < cumprob) && (j < table_size) {
                table[j] = id2probs[i].0;
                j += 1;
            }
        }

        let uniq: HashSet<usize> = table.iter().map(|x| *x).collect();
        assert_eq!(uniq.len(), id2probs.len());

        Self {
            table: table,
            rng: rng.clone(),
        }
    }

    pub fn sample(&mut self) -> usize {
        let i = self.rng.gen::<usize>() % self.table.len();
        self.table[i]
    }

    pub fn sample_n(&mut self, n: usize) -> Vec<usize> {
        let mut samples: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..n {
            samples.push(self.sample());
        }
        samples
    }
}

impl SGProbs {
    fn new(vocab: &Vocab, config: &SGDatasetConfig) -> Self {
        let mut ids: Vec<usize> = Vec::new();
        let mut probs: Vec<f64> = Vec::new();
        vocab.i2count.iter().for_each(|(id, count)| {
            ids.push(*id);
            probs.push((*count as f64) / (vocab.totalcount as f64));
        });
        let posprobs = Self::adjust_probs_by_subsample(&probs, config.subsample);
        let negprobs = Self::adjust_probs_by_power(&probs, config.power);
        Self {
            id2posprobs: ids
                .iter()
                .zip(posprobs)
                .map(|(id, prob)| (*id, prob))
                .collect(),
            id2negprobs: ids
                .iter()
                .zip(negprobs)
                .map(|(id, prob)| (*id, prob))
                .collect(),
        }
    }

    fn adjust_probs_by_subsample(probs: &Vec<f64>, subsample: f64) -> Vec<f64> {
        probs
            .iter()
            .map(|prob| -> f64 {
                (((prob / subsample).powf(0.5) + 1.) * (subsample / prob)).min(1.)
            })
            .collect_vec()
    }

    fn adjust_probs_by_power(probs: &Vec<f64>, power: f64) -> Vec<f64> {
        let probs_power = probs
            .iter()
            .map(|prob| -> f64 { prob.powf(power) })
            .collect_vec();
        let total: f64 = probs_power.iter().sum();
        probs_power
            .iter()
            .map(|prob| -> f64 { *prob / total })
            .collect()
    }
}

// ----------------------- SGDatasetConfig ------------------------

#[derive(Clone)]
pub struct SGDatasetConfig {
    pub corpus_path: String,
    pub win_size: usize,
    pub sep: Sep,
    pub mode: FileIterateMode,
    pub subsample: f64,
    pub power: f64,
    pub n_neg: usize,
}

impl Default for SGDatasetConfig {
    fn default() -> Self {
        Self {
            corpus_path: "".to_string(),
            win_size: 5,
            sep: " ".to_string(),
            mode: FileIterateMode::default(),
            subsample: 1e-3,
            power: 0.75,
            n_neg: 1,
        }
    }
}

impl SGDatasetConfig {
    pub fn new(corpus_path: &str) -> Self {
        Self {
            corpus_path: corpus_path.to_string(),
            ..Self::default()
        }
    }
}

// ----------------------- SGDatasetBuilder ------------------------

#[derive(Default)]
pub struct SGDatasetBuilder {
    config: SGDatasetConfig,
}

impl SGDatasetBuilder {
    pub fn new(config: SGDatasetConfig) -> SGDatasetBuilder {
        SGDatasetBuilder { config: config }
    }

    #[must_use]
    pub fn corpus_path_(mut self, corpus_path: &str) -> Self {
        self.config.corpus_path = corpus_path.to_string();
        self
    }

    #[must_use]
    pub fn win_size_(mut self, win_size: usize) -> Self {
        self.config.win_size = win_size;
        self
    }

    pub fn sep_<P>(mut self, sep: Sep) -> Self {
        self.config.sep = sep;
        self
    }

    pub fn mode_(mut self, mode: FileIterateMode) -> Self {
        self.config.mode = mode;
        self
    }

    pub fn build_with_vocab(&self, vocab: &Vocab) -> SGDataset {
        SGDataset {
            config: Arc::new(RwLock::new(self.config.clone())),
            vocab: Arc::new(RwLock::new(vocab.clone())),
            probs: Arc::new(RwLock::new(SGProbs::new(vocab, &self.config))),
        }
    }

    pub fn build(&self) -> SGDataset {
        let vocab_builder = VocabBuilder::new(&self.config.corpus_path);
        let vocab = vocab_builder.build();
        let sgprobs = SGProbs::new(&vocab, &self.config);
        SGDataset {
            config: Arc::new(RwLock::new(self.config.clone())),
            vocab: Arc::new(RwLock::new(vocab)),
            probs: Arc::new(RwLock::new(sgprobs)),
        }
    }
}

// ----------------------- SGDataset ------------------------

pub struct SGDataset {
    pub config: Arc<RwLock<SGDatasetConfig>>,
    pub vocab: Arc<RwLock<Vocab>>,
    pub probs: Arc<RwLock<SGProbs>>,
}

impl SGDataset {
    pub fn new(config: SGDatasetConfig) -> Arc<Self> {
        Arc::new(SGDatasetBuilder::new(config).build())
    }

    pub fn new_with_vocab(config: SGDatasetConfig, vocab: &Vocab) -> Arc<Self> {
        Arc::new(SGDatasetBuilder::new(config).build_with_vocab(vocab))
    }

    pub fn positive_sampler(&self, seed: u64, num_threads: usize) -> SGDatasetPosIter {
        let status = ReadBuffer {
            reader: get_file_reader(
                &self.config.read().unwrap().corpus_path,
                self.config.read().unwrap().mode.clone(),
            ),
            buffer: Vec::new(),
            buffer_progress: 0,
            rng: ChaCha20Rng::seed_from_u64(seed),
        };
        SGDatasetPosIter {
            dataset: SGDataset {
                config: Arc::clone(&self.config),
                vocab: Arc::clone(&self.vocab),
                probs: Arc::clone(&self.probs),
            },
            status: status,
            num_threads: num_threads,
        }
    }

    pub fn sampler(
        &self,
        seed: u64,
        num_threads: usize,
    ) -> SGDatasetIter {
        let rng = ChaCha20Rng::seed_from_u64(seed);
        SGDatasetIter {
            dataset: SGDataset {
                config: Arc::clone(&self.config),
                vocab: Arc::clone(&self.vocab),
                probs: Arc::clone(&self.probs),
            },
            posit: self.positive_sampler(seed, num_threads),
            sampler: CategoricalSampler::new(
                &self.probs.read().unwrap().id2negprobs,
                &rng,
                num_threads,
            ),
        }
    }
}

// ----------------------- SGDataset Pos Iter ------------------------
pub struct SGDatasetPosIter {
    dataset: SGDataset,
    status: ReadBuffer,
    num_threads: usize,
}

impl Iterator for SGDatasetPosIter {
    type Item = WordIdxPair;
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = self._next_batch(1);
        batch.pop()
    }
}

impl SGDatasetPosIter {
    pub fn reset(&mut self) {
        self.status.reset();
    }

    pub fn next_batch(&mut self, max_batch_size: usize) -> Option<Vec<WordIdxPair>> {
        let batch = self._next_batch(max_batch_size);
        if batch.len() > 0 {
            Some(batch)
        } else {
            None
        }
    }

    fn _next_batch(&mut self, max_batch_size: usize) -> Vec<WordIdxPair> {
        let mut res: Vec<WordIdxPair> = Vec::with_capacity(max_batch_size);
        let mut progress = 0;
        while progress < max_batch_size {
            let start = self.status.buffer_progress;
            let end = (start + max_batch_size - progress).min(self.status.buffer.len());
            res.extend_from_slice(&self.status.buffer[start..end]);
            progress += end - start;
            self.status.buffer_progress += end - start;
            if end == self.status.buffer.len() {
                let res = if self.num_threads > 1 {
                    self.update_buffer_parallel_deterministic()
                } else {
                    self.update_buffer()
                };
                if let Err(_) = res {
                    break;
                }
            }
        }
        res
    }

    fn update_buffer(&mut self) -> Result<usize, ()> {
        self.status.buffer_progress = 0;
        self.status.buffer.clear();

        let lines: Vec<Option<String>> = (0..100)
            .into_iter()
            .map(|_| self.status.reader.next())
            .collect_vec();
        if lines.len() == 0 {
            return Err(());
        }

        let mut n = 0;
        let config = &self.dataset.config.read().unwrap();
        let id2posprobs = &self.dataset.probs.read().unwrap().id2posprobs;
        lines.iter().for_each(|line: &Option<String>| {
            if let Some(line) = line {
                let mut idpairs = line_to_idpairs(
                    line,
                    &self.dataset.vocab.read().unwrap(),
                    config,
                    id2posprobs,
                    &mut self.status.rng,
                );
                n += idpairs.len();
                self.status.buffer.append(&mut idpairs);
            }
        });
        Ok(n)
    }

    fn update_buffer_parallel_deterministic(&mut self) -> Result<usize, ()> {
        self.status.buffer_progress = 0;
        self.status.buffer.clear();

        let lines: Vec<String> = (0..10000)
            .into_iter()
            .filter_map(|_| self.status.reader.next())
            .collect_vec();
        if lines.len() == 0 {
            return Err(());
        }

        let jobs = lines
            .chunks(lines.len() / self.num_threads + 1)
            .collect_vec();
        let mut n = 0;
        thread::scope(|s| {
            let mut handlers = Vec::new();
            for job in &jobs {
                let mut rng = ChaCha20Rng::seed_from_u64(self.status.rng.gen::<u64>());
                let vocab = Arc::clone(&self.dataset.vocab);
                let config = Arc::clone(&self.dataset.config);
                let probs = Arc::clone(&self.dataset.probs);
                let handler = s.spawn(move || {
                    job.iter()
                        .map(|line| -> Vec<WordIdxPair> {
                            line_to_idpairs(
                                line,
                                &vocab.read().unwrap(),
                                &config.read().unwrap(),
                                &probs.read().unwrap().id2posprobs,
                                &mut rng,
                            )
                        })
                        .flatten()
                        .collect_vec()
                });
                handlers.push(handler);
            }
            for handler in handlers.into_iter() {
                let mut pairs = handler.join().unwrap();
                n += pairs.len();
                self.status.buffer.append(&mut pairs);
            }
        });

        Ok(n)
    }
}

fn line_to_idpairs(
    line: &str,
    vocab: &Vocab,
    config: &SGDatasetConfig,
    id2prob: &HashMap<usize, f64>,
    rng: &mut ChaCha20Rng,
) -> Vec<WordIdxPair> {
    let ids = line
        .split(&config.sep)
        .into_iter()
        .filter_map(|word| match word.len() > 0 {
            true => vocab.get_id_by_str(word),
            false => None,
        })
        .map(|id| *id)
        .collect_vec();

    let len = ids.len();
    let mut pairs: Vec<WordIdxPair> = Vec::with_capacity(len * config.win_size * 2);
    for cen in 0..len {
        let idcen = ids[cen];
        let pcen = id2prob[&idcen];
        if pcen >= rng.gen_range(0.0..1.0) {
            let left = match cen >= config.win_size {
                true => cen - config.win_size,
                false => 0,
            };
            let right = len.min(cen + config.win_size);
            for ctx in left..right {
                let idctx = ids[ctx];
                let pctx = id2prob[&idctx];
                if pctx > rng.gen_range(0.0..1.0) {
                    if cen != ctx {
                        let pair = [idcen, idctx];
                        pairs.push(pair);
                    }
                }
            }
        }
    }
    pairs
}

// ----------------------- SGDataset Pos Neg Iter ------------------------
pub struct SGDatasetIter {
    dataset: SGDataset,
    posit: SGDatasetPosIter,
    sampler: CategoricalSampler,
}

impl Iterator for SGDatasetIter {
    type Item = (Vec<WordIdxPair>, Vec<bool>);
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = self.posit._next_batch(1);
        batch.pop().and_then(|pair| {
            let mut pairs: Vec<WordIdxPair> = vec![pair];
            let mut labels: Vec<bool> = vec![true];
            let cen = pair[0];
            for _ in 0..self.dataset.config.read().unwrap().n_neg {
                pairs.push([cen, self.sampler.sample()]);
                labels.push(false);
            }
            Some((pairs, labels))
        })
    }
}

impl SGDatasetIter {
    pub fn next_batch(&mut self, max_batch_size: usize) -> Option<(Vec<WordIdxPair>, Vec<bool>)> {
        // let batch = self.posit._next_batch(max_batch_size);

        let posbatch = self.posit._next_batch(max_batch_size);

        let n_neg = self.dataset.config.read().unwrap().n_neg;
        let negbatch = match posbatch.len() > 0 {
            true => self.sampler.sample_n(max_batch_size * n_neg + 1),
            false => vec![],
        };

        match posbatch.is_empty() {
            true => None,
            false => {
                let mut pairs: Vec<WordIdxPair> = Vec::with_capacity(max_batch_size * (1 + n_neg));
                let mut labels: Vec<bool> = Vec::with_capacity(max_batch_size * (1 + n_neg));
                match n_neg > 0 {
                    true => {
                        for (cen, negs) in posbatch.iter().zip(negbatch.chunks(n_neg)) {
                            pairs.push(*cen);
                            labels.push(true);
                            negs.iter().for_each(|neg| {
                                pairs.push([cen[0], *neg]);
                                labels.push(false);
                            })
                        }
                    }
                    false => {
                        pairs = posbatch;
                        labels = vec![true; max_batch_size];
                    }
                }
                Some((pairs, labels))
            }
        }
    }
}

// ----------------------- test ------------------------

#[cfg(test)]
mod tests {
    use crate::skipgram::{
        CategoricalSampler, FileIterateMode, SGDataset, SGDatasetConfig, VocabBuilder,
    };
    use crate::vocab::Vocab;
    use bincode;
    use itertools::Itertools;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use serde_json;
    use std::{collections::HashMap, fs, io, path, sync::Arc};
    use tqdm::Iter;

    #[test]
    fn caregorical_sample() {
        let id2probs: HashMap<usize, f64> = vec![(1, 1.0), (10, 0.6), (100, 0.398), (1000, 0.002)]
            .into_iter()
            .collect();
        let rng = ChaCha20Rng::seed_from_u64(0);
        let mut sampler = CategoricalSampler::new(&id2probs, &rng, 4);
        let n = 10000000;
        let mut samples = Vec::with_capacity(n);
        for _ in (0..n).tqdm() {
            samples.push(sampler.sample());
        }

        let i2counts = {
            let mut i2counts: HashMap<usize, f64> = HashMap::new();
            samples.iter().for_each(|id| {
                i2counts.entry(*id).and_modify(|e| *e += 1.).or_insert(1.);
            });
            i2counts
        };
        println!(
            "distribution of samples {:?}",
            i2counts
                .iter()
                .map(|(id, c)| (*id, *c / (n as f64)))
                .collect_vec()
        );

        assert!((i2counts[&1] / (n as f64) - 0.5).abs() < 0.01);
        assert!((i2counts[&10] / (n as f64) - 0.3).abs() < 0.01);
        assert!((i2counts[&100] / (n as f64) - 0.199).abs() < 0.01);
        assert!((i2counts[&1000] / (n as f64) - 0.001).abs() < 0.001);
    }

    #[test]
    fn chachaseed() {
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let seed1 = rng.get_seed();
        let samples1: Vec<f64> = (0..10).map(|_| rng.gen_range(0.0..1.0)).collect();
        let seed2 = rng.get_seed();
        rng = ChaCha20Rng::from_seed(rng.get_seed());
        let samples2: Vec<f64> = (0..10).map(|_| rng.gen_range(0.0..1.0)).collect();

        println!("seed1 = {:?}, seed2 = {:?}", seed1, seed2);
        println!("samples1 = {:?}, samples2 = {:?}", samples1, samples2);
        assert_eq!(seed1, seed2);
        assert_eq!(samples1, samples2);
    }

    fn load_vocab_and_dataset(mode: FileIterateMode) -> (Vocab, Arc<SGDataset>) {
        let mut path = path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("data/corpus.txt");
        let path: String = path.to_string_lossy().into();

        let vocab_path = path.clone() + ".vocab";
        let found: Option<&str> = match path::Path::new(&(vocab_path.clone() + ".bin")).is_file() {
            true => Some("binary"),
            false => match path::Path::new(&(vocab_path.clone() + ".json")).is_file() {
                true => Some("json"),
                false => None,
            },
        };

        let mut vocab = match found {
            Some("binary") => {
                println!("Found saved vocabulary. Loading from it.");
                let vocab_f = fs::File::open(vocab_path.clone() + ".bin").unwrap();
                let vocab_f = io::BufReader::new(vocab_f);
                bincode::deserialize_from(vocab_f).unwrap()
            }
            Some("json") => {
                println!("Found saved vocabulary. Loading from it.");
                let vocab_f = fs::File::open(vocab_path).unwrap();
                let vocab_f = io::BufReader::new(vocab_f);
                serde_json::from_reader(vocab_f).unwrap()
            }
            Some(_) => {
                panic!("What have you found ?");
            }
            None => {
                let vocab = VocabBuilder::new(&path).build();

                let vocab_f = fs::File::create(vocab_path.clone() + ".json").unwrap();
                let vocab_f = io::BufWriter::new(vocab_f);
                serde_json::to_writer_pretty(vocab_f, &vocab).unwrap();

                let vocab_f = fs::File::create(vocab_path.clone() + ".bin").unwrap();
                let vocab_f = io::BufWriter::new(vocab_f);
                bincode::serialize_into(vocab_f, &vocab).unwrap();

                vocab
            }
        };
        vocab.unk_("<unk>");
        vocab.truncate(1, usize::MAX);
        println!("Vocab size: {}", vocab.s2i.len());
        println!("Vocab special tokens: {:?}", vocab.special_name2i);

        let mut config = SGDatasetConfig::new(&path);
        config.subsample = 1e-3;
        config.mode = mode;
        let dataset = SGDataset::new_with_vocab(config, &vocab);
        (vocab, dataset)
    }

    #[test]
    fn repeat() {
        let (vocab, dataset) = load_vocab_and_dataset(FileIterateMode::SHUFFLE);
        let mut iterator = dataset.positive_sampler(0, 4);
        for i in 0..1000 {
            let pairs = iterator.next_batch(10000);
            if i % 100 == 0 {
                if let Some(pairs) = pairs {
                    let pair = pairs[0].to_owned();
                    println!(
                        "Iter {:0>4}: {} ({}) {} ({}). Cache size: {}",
                        i,
                        pair[0],
                        vocab.get_str_by_id(&pair[0]).unwrap(),
                        pair[1],
                        vocab.get_str_by_id(&pair[1]).unwrap(),
                        iterator.status.buffer.len()
                    );
                }
            }
        }
    }

    #[test]
    fn is_deterministic_onepass() {
        let (_, dataset) = load_vocab_and_dataset(FileIterateMode::ONEPASS);
        let mut it1 = dataset.positive_sampler(0, 4);
        let mut it2 = dataset.positive_sampler(0, 4);
        let result1 = (0..1000)
            .tqdm()
            .filter_map(|_| it1.next_batch(1000))
            .collect_vec();
        let result2 = (0..1000)
            .tqdm()
            .filter_map(|_| it2.next_batch(1000))
            .collect_vec();
        assert_eq!(result1, result2);
        println!(
            "Number of pairs: {}",
            result1.iter().flatten().collect_vec().len()
        );
    }

    #[test]
    fn is_deterministic_shuffle() {
        let (_, dataset) = load_vocab_and_dataset(FileIterateMode::SHUFFLE);
        let mut it1 = dataset.positive_sampler(0, 4);
        let mut it2 = dataset.positive_sampler(0, 4);
        let result1 = (0..1000)
            .tqdm()
            .filter_map(|_| it1.next_batch(1000))
            .collect_vec();
        let result2 = (0..1000)
            .tqdm()
            .filter_map(|_| it2.next_batch(1000))
            .collect_vec();
        assert_eq!(result1, result2);
        let length = result1.iter().flatten().collect_vec().len();
        println!("Number of pairs: {}", &length);
        assert_eq!(length, 1000 * 1000);
    }

    #[test]
    fn is_deterministic_repeat() {
        let (_, dataset) = load_vocab_and_dataset(FileIterateMode::REPEAT);
        let mut it1 = dataset.positive_sampler(0, 4);
        let mut it2 = dataset.positive_sampler(0, 4);
        let result1 = (0..1000)
            .tqdm()
            .filter_map(|_| it1.next_batch(1000))
            .collect_vec();
        let result2 = (0..1000)
            .tqdm()
            .filter_map(|_| it2.next_batch(1000))
            .collect_vec();
        assert_eq!(result1, result2);
        let length = result1.iter().flatten().collect_vec().len();
        println!("Number of pairs: {}", &length);
        assert_eq!(length, 1000 * 1000);
    }
}
