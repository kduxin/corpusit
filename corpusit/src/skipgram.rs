use itertools::Itertools;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::{
    collections::HashMap,
    collections::HashSet,
    sync::Arc,
    thread,
};

pub type WordIdxPair = [usize; 2];

type CategoricalSampler = CategoricalSamplerBSearch;

// ----------------------- SGProbs ------------------------

#[derive(Debug, Clone)]
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


// ----------------------- TokenizedSkipGram ------------------------


/// Skip-gram configuration for pre-tokenized sequences
#[derive(Clone)]
pub struct SGConfig {
    pub win_size: usize,
    pub subsample: f64,
    pub power: f64,
    pub n_neg: usize,
    pub word_counts: HashMap<usize, u64>,
    pub total_count: u64,
    pub probs: SGProbs,
}

impl SGConfig {
    pub fn new(word_counts: HashMap<usize, u64>) -> Self {
        let total_count = word_counts.values().sum();
        let mut config = Self {
            win_size: 5,
            subsample: 1e-3,
            power: 0.75,
            n_neg: 1,
            word_counts,
            total_count,
            probs: SGProbs {
                id2posprobs: HashMap::new(),
                id2negprobs: HashMap::new(),
            },
        };
        config.probs = SGProbs::new_from_counts(&config.word_counts, &config);
        config
    }

    pub fn with_win_size(mut self, win_size: usize) -> Self {
        self.win_size = win_size;
        self
    }

    pub fn with_subsample(mut self, subsample: f64) -> Self {
        self.subsample = subsample;
        self.probs = SGProbs::new_from_counts(&self.word_counts, &self);
        self
    }

    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self.probs = SGProbs::new_from_counts(&self.word_counts, &self);
        self
    }

    pub fn with_n_neg(mut self, n_neg: usize) -> Self {
        self.n_neg = n_neg;
        self
    }
}

/// Skip-gram configuration with tokenization support
#[derive(Clone)]
pub struct SGConfigWithTokenization {
    pub base_config: SGConfig,
    pub separator: String,
    pub word_to_id: HashMap<String, usize>,
}

impl SGConfigWithTokenization {
    pub fn new(
        word_counts: HashMap<usize, u64>,
        word_to_id: HashMap<String, usize>,
        separator: String,
    ) -> Self {
        let base_config = SGConfig::new(word_counts);
        Self {
            base_config,
            separator,
            word_to_id,
        }
    }

    pub fn with_win_size(mut self, win_size: usize) -> Self {
        self.base_config = self.base_config.with_win_size(win_size);
        self
    }

    pub fn with_subsample(mut self, subsample: f64) -> Self {
        self.base_config = self.base_config.with_subsample(subsample);
        self
    }

    pub fn with_power(mut self, power: f64) -> Self {
        self.base_config = self.base_config.with_power(power);
        self
    }

    pub fn with_n_neg(mut self, n_neg: usize) -> Self {
        self.base_config = self.base_config.with_n_neg(n_neg);
        self
    }
}

/// Iterator for positive samples from pre-tokenized sequences
pub struct SGPosIter {
    config: SGConfig,
    rng: ChaCha20Rng,
}

impl SGPosIter {
    pub fn new(config: SGConfig, seed: u64) -> Self {
        Self {
            config,
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    pub fn process_sequence(&mut self, sequence: &[usize]) -> Vec<WordIdxPair> {
        let id2prob = &self.config.probs.id2posprobs;
        
        let len = sequence.len();
        let mut pairs: Vec<WordIdxPair> = Vec::with_capacity(len * self.config.win_size * 2);
        
        for cen in 0..len {
            let idcen = sequence[cen];
            let pcen = id2prob.get(&idcen).copied().unwrap_or(0.0);
            if pcen >= self.rng.gen_range(0.0..1.0) {
                let left = match cen >= self.config.win_size {
                    true => cen - self.config.win_size,
                    false => 0,
                };
                let right = len.min(cen + self.config.win_size + 1);
                for ctx in left..right {
                    let idctx = sequence[ctx];
                    let pctx = id2prob.get(&idctx).copied().unwrap_or(0.0);
                    if pctx > self.rng.gen_range(0.0..1.0) {
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

    pub fn process_sequences(&mut self, sequences: &[Vec<usize>]) -> Vec<WordIdxPair> {
        let mut all_pairs = Vec::new();
        for sequence in sequences {
            let pairs = self.process_sequence(sequence);
            all_pairs.extend(pairs);
        }
        all_pairs
    }
}

/// Iterator for positive and negative samples from pre-tokenized sequences
pub struct SGIter {
    config: SGConfig,
    pos_iter: SGPosIter,
    sampler: CategoricalSampler,
}

impl SGIter {
    pub fn new(config: SGConfig, seed: u64, num_threads: usize) -> Self {
        let rng = ChaCha20Rng::seed_from_u64(seed);
        let pos_iter = SGPosIter::new(config.clone(), seed);
        let sampler = CategoricalSampler::new(&config.probs.id2negprobs, &rng, num_threads);
        
        Self {
            config,
            pos_iter,
            sampler,
        }
    }

    pub fn process_sequence(&mut self, sequence: &[usize]) -> (Vec<WordIdxPair>, Vec<bool>) {
        let pos_pairs = self.pos_iter.process_sequence(sequence);
        
        let mut pairs: Vec<WordIdxPair> = Vec::with_capacity(pos_pairs.len() * (1 + self.config.n_neg));
        let mut labels: Vec<bool> = Vec::with_capacity(pos_pairs.len() * (1 + self.config.n_neg));
        
        for pos_pair in pos_pairs {
            pairs.push(pos_pair);
            labels.push(true);
            
            let cen = pos_pair[0];
            for _ in 0..self.config.n_neg {
                pairs.push([cen, self.sampler.sample()]);
                labels.push(false);
            }
        }
        
        (pairs, labels)
    }

    pub fn process_sequences(&mut self, sequences: &[Vec<usize>]) -> (Vec<WordIdxPair>, Vec<bool>) {
        let mut all_pairs = Vec::new();
        let mut all_labels = Vec::new();
        
        for sequence in sequences {
            let (pairs, labels) = self.process_sequence(sequence);
            all_pairs.extend(pairs);
            all_labels.extend(labels);
        }
        
        (all_pairs, all_labels)
    }

}

/// Iterator for positive and negative samples with tokenization support
pub struct SGIterWithTokenization {
    config: SGConfigWithTokenization,
    pos_iter: SGPosIter,
    sampler: CategoricalSampler,
}

impl SGIterWithTokenization {
    pub fn new(config: SGConfigWithTokenization, seed: u64, num_threads: usize) -> Self {
        let rng = ChaCha20Rng::seed_from_u64(seed);
        let pos_iter = SGPosIter::new(config.base_config.clone(), seed);
        let sampler = CategoricalSampler::new(&config.base_config.probs.id2negprobs, &rng, num_threads);
        
        Self {
            config,
            pos_iter,
            sampler,
        }
    }

    /// Process string sequences with tokenization
    pub fn process_string_sequences(&mut self, text_sequences: &[String]) -> (Vec<WordIdxPair>, Vec<bool>) {
        let mut all_pairs = Vec::new();
        let mut all_labels = Vec::new();
        
        for text in text_sequences {
            let (pairs, labels) = self.process_string(text);
            all_pairs.extend(pairs);
            all_labels.extend(labels);
        }
        
        (all_pairs, all_labels)
    }

    /// Process a single string with tokenization
    pub fn process_string(&mut self, text: &str) -> (Vec<WordIdxPair>, Vec<bool>) {
        // Tokenize the string
        let tokens: Vec<&str> = text.split(&self.config.separator)
            .filter(|token| !token.is_empty())
            .collect();
        
        // Convert tokens to IDs
        let mut sequence = Vec::new();
        for token in tokens {
            if let Some(&id) = self.config.word_to_id.get(token) {
                sequence.push(id);
            }
            // Skip unknown tokens (could also use a special UNK token)
        }
        
        // Process the tokenized sequence
        if sequence.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            self.pos_iter.process_sequence(&sequence);
            let pos_pairs = self.pos_iter.process_sequence(&sequence);
            
            let mut pairs = Vec::with_capacity(pos_pairs.len() * (1 + self.config.base_config.n_neg));
            let mut labels = Vec::with_capacity(pos_pairs.len() * (1 + self.config.base_config.n_neg));
            
            for pos_pair in pos_pairs {
                // Add positive pair
                pairs.push(pos_pair);
                labels.push(true);
                
                // Add negative pairs
                for _ in 0..self.config.base_config.n_neg {
                    pairs.push([pos_pair[0], self.sampler.sample()]);
                    labels.push(false);
                }
            }
            
            (pairs, labels)
        }
    }
}

// Update SGProbs to support creation from word counts
impl SGProbs {
    fn new_from_counts(word_counts: &HashMap<usize, u64>, config: &SGConfig) -> Self {
        let total_count = config.total_count as f64;
        let mut ids: Vec<usize> = Vec::new();
        let mut probs: Vec<f64> = Vec::new();
        
        word_counts.iter().for_each(|(id, count)| {
            ids.push(*id);
            probs.push((*count as f64) / total_count);
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
}
