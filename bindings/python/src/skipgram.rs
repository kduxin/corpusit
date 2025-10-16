use pyo3::prelude::*;
use std::collections::HashMap;
extern crate corpusit as cit;
use cit::skipgram::{SGConfig, SGPosIter, SGIter, SGConfigWithTokenization, SGIterWithTokenization};
use ndarray::{Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2};


// ----------------------- SkipGram Python Bindings ------------------------

#[pyclass(module = "corpusit", name = "SkipGramConfig", unsendable)]
pub struct PySGConfig {
    config: SGConfig,
}

impl From<PySGConfig> for SGConfig {
    fn from(config: PySGConfig) -> Self {
        config.config
    }
}

impl From<SGConfig> for PySGConfig {
    fn from(config: SGConfig) -> Self {
        Self { config: config }
    }
}

#[pymethods]
impl PySGConfig {
    #[pyo3(signature = (word_counts, win_size = 5, subsample = 1e-3, power = 0.75, n_neg = 1))]
    #[new]
    fn new(
        word_counts: HashMap<usize, u64>,
        win_size: usize,
        subsample: f64,
        power: f64,
        n_neg: usize,
    ) -> Self {
        let config = SGConfig::new(word_counts)
            .with_win_size(win_size)
            .with_subsample(subsample)
            .with_power(power)
            .with_n_neg(n_neg);
        
        Self { config: config }
    }


    #[pyo3(signature = (seed = 0))]
    pub fn positive_sampler(
        &self,
        seed: u64,
    ) -> PySGPosIter {
        PySGPosIter {
            it: SGPosIter::new(self.config.clone(), seed),
        }
    }

    #[pyo3(signature = (seed = 0, num_threads = 4))]
    pub fn sampler(
        &self,
        seed: u64,
        num_threads: usize,
    ) -> PySGIter {
        PySGIter {
            it: SGIter::new(self.config.clone(), seed, num_threads),
        }
    }
}

#[pyclass(module = "corpusit", name = "SkipGramPosIter", unsendable)]
pub struct PySGPosIter {
    it: SGPosIter,
}

#[pymethods]
impl PySGPosIter {
    pub fn process_sequence(&mut self, sequence: Vec<usize>) -> Py<PyArray2<i64>> {
        let pairs = self.it.process_sequence(&sequence);
        let mut arr = Array2::<i64>::default((pairs.len(), 2));
        for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {
            let pair = pairs[i];
            row[0] = pair[0] as i64;
            row[1] = pair[1] as i64;
        }

        Python::with_gil(|py| {
            let y = PyArray2::from_owned_array_bound(py, arr);
            y.to_owned().into()
        })
    }

    pub fn process_sequences(&mut self, sequences: Vec<Vec<usize>>) -> Py<PyArray2<i64>> {
        let pairs = self.it.process_sequences(&sequences);
        let mut arr = Array2::<i64>::default((pairs.len(), 2));
        for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {
            let pair = pairs[i];
            row[0] = pair[0] as i64;
            row[1] = pair[1] as i64;
        }

        Python::with_gil(|py| {
            let y = PyArray2::from_owned_array_bound(py, arr);
            y.to_owned().into()
        })
    }
}

#[pyclass(module = "corpusit", name = "SkipGramIter", unsendable)]
pub struct PySGIter {
    it: SGIter,
}

#[pymethods]
impl PySGIter {
    pub fn process_sequence(&mut self, sequence: Vec<usize>) -> (Py<PyArray2<i64>>, Py<PyArray1<bool>>) {
        let (pairs, labels) = self.it.process_sequence(&sequence);
        let mut pairs_arr = Array2::<i64>::default((pairs.len(), 2));
        let mut labels_arr = Array1::<bool>::default(labels.len());
        
        for (i, mut row) in pairs_arr.axis_iter_mut(Axis(0)).enumerate() {
            let pair = pairs[i];
            row[0] = pair[0] as i64;
            row[1] = pair[1] as i64;
        }
        
        for (i, val) in labels_arr.iter_mut().enumerate() {
            *val = labels[i];
        }

        Python::with_gil(|py| {
            (
                PyArray2::from_owned_array_bound(py, pairs_arr).to_owned().into(),
                PyArray1::from_owned_array_bound(py, labels_arr).to_owned().into(),
            )
        })
    }

    pub fn process_sequences(&mut self, sequences: Vec<Vec<usize>>) -> (Py<PyArray2<i64>>, Py<PyArray1<bool>>) {
        let (pairs, labels) = self.it.process_sequences(&sequences);
        let mut pairs_arr = Array2::<i64>::default((pairs.len(), 2));
        let mut labels_arr = Array1::<bool>::default(labels.len());
        
        for (i, mut row) in pairs_arr.axis_iter_mut(Axis(0)).enumerate() {
            let pair = pairs[i];
            row[0] = pair[0] as i64;
            row[1] = pair[1] as i64;
        }
        
        for (i, val) in labels_arr.iter_mut().enumerate() {
            *val = labels[i];
        }

        Python::with_gil(|py| {
            (
                PyArray2::from_owned_array_bound(py, pairs_arr).to_owned().into(),
                PyArray1::from_owned_array_bound(py, labels_arr).to_owned().into(),
            )
        })
    }

}

// ----------------------- SkipGram with Tokenization Python Bindings ------------------------

#[pyclass(module = "corpusit", name = "SkipGramConfigWithTokenization", unsendable)]
pub struct PySGConfigWithTokenization {
    config: SGConfigWithTokenization,
}

impl From<PySGConfigWithTokenization> for SGConfigWithTokenization {
    fn from(config: PySGConfigWithTokenization) -> Self {
        config.config
    }
}

impl From<SGConfigWithTokenization> for PySGConfigWithTokenization {
    fn from(config: SGConfigWithTokenization) -> Self {
        Self { config: config }
    }
}

#[pymethods]
impl PySGConfigWithTokenization {
    #[pyo3(signature = (word_counts, word_to_id, separator = " ", win_size = 5, subsample = 1e-3, power = 0.75, n_neg = 1))]
    #[new]
    fn new(
        word_counts: HashMap<usize, u64>,
        word_to_id: HashMap<String, usize>,
        separator: &str,
        win_size: usize,
        subsample: f64,
        power: f64,
        n_neg: usize,
    ) -> Self {
        let config = SGConfigWithTokenization::new(word_counts, word_to_id, separator.to_string())
            .with_win_size(win_size)
            .with_subsample(subsample)
            .with_power(power)
            .with_n_neg(n_neg);
        
        Self { config: config }
    }

    #[pyo3(signature = (seed = 0, num_threads = 4))]
    pub fn sampler(
        &self,
        seed: u64,
        num_threads: usize,
    ) -> PySGIterWithTokenization {
        PySGIterWithTokenization {
            it: SGIterWithTokenization::new(self.config.clone(), seed, num_threads),
        }
    }
}

#[pyclass(module = "corpusit", name = "SkipGramIterWithTokenization", unsendable)]
pub struct PySGIterWithTokenization {
    it: SGIterWithTokenization,
}

#[pymethods]
impl PySGIterWithTokenization {
    /// Process string sequences with tokenization
    pub fn process_string_sequences(&mut self, text_sequences: Vec<String>) -> (Py<PyArray2<i64>>, Py<PyArray1<bool>>) {
        let (pairs, labels) = self.it.process_string_sequences(&text_sequences);
        let mut pairs_arr = Array2::<i64>::default((pairs.len(), 2));
        let mut labels_arr = Array1::<bool>::default(labels.len());
        
        for (i, mut row) in pairs_arr.axis_iter_mut(Axis(0)).enumerate() {
            let pair = pairs[i];
            row[0] = pair[0] as i64;
            row[1] = pair[1] as i64;
        }
        
        for (i, val) in labels_arr.iter_mut().enumerate() {
            *val = labels[i];
        }

        Python::with_gil(|py| {
            (
                PyArray2::from_owned_array_bound(py, pairs_arr).to_owned().into(),
                PyArray1::from_owned_array_bound(py, labels_arr).to_owned().into(),
            )
        })
    }

    /// Process a single string with tokenization
    pub fn process_string(&mut self, text: String) -> (Py<PyArray2<i64>>, Py<PyArray1<bool>>) {
        let (pairs, labels) = self.it.process_string(&text);
        let mut pairs_arr = Array2::<i64>::default((pairs.len(), 2));
        let mut labels_arr = Array1::<bool>::default(labels.len());
        
        for (i, mut row) in pairs_arr.axis_iter_mut(Axis(0)).enumerate() {
            let pair = pairs[i];
            row[0] = pair[0] as i64;
            row[1] = pair[1] as i64;
        }
        
        for (i, val) in labels_arr.iter_mut().enumerate() {
            *val = labels[i];
        }

        Python::with_gil(|py| {
            (
                PyArray2::from_owned_array_bound(py, pairs_arr).to_owned().into(),
                PyArray1::from_owned_array_bound(py, labels_arr).to_owned().into(),
            )
        })
    }
}
