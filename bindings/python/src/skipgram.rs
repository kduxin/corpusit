use corpusit::SGDatasetConfig;
use pyo3::prelude::*;
use std::sync::Arc;
extern crate corpusit as cit;
use crate::vocab::PyVocab;
use cit::fileread::FileIterateMode;
use cit::skipgram::{SGDataset, SGDatasetBuilder, SGDatasetIter, SGDatasetPosIter};
use ndarray::{Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2};

#[pyclass(module = "corpusit", name = "SkipGramDataset", unsendable)]
pub struct PySGDataset {
    dataset: SGDataset,
}

impl From<PySGDataset> for SGDataset {
    fn from(dataset: PySGDataset) -> Self {
        dataset.dataset
    }
}

impl From<SGDataset> for PySGDataset {
    fn from(dataset: SGDataset) -> Self {
        Self { dataset: dataset }
    }
}

#[pymethods]
impl PySGDataset {
    #[args(
        win_size = "10",
        sep = "\" \"",
        mode = "\"shuffle\"",
        subsample = "1e-5",
        power = "0.75",
        n_neg = "1"
    )]
    #[new]
    fn new(
        path_to_corpus: &str,
        vocab: &PyVocab,
        win_size: usize,
        sep: &str,
        mode: &str,
        subsample: f64,
        power: f64,
        n_neg: usize,
    ) -> Self {
        let config = SGDatasetConfig {
            corpus_path: path_to_corpus.to_string(),
            win_size: win_size,
            sep: sep.to_string(),
            mode: match mode {
                "shuffle" => FileIterateMode::SHUFFLE,
                "onepass" => FileIterateMode::ONEPASS,
                "repeat" => FileIterateMode::REPEAT,
                _other => panic!("Unknown file iterating mode {}", mode),
            },
            subsample: subsample,
            power: power,
            n_neg: n_neg,
        };
        let dataset = SGDatasetBuilder::new(config).build_with_vocab(&vocab.vocab.read().unwrap());
        Self { dataset: dataset }
    }

    #[args(seed = "0", num_threads = "4")]
    #[pyo3(text_signature = "(self, batch_size, seed=0, num_threads=4)")]
    pub fn positive_sampler(
        &self,
        batch_size: usize,
        seed: u64,
        num_threads: usize,
    ) -> PySGDatasetPosIter {
        PySGDatasetPosIter {
            it: self.dataset.positive_sampler(seed, num_threads),
            batch_size: batch_size,
        }
    }

    #[args(seed = "0", num_threads = "4")]
    #[pyo3(text_signature = "(self, batch_size, seed=0, num_threads=4)")]
    pub fn sampler(
        &self,
        batch_size: usize,
        seed: u64,
        num_threads: usize,
    ) -> PySGDatasetIter {
        PySGDatasetIter {
            it: self.dataset.sampler(seed, num_threads),
            batch_size: batch_size,
        }
    }

    #[getter]
    pub fn vocab(&self) -> PyVocab {
        PyVocab::from(Arc::clone(&self.dataset.vocab))
    }
}

#[pyclass(module = "corpusit", name = "SkipGramPosIter", unsendable)]
pub struct PySGDatasetPosIter {
    it: SGDatasetPosIter,
    batch_size: usize,
}

#[pymethods]
impl PySGDatasetPosIter {
    pub fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    pub fn __next__(mut slf: PyRefMut<Self>) -> Option<Py<PyArray2<i64>>> {
        let batch_size = slf.batch_size;
        let batch = slf.it.next_batch(batch_size);
        match batch {
            Some(batch) => {
                let mut arr = Array2::<i64>::default((batch.len(), 2));
                for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {
                    let pair = batch[i];
                    row[0] = pair[0] as i64;
                    row[1] = pair[1] as i64;
                }

                let res = Python::with_gil(|py| {
                    let y = PyArray2::from_owned_array(py, arr);
                    y.to_owned()
                });
                Some(res)
            }
            None => None,
        }
    }
}

#[pyclass(module = "corpusit", name = "SkipGramIter", unsendable)]
pub struct PySGDatasetIter {
    it: SGDatasetIter,
    batch_size: usize,
}

#[pymethods]
impl PySGDatasetIter {
    pub fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    pub fn __next__(mut slf: PyRefMut<Self>) -> Option<(Py<PyArray2<i64>>, Py<PyArray1<bool>>)> {
        let batch_size = slf.batch_size;
        let batch = slf.it.next_batch(batch_size);
        match batch {
            Some(batch) => {
                let len = batch.0.len();
                let mut pairs = Array2::<i64>::default((len, 2));
                let mut labels = Array1::<bool>::default(len);
                for (i, mut row) in pairs.axis_iter_mut(Axis(0)).enumerate() {
                    let pair = batch.0[i];
                    row[0] = pair[0] as i64;
                    row[1] = pair[1] as i64;
                }
                for (i, val) in labels.iter_mut().enumerate() {
                    *val = batch.1[i];
                }

                Python::with_gil(|py| {
                    Some((
                        PyArray2::from_owned_array(py, pairs).to_owned(),
                        PyArray1::from_owned_array(py, labels).to_owned(),
                    ))
                })
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::skipgram::PySGDataset;
    use crate::vocab::PyVocab;
    use tqdm::Iter;

    #[test]
    fn iterate_positive() {
        let path =
            "/home/duxin/code/firelang/data/wacky/wacky_combined.txt.UNCASED.tokens".to_string();
        // let path = "/home/duxin/code/firelang/data/wacky/wacky_combined.txt.1000000".to_string();
        let vocab_path = path.clone() + ".vocab.bin";

        let vocab = PyVocab::from_bin(vocab_path, Some(100), Some(99999999), Some("<unk>"));
        let dataset = PySGDataset::new(&path, &vocab, 10, " ", "shuffle", 1e-5, 0.75, 1);
        let mut it = dataset.positive_sampler(30000, 0, 4);
        for _ in (0..1000).into_iter().tqdm() {
            let x = it.it.next_batch(30000);
            if let Some(x) = x {
                if x.len() >= 60000 {
                    println!("123");
                }
            }
        }
    }

    #[test]
    fn iterate() {
        let path =
            "/home/duxin/code/firelang/data/wacky/wacky_combined.txt.UNCASED.tokens".to_string();
        let vocab_path = path.clone() + ".vocab.bin";
        let vocab = PyVocab::from_bin(vocab_path, Some(100), Some(99999999), Some("<unk>"));
        let dataset = PySGDataset::new(&path, &vocab, 10, " ", "shuffle", 1e-5, 0.75, 1);

        let mut it = dataset.sampler(30000, 0, 4);
        for _ in (0..1000).into_iter().tqdm() {
            let x = it.it.next_batch(30000);
            if let Some(x) = x {
                if x.1.len() >= 1000000000 {
                    println!("123");
                }
            }
        }
    }
}
