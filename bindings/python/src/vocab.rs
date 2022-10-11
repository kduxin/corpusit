use bincode;
use serde_json;
use std::{fs, io};
use std::collections::{
    HashMap,
};
use std::sync::{Arc, RwLock};
use pyo3::{
    prelude::*,
    types::PyBytes,
};

extern crate corpusit;
use corpusit::Vocab;

/// A vocabulary, containing indexed words and their counts in a corpus.
#[pyclass(dict, module = "corpusit", name = "Vocab", subclass)]
pub struct PyVocab {
    pub vocab: Arc<RwLock<Vocab>>,
    #[pyo3(get)]
    pub s2i: S2I,
    #[pyo3(get)]
    pub i2s: I2S,
    #[pyo3(get)]
    pub counts: Counts,
    #[pyo3(get)]
    pub i2count: I2Count,
}

impl From<Vocab> for PyVocab {
    fn from(vocab: Vocab) -> Self {
        let vocab = Arc::new(RwLock::new(vocab));
        Self::from(vocab)
    }
}

impl From<Arc<RwLock<Vocab>>> for PyVocab {
    fn from(vocab: Arc<RwLock<Vocab>>) -> Self {
        let vocab = vocab;
        let s2i = S2I {vocab: Arc::clone(&vocab)};
        let i2s = I2S {vocab: Arc::clone(&vocab)};
        let counts = Counts {vocab: Arc::clone(&vocab)};
        let i2count = I2Count {vocab: Arc::clone(&vocab)};
        Self { vocab: Arc::clone(&vocab), s2i: s2i, i2s: i2s, counts: counts, i2count: i2count}
    }
}

impl PyVocab {
    fn update(&mut self, min_count: Option<u64>, max_size: Option<usize>, unk: Option<&str>) {
        if let Some(unk) = unk {
            self.vocab.write().unwrap().unk_(&unk);
        }
        let min_count = min_count.unwrap_or(1);
        let max_size = max_size.unwrap_or(usize::MAX);
        self.vocab.write().unwrap().truncate(min_count, max_size);
    }
}

#[pymethods]
impl PyVocab {
    /// Read a Vocab stored in a json file at `path_to_json`
    /// Parameters
    ///   - min_count: set a new count threshold. All words with smaller
    ///         counts are truncated, and viewed as {unk}.
    ///   - max_size: set a new vocabulary size limit.
    ///   - unk: set / reset the {unk} token.
    #[args(min_count = "None", max_size = "None", unk = "None")]
    #[pyo3(text_signature = "(path_to_json, min_count=None, max_size=None, unk=None)")]
    #[staticmethod]
    pub fn from_json(
        path_to_json: String,
        min_count: Option<u64>,
        max_size: Option<usize>,
        unk: Option<&str>,
    ) -> Self {
        let vocab_f = fs::File::open(path_to_json).unwrap();
        let vocab_f = io::BufReader::new(vocab_f);
        let vocab: corpusit::Vocab = serde_json::from_reader(vocab_f).unwrap();
        let mut pyvocab = Self::from(vocab);
        pyvocab.update(min_count, max_size, unk);
        pyvocab
    }

    /// Read a Vocab stored in a binary file at `path_to_bin`
    /// Parameters
    ///   - min_count: set a new count threshold. All words with smaller
    ///         counts are truncated, and viewed as {unk}.
    ///   - max_size: set a new vocabulary size limit.
    ///   - unk: set / reset the {unk} token.
    #[args(min_count = "None", max_size = "None", unk = "None")]
    #[pyo3(text_signature = "(path_to_bin, min_count=None, max_size=None, unk=None)")]
    #[staticmethod]
    pub fn from_bin(
        path_to_bin: String,
        min_count: Option<u64>,
        max_size: Option<usize>,
        unk: Option<&str>,
    ) -> Self {
        let vocab_f = fs::File::open(path_to_bin).unwrap();
        let vocab_f = io::BufReader::new(vocab_f);
        let vocab: corpusit::Vocab = bincode::deserialize_from(vocab_f).unwrap();
        let mut pyvocab = Self::from(vocab);
        pyvocab.update(min_count, max_size, unk);
        pyvocab
    }

    /// Build a Vocab by with a corpus at `path_to_corpus`
    /// Parameters
    ///   - min_count: set a new count threshold. All words with smaller
    ///         counts are truncated, and viewed as {unk} (or discarded).
    ///   - max_size: set a new vocabulary size limit.
    ///   - unk: set the {unk} token.
    ///   - path_to_save_json: if not specified, will save at
    ///         ${path_to_corpus}.vocab.json
    ///   - path_to_save_bin: if not specified, will save at
    ///         ${path_to_corpus}.vocab.bin
    #[args(min_count = "5", max_size = "None", unk = "\"<unk>\"")]
    #[pyo3(
        text_signature = "(path_to_corpus, min_count=None, max_size=None, \
                          unk=None, path_to_save_json=None, path_to_save_bin=None)"
    )]
    #[staticmethod]
    pub fn build(
        path_to_corpus: &str,
        min_count: Option<u64>,
        max_size: Option<usize>,
        unk: Option<&str>,
        path_to_save_json: Option<&str>,
        path_to_save_bin: Option<&str>,
    ) -> Self {
        let min_count = min_count.unwrap_or(1);
        let max_size = max_size.unwrap_or(usize::MAX);
        let mut vocab_builder = corpusit::VocabBuilder::new(&path_to_corpus);
        if let Some(unk) = unk {
            vocab_builder = vocab_builder.infrequent_filtering_(min_count, &unk);
        }
        let mut vocab = vocab_builder.build();

        let vocab_path = match path_to_save_json {
            Some(path) => path.to_string(),
            None => path_to_corpus.to_string() + ".vocab.json",
        };
        let vocab_f = fs::File::create(vocab_path).unwrap();
        let vocab_f = io::BufWriter::new(vocab_f);
        serde_json::to_writer_pretty(vocab_f, &vocab).unwrap();

        let vocab_path = match path_to_save_bin {
            Some(path) => path.to_string(),
            None => path_to_corpus.to_string() + ".vocab.bin",
        };
        let vocab_f = fs::File::create(vocab_path).unwrap();
        let vocab_f = io::BufWriter::new(vocab_f);
        bincode::serialize_into(vocab_f, &vocab).unwrap();

        vocab.truncate(min_count, max_size);
        Self::from(vocab)
    }

    #[getter]
    pub fn __len__(slf: PyRef<Self>) -> usize {
        slf.vocab.read().unwrap().i2s.len()
    }

    pub fn i2s_dict(slf: PyRef<Self>) -> HashMap<usize, String> {
        slf.vocab.read().unwrap().i2s.clone()
    }

    pub fn s2i_dict(slf: PyRef<Self>) -> HashMap<String, usize> {
        slf.vocab.read().unwrap().s2i.clone()
    }

    pub fn i2count_dict(slf: PyRef<Self>) -> HashMap<usize, u64> {
        slf.vocab.read().unwrap().i2count.clone()
    }

    pub fn counts_dict(slf: PyRef<Self>) -> HashMap<String, u64> {
        let vocab = slf.vocab.read().unwrap();
        vocab.i2count.iter().map(|(i, c)| {
            (vocab.i2s[i].clone(), *c)
        }).collect()
    }

    pub fn get_special_tokens(slf: PyRef<Self>) -> HashMap<String, String> {
        slf.vocab.read().unwrap().special_name2i.iter().map(|(name, id)| {
            (name.to_string(), slf.vocab.read().unwrap().i2s[id].clone())
        }).collect()
    }

    pub fn __contains__(slf: PyRef<Self>, s: &str) -> bool {
        match slf.vocab.read().unwrap().s2i.get(s) {
            Some(_) => true,
            None => false,
        }
    }

    pub fn keys(slf: PyRef<Self>) -> Vec<String> {
        slf.vocab.read().unwrap().s2i.keys().map(|s| s.to_string()).collect()
    }

    pub fn __repr__(slf: PyRef<Self>) -> String {
        let vocab = slf.vocab.read().unwrap();
        let mut special_tokens_str = String::new();
        for (name, id) in vocab.special_name2i.iter() {
            special_tokens_str += &format!("{}: {}, ", name, vocab.i2s[id]);
        }
        special_tokens_str.pop();
        special_tokens_str.pop();
        format!("<Vocab(size={}, special_tokens={{{}}})", vocab.i2s.len(), special_tokens_str)
    }

    #[getter]
    pub fn unk(slf: PyRef<Self>) -> Option<String> {
        slf.vocab.read().unwrap().unk_str().and_then(|s| Some(s.to_string()))
    }

    #[getter]
    pub fn unk_id(slf: PyRef<Self>) -> Option<usize> {
        slf.vocab.read().unwrap().unk_id().and_then(|id| Some(*id))
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                let saved: Vocab = bincode::deserialize(s.as_bytes()).unwrap();
                let tmp = PyVocab::from(saved);
                self.vocab = tmp.vocab;
                self.s2i = tmp.s2i;
                self.i2s = tmp.i2s;
                self.counts = tmp.counts;
                self.i2count = tmp.i2count;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &bincode::serialize(&self.vocab).unwrap()).to_object(py))
    }

}


#[derive(Clone)]
#[pyclass]
pub struct S2I {
    vocab: Arc<RwLock<Vocab>>,
}

#[derive(Clone)]
#[pyclass]
pub struct I2S {
    vocab: Arc<RwLock<Vocab>>,
}

#[derive(Clone)]
#[pyclass]
pub struct I2Count {
    vocab: Arc<RwLock<Vocab>>,
}

#[derive(Clone)]
#[pyclass]
pub struct Counts {
    vocab: Arc<RwLock<Vocab>>,
}

#[pymethods]
impl S2I {
    pub fn __getitem__(slf: PyRef<Self>, s: &str) -> Option<usize> {
        slf.vocab.read().unwrap().s2i.get(s).and_then(|id| Some(*id))
    }

    pub fn get(slf: PyRef<Self>, s: &str, default: usize) -> usize {
        match S2I::__getitem__(slf, s) {
            Some(i) => i,
            None => default,
        }
    }

    pub fn __len__(slf: PyRef<Self>) -> usize {
        slf.vocab.read().unwrap().i2s.len()
    }
}

#[pymethods]
impl I2S {
    pub fn __getitem__(slf: PyRef<Self>, id: usize) -> Option<String> {
        slf.vocab.read().unwrap().i2s.get(&id).and_then(|s| Some(s.to_string()))
    }

    pub fn get(slf: PyRef<Self>, id: usize, default: String) -> String {
        match I2S::__getitem__(slf, id) {
            Some(s) => s,
            None => default,
        }
    }

    pub fn __len__(slf: PyRef<Self>) -> usize {
        slf.vocab.read().unwrap().i2s.len()
    }
}

#[pymethods]
impl I2Count {
    pub fn __getitem__(slf: PyRef<Self>, id: usize) -> Option<u64> {
        slf.vocab.read().unwrap().i2count.get(&id).and_then(|c| Some(*c))
    }

    pub fn get(slf: PyRef<Self>, id: usize, default: u64) -> u64 {
        match I2Count::__getitem__(slf, id) {
            Some(c) => c,
            None => default,
        }
    }

    pub fn __len__(slf: PyRef<Self>) -> usize {
        slf.vocab.read().unwrap().i2s.len()
    }
}

#[pymethods]
impl Counts {
    pub fn __getitem__(slf: PyRef<Self>, s: &str) -> Option<u64> {
        let vocab = slf.vocab.read().unwrap();
        vocab.s2i.get(s).and_then(|id| {
            Some(vocab.i2count[id])
        })
    }

    pub fn get(slf: PyRef<Self>, s: &str, default: u64) -> u64 {
        match Counts::__getitem__(slf, s) {
            Some(c) => c,
            None => default,
        }
    }

    pub fn __len__(slf: PyRef<Self>) -> usize {
        slf.vocab.read().unwrap().i2s.len()
    }
}