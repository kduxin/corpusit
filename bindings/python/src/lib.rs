
// extern crate corpusit;

use pyo3::prelude::*;

mod vocab;
mod skipgram;

/// Rust based corpus iterators
#[pymodule]
#[pyo3(name = "corpusit")]
fn python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<vocab::PyVocab>()?;
    m.add_class::<skipgram::PySGDataset>()?;
    m.add_class::<skipgram::PySGDatasetPosIter>()?;
    m.add_class::<skipgram::PySGDatasetIter>()?;
    
    Ok(())
}