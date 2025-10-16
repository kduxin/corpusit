
// extern crate corpusit;

use pyo3::prelude::*;

mod skipgram;

/// Rust based corpus iterators
#[pymodule]
#[pyo3(name = "corpusit")]
fn python(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<skipgram::PySGConfig>()?;
    m.add_class::<skipgram::PySGPosIter>()?;
    m.add_class::<skipgram::PySGIter>()?;
    m.add_class::<skipgram::PySGConfigWithTokenization>()?;
    m.add_class::<skipgram::PySGIterWithTokenization>()?;
    
    Ok(())
}