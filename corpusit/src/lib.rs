pub mod fileread;
pub mod vocab;
pub mod skipgram;
pub use vocab::{Vocab, VocabBuilder, VocabReader};
pub use skipgram::{SGDataset, SGDatasetConfig};