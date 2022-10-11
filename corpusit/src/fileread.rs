use itertools::Itertools;
use positioned_io::{RandomAccessFile, ReadAt};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Lines};

pub trait ResetableFileReader: Iterator<Item = String> + Reset {}
impl<T: Iterator<Item = String> + Reset> ResetableFileReader for T {}

pub trait Reset {
    fn reset(&mut self);
}

pub struct RanLineReader {
    file: RandomAccessFile,
    file_size: u64,
    buffer_size: usize,
    lines_buffer: Vec<String>,
    progress: usize,
    rng: ChaCha20Rng,
}

pub struct SeqLineReader {
    path: String,
    lines: Lines<BufReader<File>>,
    repeat: bool,
}

impl Reset for RanLineReader {
    fn reset(&mut self) {
        self.lines_buffer.clear();
        self.rng = ChaCha20Rng::from_seed(self.rng.get_seed());
    }
}

impl Reset for SeqLineReader {
    fn reset(&mut self) {
        self.lines = io::BufReader::new(File::open(&self.path).unwrap()).lines();
    }
}

impl RanLineReader {
    pub fn open(path: &str) -> Self {
        let file = File::open(path).unwrap();
        let file_size = file.metadata().unwrap().len();
        let rfile = RandomAccessFile::open(path).unwrap();
        RanLineReader {
            file: rfile,
            file_size: file_size,
            buffer_size: 1024 * 1024 * 100,
            lines_buffer: Vec::new(),
            progress: 0,
            rng: ChaCha20Rng::seed_from_u64(0),
        }
    }

    pub fn buffer_size_(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }
}

impl Iterator for RanLineReader {
    type Item = String;
    fn next(&mut self) -> Option<Self::Item> {
        if self.progress < self.lines_buffer.len() {
            let line = self.lines_buffer[self.progress].to_string();
            self.progress += 1;
            return Some(line);
        }

        let pos = if self.file_size > (self.buffer_size as u64) {
            self.rng.gen::<u64>() % (self.file_size - self.buffer_size as u64)
        } else {
            0
        };
        let mut buf = vec![0; self.buffer_size.min((self.file_size - pos) as usize)];
        let bytes_read = self.file.read_at(pos, &mut buf).unwrap();
        if bytes_read == 0 {
            self.next()
        } else {
            let buffer = String::from_utf8_lossy(&buf);
            let lines = buffer.lines().collect_vec();
            if lines.len() <= 2 {
                println!("Warning: found a line too long (length > {}). Consider using a larger buffer_size.",
                    lines.into_iter().map(|line| line.len()).max().unwrap_or(0));
                self.next()
            } else {
                self.lines_buffer.clear();
                for line in lines[1..lines.len() - 1].into_iter() {
                    self.lines_buffer.push(line.to_string());
                }
                self.progress = 1;
                Some(self.lines_buffer[0].to_string())
            }
        }
    }
}

static BUFSIZE: usize = 1024 * 1024 * 100;

impl SeqLineReader {
    pub fn open(path: &str, repeat: bool) -> Self {
        let file = File::open(path).unwrap();
        let reader = BufReader::with_capacity(BUFSIZE, file);
        let lines = reader.lines();
        SeqLineReader {
            path: path.to_string(),
            lines: lines,
            repeat: repeat,
        }
    }

    fn reset(&mut self) {
        let file = File::open(&self.path).unwrap();
        let file = BufReader::with_capacity(BUFSIZE, file);
        self.lines = file.lines();
    }
}

impl Iterator for SeqLineReader {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        match self.lines.next() {
            Some(line) => Some(line.unwrap()),
            None => match self.repeat {
                true => {
                    self.reset();
                    self.next()
                }
                false => None,
            },
        }
    }
}

// ----------------------- File Iterate Mode ------------------------

#[derive(Clone)]
pub enum FileIterateMode {
    ONEPASS,
    REPEAT,
    SHUFFLE,
}

impl Default for FileIterateMode {
    fn default() -> FileIterateMode {
        FileIterateMode::ONEPASS
    }
}

pub fn get_file_reader(path: &str, mode: FileIterateMode) -> Box<dyn ResetableFileReader + Send> {
    match mode {
        FileIterateMode::SHUFFLE => Box::new(RanLineReader::open(&path)),
        FileIterateMode::ONEPASS => Box::new(SeqLineReader::open(&path, false)),
        FileIterateMode::REPEAT => Box::new(SeqLineReader::open(&path, true)),
    }
}
