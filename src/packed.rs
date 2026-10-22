//! Bindings for sequences encoded for scanning.

use std::collections::HashMap;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use universalmotif::io::fasta;
use universalmotif::packed::EncodedSequences;

use crate::{err, parse_alphabet};

/// Named sequences packed for scanning (wrapped by the Python
/// `PackedSequences` mapping).
#[pyclass(name = "_PackedSequences", module = "universalmotif", frozen)]
pub(crate) struct PyPacked {
    pub(crate) names: Vec<String>,
    index: HashMap<String, usize>,
    pub(crate) seqs: EncodedSequences,
}

impl PyPacked {
    fn new(names: Vec<String>, seqs: EncodedSequences) -> PyResult<PyPacked> {
        let mut index = HashMap::with_capacity(names.len());
        for (i, n) in names.iter().enumerate() {
            if index.insert(n.clone(), i).is_some() {
                return Err(PyValueError::new_err(format!(
                    "the sequence name '{n}' appears more than once; packed sequences need unique names"
                )));
            }
        }
        Ok(PyPacked { names, index, seqs })
    }
}

#[pymethods]
impl PyPacked {
    fn __len__(&self) -> usize {
        self.names.len()
    }

    fn names(&self) -> Vec<String> {
        self.names.clone()
    }

    /// The position of the sequence called `name`, if there is one.
    fn index(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    /// Sequence `i` as text.
    fn decode(&self, py: Python<'_>, i: usize) -> PyResult<String> {
        if i >= self.names.len() {
            return Err(PyIndexError::new_err("sequence index out of range"));
        }
        Ok(py.detach(|| self.seqs.decode(i)))
    }

    /// Positions `start..end` (0-based, half-open) of sequence `i` as text.
    fn decode_range(&self, i: usize, start: usize, end: usize) -> PyResult<String> {
        if i >= self.names.len() {
            return Err(PyIndexError::new_err("sequence index out of range"));
        }
        if start > end || end > self.seqs.seq_len(i) {
            return Err(PyIndexError::new_err("range outside the sequence"));
        }
        let mut out = Vec::with_capacity(end - start);
        self.seqs.decode_into(i, start, end, &mut out);
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    fn lengths(&self) -> Vec<usize> {
        (0..self.seqs.len()).map(|i| self.seqs.seq_len(i)).collect()
    }

    #[getter]
    fn alphabet(&self) -> String {
        self.seqs.alphabet().name()
    }

    /// Memory held by the packed letters, in bytes.
    #[getter]
    fn nbytes(&self) -> usize {
        self.seqs.heap_bytes()
    }
}

#[pyfunction]
fn _read_fasta_packed(
    py: Python<'_>,
    path: std::path::PathBuf,
    alphabet: &str,
    skip: usize,
) -> PyResult<PyPacked> {
    let alphabet = parse_alphabet(alphabet)?;
    let (names, seqs) = py
        .detach(|| fasta::read_encoded(&path, &alphabet, skip))
        .map_err(err)?;
    PyPacked::new(names, seqs)
}

#[pyfunction]
fn _pack_sequences(
    py: Python<'_>,
    names: Vec<String>,
    sequences: Vec<PyBackedStr>,
    alphabet: &str,
    nthreads: usize,
) -> PyResult<PyPacked> {
    let alphabet = parse_alphabet(alphabet)?;
    let seqs = py
        .detach(|| EncodedSequences::new(&sequences, &alphabet, nthreads))
        .map_err(err)?;
    PyPacked::new(names, seqs)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPacked>()?;
    m.add_function(wrap_pyfunction!(_read_fasta_packed, m)?)?;
    m.add_function(wrap_pyfunction!(_pack_sequences, m)?)?;
    Ok(())
}
