//! Bindings for genomes read a piece at a time from disk.

use pyo3::prelude::*;
use universalmotif::genome::Genome;
use universalmotif::ranges::GRange;

use crate::err;

/// A genome on disk (wrapped by the Python `Genome` mapping).
#[pyclass(name = "_Genome", module = "universalmotif", frozen)]
pub(crate) struct PyGenome {
    genome: Genome,
}

#[pymethods]
impl PyGenome {
    #[new]
    fn new(py: Python<'_>, path: std::path::PathBuf) -> PyResult<PyGenome> {
        let genome = py.detach(|| Genome::open(&path)).map_err(err)?;
        Ok(PyGenome { genome })
    }

    fn names(&self) -> Vec<String> {
        self.genome.names().to_vec()
    }

    fn lengths(&self) -> Vec<u64> {
        self.genome.lengths().to_vec()
    }

    /// The position of the sequence called `name`, if there is one.
    fn index(&self, name: &str) -> Option<usize> {
        self.genome.position(name)
    }

    /// Positions `start..end` (0-based, half-open) of sequence `i`.
    fn fetch(&self, py: Python<'_>, i: usize, start: u64, end: u64) -> PyResult<String> {
        py.detach(|| self.genome.fetch_at(i, start, end))
            .map_err(err)
    }

    /// Many ranges at once, as (sequence index, start, end) with 0-based,
    /// half-open positions, read in parallel.
    #[pyo3(signature = (ranges, nthreads=1))]
    fn fetch_many(
        &self,
        py: Python<'_>,
        ranges: Vec<(usize, u64, u64)>,
        nthreads: usize,
    ) -> PyResult<Vec<String>> {
        let ranges: Vec<GRange> = ranges
            .into_iter()
            .map(|(i, s, e)| GRange::new(i, s as i64, e as i64))
            .collect();
        py.detach(|| self.genome.fetch_many(&ranges, nthreads))
            .map_err(err)
    }

    /// The GC fraction and the fraction of other letters than A, C, G, T
    /// and U of many ranges, given as for `fetch_many`, read in parallel
    /// without keeping the sequences.
    #[pyo3(signature = (ranges, nthreads=1))]
    fn composition_many(
        &self,
        py: Python<'_>,
        ranges: Vec<(usize, u64, u64)>,
        nthreads: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let ranges: Vec<GRange> = ranges
            .into_iter()
            .map(|(i, s, e)| GRange::new(i, s as i64, e as i64))
            .collect();
        let comp = py
            .detach(|| self.genome.composition_many(&ranges, nthreads))
            .map_err(err)?;
        Ok(comp.into_iter().unzip())
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGenome>()?;
    Ok(())
}
