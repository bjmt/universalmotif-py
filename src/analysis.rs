//! Bindings for P-values, scanning and hit deduplication.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use universalmotif::Motif;
use universalmotif::dedup::{Interval, dedup};
use universalmotif::pvalue::{self, PvalueOptions};
use universalmotif::scan::{NoOverlaps, OverlapPriority, ScanOptions, scan_sequences};

use crate::{PyMotif, err, parse_bkg};

fn unwrap(motifs: &[PyRef<'_, PyMotif>]) -> Vec<Motif> {
    motifs.iter().map(|m| m.inner.clone()).collect()
}

/// Backgrounds: `None`, one (a sequence of numbers or a mapping) for every
/// motif, or a list with one per motif.
fn backgrounds(motifs: &[Motif], bkg: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Vec<f64>>> {
    let Some(b) = bkg.filter(|b| !b.is_none()) else {
        return Ok(Vec::new());
    };
    let per_motif = b
        .cast::<PyList>()
        .ok()
        .and_then(|l| l.get_item(0).ok())
        .is_some_and(|first| first.extract::<f64>().is_err());
    if per_motif {
        let list = b.cast::<PyList>()?;
        if list.len() != motifs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "give one background for all motifs or one per motif",
            ));
        }
        list.iter()
            .zip(motifs)
            .map(|(x, m)| parse_bkg(&x, &m.alphabet))
            .collect()
    } else {
        motifs.iter().map(|m| parse_bkg(b, &m.alphabet)).collect()
    }
}

#[pyfunction]
#[pyo3(signature = (motifs, queries, bkg, nthreads, to_score))]
fn _motif_pvalue(
    py: Python<'_>,
    motifs: Vec<PyRef<'_, PyMotif>>,
    queries: Vec<(usize, f64)>,
    bkg: Option<&Bound<'_, PyAny>>,
    nthreads: usize,
    to_score: bool,
) -> PyResult<Vec<f64>> {
    let motifs = unwrap(&motifs);
    let opts = PvalueOptions {
        bkg: backgrounds(&motifs, bkg)?,
        nthreads,
    };
    py.detach(|| {
        if to_score {
            pvalue::motif_score(&motifs, &queries, &opts)
        } else {
            pvalue::motif_pvalue(&motifs, &queries, &opts)
        }
    })
    .map_err(err)
}

#[pyfunction]
#[pyo3(signature = (motifs, sequences, pvalue, rc, no_overlaps, nthreads))]
fn _scan_sequences<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    sequences: Vec<String>,
    pvalue: f64,
    rc: bool,
    no_overlaps: Option<(String, bool, bool)>,
    nthreads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let motifs = unwrap(&motifs);
    let no_overlaps = no_overlaps
        .map(|(by, ignore_strand, ignore_motif)| {
            let by = match by.as_str() {
                "pvalue" => OverlapPriority::Pvalue,
                "score" => OverlapPriority::Score,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "no_overlaps_by must be 'pvalue' or 'score'",
                    ));
                }
            };
            Ok(NoOverlaps {
                by,
                ignore_strand,
                ignore_motif,
            })
        })
        .transpose()?;
    let opts = ScanOptions {
        pvalue,
        rc,
        no_overlaps,
        nthreads,
    };
    let hits = py
        .detach(|| scan_sequences(&motifs, &sequences, &opts))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("motif_i", hits.iter().map(|h| h.motif).collect::<Vec<_>>())?;
    d.set_item(
        "sequence_i",
        hits.iter().map(|h| h.sequence).collect::<Vec<_>>(),
    )?;
    d.set_item("start", hits.iter().map(|h| h.start).collect::<Vec<_>>())?;
    d.set_item("end", hits.iter().map(|h| h.end).collect::<Vec<_>>())?;
    d.set_item(
        "strand",
        hits.iter().map(|h| h.strand.as_str()).collect::<Vec<_>>(),
    )?;
    d.set_item("score", hits.iter().map(|h| h.score).collect::<Vec<_>>())?;
    d.set_item(
        "score_pct",
        hits.iter().map(|h| h.score_pct).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "match",
        hits.iter().map(|h| h.matched.as_str()).collect::<Vec<_>>(),
    )?;
    d.set_item("pvalue", hits.iter().map(|h| h.pvalue).collect::<Vec<_>>())?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (sequence, motif, strand, start, end, priority, ignore_strand, ignore_motif))]
#[allow(clippy::too_many_arguments)]
fn _dedup(
    sequence: Vec<usize>,
    motif: Vec<usize>,
    strand: Vec<i32>,
    start: Vec<i64>,
    end: Vec<i64>,
    priority: Vec<f64>,
    ignore_strand: bool,
    ignore_motif: bool,
) -> Vec<bool> {
    let hits: Vec<Interval> = (0..sequence.len())
        .map(|i| Interval {
            sequence: sequence[i],
            motif: motif[i],
            strand: strand[i],
            start: start[i],
            end: end[i],
            priority: priority[i],
        })
        .collect();
    dedup(&hits, ignore_strand, ignore_motif)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_motif_pvalue, m)?)?;
    m.add_function(wrap_pyfunction!(_scan_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(_dedup, m)?)?;
    Ok(())
}
