//! Bindings for P-values, scanning and hit deduplication.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use universalmotif::Motif;
use universalmotif::cluster::{self, Method, Tree, TreeOptions};
use universalmotif::compare::{self, CompareOptions, NullModel};
use universalmotif::dedup::{Interval, dedup};
use universalmotif::merge::{self, MergeOptions, MergeSimilarOptions};
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

fn null_model(null: &str) -> PyResult<NullModel> {
    match null {
        "empirical" => Ok(NullModel::Empirical),
        "parametric" => Ok(NullModel::Parametric),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "null must be 'empirical' or 'parametric'",
        )),
    }
}

fn bkg4(bkg: Option<Vec<f64>>) -> PyResult<[f64; 4]> {
    match bkg {
        None => Ok([0.25; 4]),
        Some(b) if b.len() == 4 => {
            if b.iter().any(|v| v.is_nan() || *v < 0.0)
                || (b.iter().sum::<f64>() - 1.0).abs() > 1e-6
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "bkg must be four non-negative values summing to 1",
                ));
            }
            Ok([b[0], b[1], b[2], b[3]])
        }
        Some(_) => Err(pyo3::exceptions::PyValueError::new_err(
            "bkg must have four values (A, C, G, T)",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (motifs, queries, min_overlap, rc, null, bkg, need_pvalues, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _compare<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    queries: Option<Vec<usize>>,
    min_overlap: usize,
    rc: bool,
    null: &str,
    bkg: Option<Vec<f64>>,
    need_pvalues: bool,
    nthreads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let motifs = unwrap(&motifs);
    let opts = CompareOptions {
        min_overlap,
        rc,
        null: null_model(null)?,
        bkg: bkg4(bkg)?,
        nthreads,
    };
    let d = PyDict::new(py);
    match queries {
        None => {
            let m = py
                .detach(|| compare::compare_matrix(&motifs, &opts, need_pvalues))
                .map_err(err)?;
            d.set_item("score", m.scores)?;
            d.set_item("pvalue", m.pvalues)?;
            d.set_item("qvalue", m.qvalues)?;
        }
        Some(q) => {
            let rows = py
                .detach(|| compare::compare_motifs(&motifs, &q, &opts))
                .map_err(err)?;
            let col = |f: fn(&compare::Comparison) -> f64| rows.iter().map(f).collect::<Vec<_>>();
            d.set_item(
                "subject_i",
                rows.iter().map(|c| c.query).collect::<Vec<_>>(),
            )?;
            d.set_item(
                "target_i",
                rows.iter().map(|c| c.target).collect::<Vec<_>>(),
            )?;
            d.set_item("offset", rows.iter().map(|c| c.offset).collect::<Vec<_>>())?;
            d.set_item(
                "strand",
                rows.iter().map(|c| c.strand.as_str()).collect::<Vec<_>>(),
            )?;
            d.set_item(
                "overlap",
                rows.iter().map(|c| c.overlap).collect::<Vec<_>>(),
            )?;
            d.set_item("score", col(|c| c.score))?;
            d.set_item(
                "subject_consensus",
                rows.iter()
                    .map(|c| c.query_consensus.as_str())
                    .collect::<Vec<_>>(),
            )?;
            d.set_item(
                "target_consensus",
                rows.iter()
                    .map(|c| c.target_consensus.as_str())
                    .collect::<Vec<_>>(),
            )?;
            d.set_item("pvalue", col(|c| c.pvalue))?;
            d.set_item("qvalue", col(|c| c.qvalue))?;
        }
    }
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (motifs, min_overlap, rc, name, weighted, nthreads))]
fn _merge_motifs(
    py: Python<'_>,
    motifs: Vec<PyRef<'_, PyMotif>>,
    min_overlap: usize,
    rc: bool,
    name: Option<String>,
    weighted: bool,
    nthreads: usize,
) -> PyResult<PyMotif> {
    let motifs = unwrap(&motifs);
    let opts = MergeOptions {
        min_overlap,
        rc,
        name,
        weighted,
        nthreads,
    };
    py.detach(|| merge::merge_motifs(&motifs, &opts))
        .map(PyMotif::from)
        .map_err(err)
}

#[pyfunction]
#[pyo3(signature = (motifs, qvalue, null, min_overlap, rc, bkg, weighted, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _merge_similar(
    py: Python<'_>,
    motifs: Vec<PyRef<'_, PyMotif>>,
    qvalue: f64,
    null: &str,
    min_overlap: usize,
    rc: bool,
    bkg: Option<Vec<f64>>,
    weighted: bool,
    nthreads: usize,
) -> PyResult<(Vec<usize>, Vec<PyMotif>)> {
    let motifs = unwrap(&motifs);
    let opts = MergeSimilarOptions {
        qvalue,
        null: null_model(null)?,
        min_overlap,
        rc,
        bkg: bkg4(bkg)?,
        weighted,
        nthreads,
    };
    let (clusters, merged) = py
        .detach(|| merge::merge_similar(&motifs, &opts))
        .map_err(err)?;
    Ok((clusters, merged.into_iter().map(PyMotif::from).collect()))
}

fn tree_dict<'py>(py: Python<'py>, tree: &Tree, labels: &[String]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(
        "merge",
        tree.merge.iter().map(|m| m.to_vec()).collect::<Vec<_>>(),
    )?;
    d.set_item("height", &tree.height)?;
    d.set_item("order", &tree.order)?;
    d.set_item("newick", tree.newick(labels))?;
    d.set_item(
        "linkage",
        tree.linkage()
            .iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
    )?;
    Ok(d)
}

fn method(m: &str) -> PyResult<Method> {
    Method::parse(m).map_err(err)
}

#[pyfunction]
fn _hclust<'py>(
    py: Python<'py>,
    dist: Vec<Vec<f64>>,
    method: &str,
    labels: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = self::method(method)?;
    let tree = py.detach(|| cluster::hclust(&dist, m)).map_err(err)?;
    tree_dict(py, &tree, &labels)
}

#[pyfunction]
#[pyo3(signature = (motifs, min_overlap, rc, method, labels, nthreads))]
fn _motif_tree<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    min_overlap: usize,
    rc: bool,
    method: &str,
    labels: Vec<String>,
    nthreads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let motifs = unwrap(&motifs);
    let opts = TreeOptions {
        min_overlap,
        rc,
        method: self::method(method)?,
        nthreads,
    };
    let tree = py
        .detach(|| cluster::motif_tree(&motifs, &opts))
        .map_err(err)?;
    tree_dict(py, &tree, &labels)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_compare, m)?)?;
    m.add_function(wrap_pyfunction!(_merge_motifs, m)?)?;
    m.add_function(wrap_pyfunction!(_merge_similar, m)?)?;
    m.add_function(wrap_pyfunction!(_hclust, m)?)?;
    m.add_function(wrap_pyfunction!(_motif_tree, m)?)?;
    m.add_function(wrap_pyfunction!(_motif_pvalue, m)?)?;
    m.add_function(wrap_pyfunction!(_scan_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(_dedup, m)?)?;
    Ok(())
}
