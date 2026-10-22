//! Bindings for enrichment, co-occurrence, positional enrichment and tree
//! cutting.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use universalmotif::Motif;
use universalmotif::cluster::Tree;
use universalmotif::coocc::{self, CooccHit, CooccOptions};
use universalmotif::enrich::{self, EnrichOptions, EnrichTest};
use universalmotif::peaks::{self, PeakHit, PeaksMode, PeaksOptions};

use crate::{PyMotif, err};

#[pyfunction]
#[pyo3(signature = (motifs, sequences, background, pvalue, test, rc, shuffle_k, seed, pseudocount, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _enrich<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    sequences: Vec<String>,
    background: Option<Vec<String>>,
    pvalue: f64,
    test: &str,
    rc: bool,
    shuffle_k: usize,
    seed: u64,
    pseudocount: u64,
    nthreads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let motifs: Vec<Motif> = motifs.iter().map(|m| m.inner.clone()).collect();
    let opts = EnrichOptions {
        pvalue,
        test: EnrichTest::parse(test).map_err(err)?,
        rc,
        shuffle_k,
        seed,
        pseudocount,
        nthreads,
    };
    let rows = py
        .detach(|| enrich::enrich_motifs(&motifs, &sequences, background.as_deref(), &opts))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("motif_i", rows.iter().map(|r| r.motif).collect::<Vec<_>>())?;
    d.set_item(
        "target_seq_n",
        rows.iter().map(|r| r.target_seq_n).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "target_seq_hits",
        rows.iter().map(|r| r.target_seq_hits).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "target_site_hits",
        rows.iter().map(|r| r.target_site_hits).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "bkg_seq_n",
        rows.iter().map(|r| r.bkg_seq_n).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "bkg_seq_hits",
        rows.iter().map(|r| r.bkg_seq_hits).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "bkg_site_hits",
        rows.iter().map(|r| r.bkg_site_hits).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "enrichment",
        rows.iter().map(|r| r.enrichment).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "log2_enrichment",
        rows.iter().map(|r| r.log2_enrichment).collect::<Vec<_>>(),
    )?;
    d.set_item("pvalue", rows.iter().map(|r| r.pvalue).collect::<Vec<_>>())?;
    d.set_item("qvalue", rows.iter().map(|r| r.qvalue).collect::<Vec<_>>())?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (motif, sequence, start, n_motifs, n_sequences, max_distance, min_coocc, pseudocount, self_pairs))]
#[allow(clippy::too_many_arguments)]
fn _coocc<'py>(
    py: Python<'py>,
    motif: Vec<usize>,
    sequence: Vec<usize>,
    start: Vec<i64>,
    n_motifs: usize,
    n_sequences: usize,
    max_distance: Option<u64>,
    min_coocc: u64,
    pseudocount: u64,
    self_pairs: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let hits: Vec<CooccHit> = (0..motif.len())
        .map(|i| CooccHit {
            motif: motif[i],
            sequence: sequence[i],
            start: start.get(i).copied().unwrap_or(0),
        })
        .collect();
    let opts = CooccOptions {
        max_distance,
        min_coocc,
        pseudocount,
        self_pairs,
    };
    let rows = py
        .detach(|| coocc::motif_coocc(&hits, n_motifs, n_sequences, &opts))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("a", rows.iter().map(|r| r.a).collect::<Vec<_>>())?;
    d.set_item("b", rows.iter().map(|r| r.b).collect::<Vec<_>>())?;
    d.set_item("a_only", rows.iter().map(|r| r.a_only).collect::<Vec<_>>())?;
    d.set_item("b_only", rows.iter().map(|r| r.b_only).collect::<Vec<_>>())?;
    d.set_item("both", rows.iter().map(|r| r.both).collect::<Vec<_>>())?;
    d.set_item(
        "neither",
        rows.iter().map(|r| r.neither).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "odds_ratio",
        rows.iter().map(|r| r.odds_ratio).collect::<Vec<_>>(),
    )?;
    d.set_item("pvalue", rows.iter().map(|r| r.pvalue).collect::<Vec<_>>())?;
    d.set_item("qvalue", rows.iter().map(|r| r.qvalue).collect::<Vec<_>>())?;
    d.set_item(
        "both_clustered",
        rows.iter().map(|r| r.both_clustered).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "median_distance",
        rows.iter().map(|r| r.median_distance).collect::<Vec<_>>(),
    )?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (motif, sequence, center, score, seq_length, mode, min_window, max_window, window_step, position_step, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _peaks<'py>(
    py: Python<'py>,
    motif: Vec<usize>,
    sequence: Vec<usize>,
    center: Vec<f64>,
    score: Vec<f64>,
    seq_length: usize,
    mode: &str,
    min_window: usize,
    max_window: Option<usize>,
    window_step: usize,
    position_step: usize,
    nthreads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let hits: Vec<PeakHit> = (0..motif.len())
        .map(|i| PeakHit {
            motif: motif[i],
            sequence: sequence[i],
            center: center[i],
            score: score[i],
        })
        .collect();
    let opts = PeaksOptions {
        mode: PeaksMode::parse(mode).map_err(err)?,
        min_window,
        max_window,
        window_step,
        position_step,
        nthreads,
    };
    let rows = py
        .detach(|| peaks::motif_peaks(&hits, seq_length, &opts))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("motif", rows.iter().map(|r| r.motif).collect::<Vec<_>>())?;
    d.set_item("nhits", rows.iter().map(|r| r.nhits).collect::<Vec<_>>())?;
    d.set_item(
        "best_window",
        rows.iter().map(|r| r.best_window).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "best_center",
        rows.iter().map(|r| r.best_center).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "hits_in",
        rows.iter().map(|r| r.hits_in).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "hits_out",
        rows.iter().map(|r| r.hits_out).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "expected_in",
        rows.iter().map(|r| r.expected_in).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "enrichment",
        rows.iter().map(|r| r.enrichment).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "log2_enrichment",
        rows.iter().map(|r| r.log2_enrichment).collect::<Vec<_>>(),
    )?;
    d.set_item("pvalue", rows.iter().map(|r| r.pvalue).collect::<Vec<_>>())?;
    d.set_item("qvalue", rows.iter().map(|r| r.qvalue).collect::<Vec<_>>())?;
    d.set_item(
        "centers",
        rows.iter().map(|r| r.centers.clone()).collect::<Vec<_>>(),
    )?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (merge, height, k, h))]
fn _cutree(
    merge: Vec<[i64; 2]>,
    height: Vec<f64>,
    k: Option<usize>,
    h: Option<f64>,
) -> PyResult<Vec<usize>> {
    let tree = Tree {
        merge,
        height,
        order: Vec::new(),
    };
    match (k, h) {
        (Some(k), None) => tree.cutree(k),
        (None, Some(h)) => tree.cutree_height(h),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "give exactly one of k and h",
            ));
        }
    }
    .map_err(err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_enrich, m)?)?;
    m.add_function(wrap_pyfunction!(_coocc, m)?)?;
    m.add_function(wrap_pyfunction!(_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(_cutree, m)?)?;
    Ok(())
}
