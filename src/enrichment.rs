//! Bindings for enrichment, co-occurrence, positional enrichment and tree
//! cutting.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;
use universalmotif::Motif;
use universalmotif::cluster::Tree;
use universalmotif::coocc::{self, CooccHit, CooccOptions};
use universalmotif::enrich::{self, EnrichOptions, EnrichTest};
use universalmotif::peaks::{self, PeakHit, PeaksMode, PeaksOptions};

use pyo3_arrow::PyTable;

use crate::arrow::{i64s, opt_f64s, opt_i64s, strs, table};
use crate::{PyMotif, err};

#[pyfunction]
#[pyo3(signature = (motifs, sequences, background, pvalue, test, rc, shuffle_k, seed, pseudocount, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _enrich<'py>(
    py: Python<'py>,
    motifs: Vec<PyRef<'py, PyMotif>>,
    sequences: Vec<PyBackedStr>,
    background: Option<Vec<PyBackedStr>>,
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
        test: test.parse::<EnrichTest>().map_err(err)?,
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
    d.set_item(
        "log10_pvalue",
        rows.iter().map(|r| r.log10_pvalue).collect::<Vec<_>>(),
    )?;
    d.set_item("qvalue", rows.iter().map(|r| r.qvalue).collect::<Vec<_>>())?;
    d.set_item(
        "log10_qvalue",
        rows.iter().map(|r| r.log10_qvalue).collect::<Vec<_>>(),
    )?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (motif, sequence, start, n_motifs, n_sequences, max_distance, min_coocc, pseudocount, self_pairs, nthreads, names))]
#[allow(clippy::too_many_arguments)]
fn _coocc(
    py: Python<'_>,
    motif: PyReadonlyArray1<i64>,
    sequence: PyReadonlyArray1<i64>,
    start: PyReadonlyArray1<i64>,
    n_motifs: usize,
    n_sequences: usize,
    max_distance: Option<u64>,
    min_coocc: u64,
    pseudocount: u64,
    self_pairs: bool,
    nthreads: usize,
    names: Vec<String>,
) -> PyResult<PyTable> {
    let (motif, sequence, start) = (motif.as_slice()?, sequence.as_slice()?, start.as_slice()?);
    let hits: Vec<CooccHit> = (0..motif.len())
        .map(|i| CooccHit {
            motif: motif[i] as usize,
            sequence: sequence[i] as usize,
            start: start.get(i).copied().unwrap_or(0),
        })
        .collect();
    let opts = CooccOptions {
        max_distance,
        min_coocc,
        pseudocount,
        self_pairs,
        nthreads,
    };
    let rows = py
        .detach(|| coocc::motif_coocc(&hits, n_motifs, n_sequences, &opts))
        .map_err(err)?;
    let mut cols = vec![
        ("motif_a", strs(rows.iter().map(|r| names[r.a].as_str()))),
        ("motif_b", strs(rows.iter().map(|r| names[r.b].as_str()))),
        ("a_only", i64s(rows.iter().map(|r| r.a_only as i64))),
        ("b_only", i64s(rows.iter().map(|r| r.b_only as i64))),
        ("both", i64s(rows.iter().map(|r| r.both as i64))),
        ("neither", i64s(rows.iter().map(|r| r.neither as i64))),
        ("odds_ratio", opt_f64s(rows.iter().map(|r| r.odds_ratio))),
        ("pvalue", opt_f64s(rows.iter().map(|r| r.pvalue))),
        ("qvalue", opt_f64s(rows.iter().map(|r| r.qvalue))),
    ];
    if max_distance.is_some() {
        cols.push((
            "both_clustered",
            opt_i64s(rows.iter().map(|r| r.both_clustered.map(|x| x as i64))),
        ));
        cols.push((
            "median_distance",
            opt_f64s(rows.iter().map(|r| r.median_distance)),
        ));
    }
    table(cols)
}

#[pyfunction]
#[pyo3(signature = (motif, sequence, centre, score, seq_length, mode, min_window, max_window, window_step, position_step, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _peaks<'py>(
    py: Python<'py>,
    motif: PyReadonlyArray1<i64>,
    sequence: PyReadonlyArray1<i64>,
    centre: PyReadonlyArray1<f64>,
    score: PyReadonlyArray1<f64>,
    seq_length: usize,
    mode: &str,
    min_window: usize,
    max_window: Option<usize>,
    window_step: usize,
    position_step: usize,
    nthreads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let (motif, sequence) = (motif.as_slice()?, sequence.as_slice()?);
    let (centre, score) = (centre.as_slice()?, score.as_slice()?);
    let hits: Vec<PeakHit> = (0..motif.len())
        .map(|i| PeakHit {
            motif: motif[i] as usize,
            sequence: sequence[i] as usize,
            centre: centre[i],
            score: score[i],
        })
        .collect();
    let opts = PeaksOptions {
        mode: mode.parse::<PeaksMode>().map_err(err)?,
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
        "best_centre",
        rows.iter().map(|r| r.best_centre).collect::<Vec<_>>(),
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
        "centres",
        rows.iter().map(|r| r.centres.clone()).collect::<Vec<_>>(),
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
