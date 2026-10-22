//! Bindings for proximity to genomic anchors and background matching.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use universalmotif::background::{self, BinType, Covariate, MatchOptions};
use universalmotif::proximity::{
    self, ProximityCount, ProximityHit, ProximityMethod, ProximityOptions,
};
use universalmotif::ranges::{GRange, GStrand, Orientation};

use crate::err;

type RangeCols = (Vec<usize>, Vec<i64>, Vec<i64>, Vec<String>);

fn granges(cols: RangeCols) -> PyResult<Vec<GRange>> {
    let (chrom, start, end, strand) = cols;
    (0..chrom.len())
        .map(|i| {
            Ok(GRange {
                chrom: chrom[i],
                start: start[i],
                end: end[i],
                strand: GStrand::parse(strand.get(i).map_or("*", |s| s.as_str())).map_err(err)?,
            })
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (hits, motif, score, anchors, universe, chrom_lengths, method, count, orientation, dist_thresholds, n_perm, perm_per_chrom, ignore_strand, seed, nthreads))]
#[allow(clippy::too_many_arguments)]
fn _proximity<'py>(
    py: Python<'py>,
    hits: RangeCols,
    motif: Vec<usize>,
    score: Vec<f64>,
    anchors: RangeCols,
    universe: RangeCols,
    chrom_lengths: Vec<Option<i64>>,
    method: &str,
    count: &str,
    orientation: &str,
    dist_thresholds: Option<Vec<i64>>,
    n_perm: usize,
    perm_per_chrom: bool,
    ignore_strand: bool,
    seed: u64,
    nthreads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let hit_ranges = granges(hits)?;
    let hits: Vec<ProximityHit> = hit_ranges
        .into_iter()
        .enumerate()
        .map(|(i, range)| ProximityHit {
            motif: motif[i],
            range,
            score: score[i],
        })
        .collect();
    let anchors = granges(anchors)?;
    let universe = granges(universe)?;
    let opts = ProximityOptions {
        method: ProximityMethod::parse(method).map_err(err)?,
        count: ProximityCount::parse(count).map_err(err)?,
        orientation: Orientation::parse(orientation).map_err(err)?,
        dist_thresholds,
        n_perm,
        perm_per_chrom,
        ignore_strand,
        seed,
        nthreads,
    };
    let rows = py
        .detach(|| proximity::motif_proximity(&hits, &anchors, &universe, &chrom_lengths, &opts))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("motif", rows.iter().map(|r| r.motif).collect::<Vec<_>>())?;
    d.set_item("nhits", rows.iter().map(|r| r.nhits).collect::<Vec<_>>())?;
    d.set_item(
        "best_dist",
        rows.iter().map(|r| r.best_dist).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "hits_near",
        rows.iter().map(|r| r.hits_near).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "hits_far",
        rows.iter().map(|r| r.hits_far).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "frac_near",
        rows.iter().map(|r| r.frac_near).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "expected_near",
        rows.iter().map(|r| r.expected_near).collect::<Vec<_>>(),
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
    d.set_item("ks_d", rows.iter().map(|r| r.ks_d).collect::<Vec<_>>())?;
    d.set_item(
        "dist_to_nearest",
        rows.iter()
            .map(|r| r.dist_to_nearest.clone())
            .collect::<Vec<_>>(),
    )?;
    Ok(d)
}

#[pyfunction]
fn _gc_fraction(seqs: Vec<String>) -> Vec<f64> {
    seqs.iter().map(|s| background::gc_fraction(s)).collect()
}

/// Covariates as (name, target values, universe values, bins, bin type).
type CovariateSpec = (String, Vec<f64>, Vec<f64>, usize, String);

#[pyfunction]
#[pyo3(signature = (targets, universe, covariates, n_per_target, n_bins_gc, n_bins_length, unique, seed))]
#[allow(clippy::too_many_arguments)]
fn _match_bkg(
    py: Python<'_>,
    targets: Vec<String>,
    universe: Vec<String>,
    covariates: Vec<CovariateSpec>,
    n_per_target: usize,
    n_bins_gc: usize,
    n_bins_length: usize,
    unique: bool,
    seed: u64,
) -> PyResult<(Vec<usize>, usize)> {
    let covariates: Vec<Covariate> = covariates
        .into_iter()
        .map(|(name, t, u, n_bins, bt)| {
            Ok(Covariate {
                name,
                targets: t,
                universe: u,
                n_bins,
                bin_type: BinType::parse(&bt).map_err(err)?,
            })
        })
        .collect::<PyResult<_>>()?;
    let opts = MatchOptions {
        n_per_target,
        n_bins_gc,
        n_bins_length,
        unique,
        seed,
    };
    let m = py
        .detach(|| background::match_bkg(&targets, &universe, &covariates, &opts))
        .map_err(err)?;
    Ok((m.universe_index, m.fallback_targets))
}

#[pyfunction]
fn _genome_windows(
    chrom_lengths: Vec<i64>,
    target_widths: Vec<i64>,
    n_candidates: usize,
    exclude: RangeCols,
    seed: u64,
) -> PyResult<Vec<(usize, i64, i64)>> {
    let exclude = granges(exclude)?;
    background::genome_windows(&chrom_lengths, &target_widths, n_candidates, &exclude, seed)
        .map(|v| v.iter().map(|g| (g.chrom, g.start, g.end)).collect())
        .map_err(err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_proximity, m)?)?;
    m.add_function(wrap_pyfunction!(_gc_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(_match_bkg, m)?)?;
    m.add_function(wrap_pyfunction!(_genome_windows, m)?)?;
    Ok(())
}
