//! Bindings used by the plotting functions.

use pyo3::prelude::*;
use universalmotif::compare::{CompareOptions, anchor_layout};
use universalmotif::convert::{convert_type, ppm_to_icm, ppm_to_pcm, ppm_to_pwm};
use universalmotif::similarity::{CwmCompareOptions, CwmMetric, cwm_anchor_layout};
use universalmotif::{Motif, MotifType};

use crate::{PyMotif, err};

/// The matrix a motif is drawn with, as the columns of its PPM converted to
/// `use_type`. Empty columns stay empty, and a missing or single `nsites`
/// counts as 100.
#[pyfunction]
fn _display_matrix(
    motif: PyRef<'_, PyMotif>,
    use_type: &str,
    relative_entropy: bool,
) -> PyResult<Vec<Vec<f64>>> {
    let ppm = convert_type(&motif.inner, MotifType::Ppm, None, false).map_err(err)?;
    let nsites = ppm.nsites.filter(|&n| n != 1.0).unwrap_or(100.0);
    let bkg = &ppm.bkg;
    ppm.matrix
        .columns()
        .map(|col| {
            if col.iter().all(|&v| v == 0.0) {
                return Ok(col.to_vec());
            }
            Ok(match use_type {
                "PPM" => col.to_vec(),
                "PCM" => ppm_to_pcm(col, nsites),
                "PWM" => ppm_to_pwm(col, bkg, ppm.pseudocount, nsites),
                "ICM" => ppm_to_icm(col, bkg, relative_entropy),
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "use_type must be one of PCM, PPM, PWM and ICM",
                    ));
                }
            })
        })
        .collect()
}

/// Placements (left padding and whether to reverse complement) of motifs
/// aligned to `motifs[anchor]`, and the width of the shared frame. CWMs are
/// aligned by their similarity, with `metric` and `ignore_sign` as in
/// comparisons, and other motifs by column correlation.
#[pyfunction]
#[pyo3(signature = (motifs, anchor, min_overlap, rc, nthreads, metric=None, ignore_sign=false))]
#[allow(clippy::too_many_arguments)]
fn _anchor_layout(
    py: Python<'_>,
    motifs: Vec<PyRef<'_, PyMotif>>,
    anchor: usize,
    min_overlap: usize,
    rc: bool,
    nthreads: usize,
    metric: Option<&str>,
    ignore_sign: bool,
) -> PyResult<(Vec<(usize, bool)>, usize)> {
    let motifs: Vec<Motif> = motifs.iter().map(|m| m.inner.clone()).collect();
    let cwms = motifs.iter().all(|m| m.motif_type == MotifType::Cwm);
    let layout = match metric {
        Some(metric) if cwms => {
            let opts = CwmCompareOptions {
                metric: metric.parse::<CwmMetric>().map_err(err)?,
                ignore_sign,
                min_overlap,
                rc,
                nthreads,
            };
            py.detach(|| cwm_anchor_layout(&motifs, anchor, &opts))
        }
        _ => {
            let opts = CompareOptions {
                min_overlap,
                rc,
                nthreads,
                ..Default::default()
            };
            py.detach(|| anchor_layout(&motifs, anchor, &opts))
        }
    }
    .map_err(err)?;
    Ok((
        layout.placements.iter().map(|p| (p.left, p.rc)).collect(),
        layout.width,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_display_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_anchor_layout, m)?)?;
    Ok(())
}
