//! Bindings used by the plotting functions.

use pyo3::prelude::*;
use universalmotif::compare::{CompareOptions, anchor_layout};
use universalmotif::convert::{convert_type, ppm_to_icm, ppm_to_pcm, ppm_to_pwm};
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
/// aligned to `motifs[anchor]`, and the width of the shared frame.
#[pyfunction]
fn _anchor_layout(
    py: Python<'_>,
    motifs: Vec<PyRef<'_, PyMotif>>,
    anchor: usize,
    min_overlap: usize,
    rc: bool,
    nthreads: usize,
) -> PyResult<(Vec<(usize, bool)>, usize)> {
    let motifs: Vec<Motif> = motifs.iter().map(|m| m.inner.clone()).collect();
    let opts = CompareOptions {
        min_overlap,
        rc,
        nthreads,
        ..Default::default()
    };
    let layout = py
        .detach(|| anchor_layout(&motifs, anchor, &opts))
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
