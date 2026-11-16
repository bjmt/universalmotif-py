//! Progress handles, read from Python while a call runs on another thread.

use pyo3::prelude::*;
use universalmotif::progress::{Progress, with_progress};

/// A handle that long-running functions report their progress to.
#[pyclass(name = "Progress", module = "universalmotif._core", frozen)]
pub(crate) struct PyProgress(Progress);

#[pymethods]
impl PyProgress {
    #[new]
    fn new() -> Self {
        PyProgress(Progress::new())
    }

    /// The current stage, such as "scanning".
    #[getter]
    fn stage(&self) -> String {
        self.0.stage()
    }

    /// Units of work done in the current stage.
    #[getter]
    fn done(&self) -> u64 {
        self.0.done()
    }

    /// Units of work in the current stage, or 0 when not known ahead.
    #[getter]
    fn total(&self) -> u64 {
        self.0.total()
    }

    /// Ask the running function to stop.
    fn cancel(&self) {
        self.0.cancel();
    }
}

/// Call `f` with `progress` as the handle that the functions it calls on
/// this thread report to.
#[pyfunction]
fn _with_progress<'py>(
    progress: &PyProgress,
    f: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    with_progress(&progress.0, || f.call0())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProgress>()?;
    m.add_function(wrap_pyfunction!(_with_progress, m)?)?;
    Ok(())
}
