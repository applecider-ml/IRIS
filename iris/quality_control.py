"""
iris.quality_control
====================
Spectral quality control metrics and filtering.

Usage:
    from iris.quality_control import compute_qc, apply_qc_filter

    df_qc = compute_qc(df, spectra_dir="fritz_data_clean")
    df_pass, df_fail = apply_qc_filter(df_qc)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional


# =============================================================================
# METRIC COLUMNS
# =============================================================================

METRIC_COLS = [
    "N_wl", "coverage", "nan_frac",
    "snr_proxy", "res_proxy", "flux_dyn", "spec_entropy",
    "snr_continuum", "snr_structure", "flatness",
]

_NAN_RECORD = {c: np.nan for c in METRIC_COLS}
_BAD_RECORD = {
    "N_wl": 0, "coverage": 0.0, "nan_frac": 1.0,
    "snr_proxy": 0.0, "res_proxy": 0.0, "flux_dyn": 0.0,
    "spec_entropy": 0.0, "snr_continuum": 0.0, "snr_structure": 0.0,
    "flatness": 1.0,
}

DEFAULT_QC = {
    "snr_structure_min": 2.0,
    "flatness_max": 50.0,
    "nan_frac_max": 0.1,
    "coverage_min": 0.5 * (9000 - 3850),
    "N_wl_min": 30,
}


# =============================================================================
# SPECTRUM LOADER
# =============================================================================

def load_dat_spectrum(filepath):
    """Load a .dat spectrum file -> (wavelength, flux) arrays."""
    try:
        df = pd.read_csv(
            filepath, comment="#", sep=r"\s+|,",
            engine="python", header=None, usecols=[0, 1],
        )
        wl = df.iloc[:, 0].to_numpy(dtype=float)
        fx = df.iloc[:, 1].to_numpy(dtype=float)
        return wl, fx
    except Exception:
        return None, None


# =============================================================================
# QUALITY METRICS
# =============================================================================

def compute_quality_metrics(wavelength, flux, smooth_win=51) -> dict:
    """Compute quality metrics for a single spectrum."""
    wl = np.asarray(wavelength, dtype=float)
    fx = np.asarray(flux, dtype=float)

    m = np.isfinite(wl) & np.isfinite(fx)
    nan_frac = 1.0 - float(m.sum()) / float(len(m))

    if m.sum() < 5:
        return _BAD_RECORD.copy()

    wl, fx = wl[m], fx[m]

    order = np.argsort(wl)
    wl, fx = wl[order], fx[order]
    wl, idx = np.unique(wl, return_index=True)
    fx = fx[idx]

    N_wl = len(wl)
    if N_wl < 5:
        return {**_BAD_RECORD, "nan_frac": nan_frac}

    coverage = float(wl.max() - wl.min())
    d = np.diff(wl)
    res_proxy = float(np.median(d[d > 0])) if np.any(d > 0) else 0.0

    grid = np.linspace(wl.min(), wl.max(), N_wl)
    fx_grid = np.interp(grid, wl, fx)

    finite_nonzero = np.where(fx_grid != 0)[0]
    if len(finite_nonzero) < 5:
        return {**_BAD_RECORD, "N_wl": N_wl, "coverage": coverage, "nan_frac": nan_frac}

    lo, hi = int(finite_nonzero[0]), int(finite_nonzero[-1]) + 1
    fx_inner = fx_grid[lo:hi]
    N_inner = len(fx_inner)

    p95_abs = np.percentile(np.abs(fx_inner), 95) + 1e-30
    fxn = fx_inner / p95_abs

    from scipy.signal import savgol_filter
    sg_win = int(smooth_win)
    if sg_win % 2 == 0:
        sg_win += 1
    sg_win = max(7, min(sg_win, N_inner if N_inner % 2 == 1 else N_inner - 1))
    sg_order = min(3, sg_win - 2)
    cont = savgol_filter(fxn, window_length=sg_win, polyorder=sg_order)
    resid = fxn - cont

    q25, q75 = np.percentile(resid, [25, 75])
    noise_est = (q75 - q25) / 1.35

    if not np.isfinite(noise_est) or noise_est < 1e-12:
        return {**_BAD_RECORD, "N_wl": N_wl, "coverage": coverage, "nan_frac": nan_frac}

    r5, r95 = np.percentile(resid, [5, 95])
    snr_structure = (r95 - r5) / noise_est
    snr_continuum = float(np.abs(np.median(fxn)) / noise_est)
    snr_proxy = snr_structure

    diff_resid = np.diff(resid)
    flatness = float(np.std(diff_resid) / noise_est)

    flux_dyn = float(
        np.percentile(np.abs(resid), 95) - np.percentile(np.abs(resid), 5)
    )

    power = resid ** 2
    power /= (power.sum() + 1e-30)
    spec_entropy = float(-np.sum(power * np.log(power + 1e-30)))

    return {
        "N_wl": N_wl, "coverage": coverage, "nan_frac": nan_frac,
        "snr_proxy": snr_proxy, "res_proxy": res_proxy,
        "flux_dyn": flux_dyn, "spec_entropy": spec_entropy,
        "snr_continuum": snr_continuum, "snr_structure": snr_structure,
        "flatness": flatness,
    }


# =============================================================================
# BATCH QC COMPUTATION
# =============================================================================

def _process_row(args):
    """Worker for parallel QC computation."""
    row, spectra_dir, dat_col = args
    dat_path = Path(spectra_dir) / row[dat_col]

    if not dat_path.exists():
        return _NAN_RECORD.copy()

    wl, fx = load_dat_spectrum(dat_path)
    if wl is None or len(wl) < 5:
        return _NAN_RECORD.copy()

    return compute_quality_metrics(wl, fx, smooth_win=51)


def compute_qc(
    df: pd.DataFrame,
    spectra_dir: str,
    dat_col: str = "dat_file",
    n_workers: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute quality metrics for all spectra in a DataFrame."""
    args = [(row, spectra_dir, dat_col) for _, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        iterator = executor.map(_process_row, args)
        if verbose:
            iterator = tqdm(iterator, total=len(df), desc="Computing QC metrics")
        records = list(iterator)

    metrics_df = pd.DataFrame(records, index=df.index)
    result = pd.concat([df, metrics_df], axis=1)

    if verbose:
        print(f"\nQC summary:")
        print(result[METRIC_COLS].describe().round(2).to_string())

    return result


# =============================================================================
# QC FILTERING
# =============================================================================

def apply_qc_filter(
    df: pd.DataFrame,
    thresholds: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply quality cuts. Returns (df_pass, df_fail)."""
    t = {**DEFAULT_QC, **(thresholds or {})}

    mask = (
        (df["snr_structure"] >= t["snr_structure_min"]) &
        (df["flatness"] <= t["flatness_max"]) &
        (df["nan_frac"] <= t["nan_frac_max"]) &
        (df["coverage"] >= t["coverage_min"]) &
        (df["N_wl"] >= t["N_wl_min"])
    )

    df_pass = df[mask].copy()
    df_fail = df[~mask].copy()

    if verbose:
        print(f"\nQC filter results:")
        print(f"  Pass: {len(df_pass):6d} ({100*len(df_pass)/len(df):.1f}%)")
        print(f"  Fail: {len(df_fail):6d} ({100*len(df_fail)/len(df):.1f}%)")
        print(f"\n  Thresholds used:")
        for k, v in t.items():
            print(f"    {k:25s}: {v}")
        print()

        criteria = {
            "snr_structure": df["snr_structure"] < t["snr_structure_min"],
            "flatness":      df["flatness"] > t["flatness_max"],
            "nan_frac":      df["nan_frac"] > t["nan_frac_max"],
            "coverage":      df["coverage"] < t["coverage_min"],
            "N_wl":          df["N_wl"] < t["N_wl_min"],
        }
        print("  Failures per criterion:")
        for name, fails in criteria.items():
            print(f"    {name:25s}: {fails.sum():6d}")

    return df_pass, df_fail
