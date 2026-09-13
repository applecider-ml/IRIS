"""
iris.processing
===============
Spectral preprocessing pipeline: load, deredshift, interpolate, scale.

Usage:
    from iris.processing import batch_process, process_single_spectrum, process_spectrum_array

    out = batch_process("fritz_data_clean", "spectra_pt", label_df=my_df)
"""

import os
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
WL_MIN     = 3850
WL_MAX     = 9000
INTERP_LEN = 4096

PHASE_MISSING_VALUE = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# SPECTRAL PROCESSING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def remove_continuum(flux: np.ndarray, window: int = 401) -> np.ndarray:
    """Remove continuum via rolling median. Returns flux / continuum."""
    s    = pd.Series(flux)
    cont = s.rolling(window=window, center=True,
                     min_periods=max(5, window // 4)).median()
    cont = cont.interpolate(limit_direction="both").to_numpy()
    cont = np.where(np.abs(cont) < 1e-30, 1e-30, cont)
    return flux / cont


def robust_scale(flux: np.ndarray) -> np.ndarray:
    """Percentile-based scaling to ~[0, 1] using p5/p95."""
    p5, p95 = np.percentile(flux, [5, 95])
    return (flux - p5) / ((p95 - p5) + 1e-30)


def zscore_scale(flux: np.ndarray) -> np.ndarray:
    """Standard z-score normalization: (flux - mean) / std."""
    m = np.nanmean(flux)
    s = np.nanstd(flux)
    if s < 1e-30:
        return flux - m
    return (flux - m) / s


# ──────────────────────────────────────────────────────────────────────────────
# METADATA HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _safe_float(val):
    """Convert to float, return None if NaN/invalid."""
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _parse_metadata_row(row: pd.Series) -> dict:
    """
    Extract metadata from a label DataFrame row.

    Returns dict with: redshift, phase, instrument, class, taxonomy_I, taxonomy_II
    """
    redshift = _safe_float(row.get("redshift", None))

    phase = None
    for col in ("phase", "phase_phot", "phase_snid", "snid_age"):
        phase = _safe_float(row.get(col, None))
        if phase is not None:
            break

    instrument = str(row.get("instrument_name", "unknown")).strip()
    if not instrument or instrument.lower() in ("nan", "none", ""):
        instrument = "unknown"

    latest_classification = str(row.get("latest_classification", "unknown")).strip()
    if not latest_classification or latest_classification.lower() in ("nan", "none", ""):
        latest_classification = "unknown"

    taxonomy_I = str(row.get("taxonomy_I", "unknown")).strip()
    if not taxonomy_I or taxonomy_I.lower() in ("nan", "none", ""):
        taxonomy_I = "unknown"

    taxonomy_II = str(row.get("taxonomy_II", "unknown")).strip()
    if not taxonomy_II or taxonomy_II.lower() in ("nan", "none", ""):
        taxonomy_II = "unknown"

    return {
        "redshift":    redshift,
        "phase":       phase,
        "instrument":  instrument,
        "class":       latest_classification,
        "taxonomy_I":  taxonomy_I,
        "taxonomy_II": taxonomy_II,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE SPECTRUM PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def process_spectrum_array(
    wavelength,
    flux,
    wavelength_range: tuple = (WL_MIN, WL_MAX),
    interp_len: int = INTERP_LEN,
    remove_cont: bool = True,
    scale_method: str = "robust",
) -> "np.ndarray | None":
    """
    Core preprocessing on in-memory arrays: window, interpolate, (optionally)
    remove continuum, scale.

    This is the single source of truth for the IRIS transform; both the file-based
    `process_single_spectrum` and downstream consumers (e.g. the ASTRAnet inference
    package) call it, so a spectrum is transformed identically whether it comes
    from a `.dat` file or straight from the SkyPortal API.

    Returns the processed flux on the uniform grid (float32, length `interp_len`),
    or None if the spectrum is rejected (too few points, <80% coverage, non-finite).
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux       = np.asarray(flux, dtype=float)

    mask_finite = np.isfinite(wavelength) & np.isfinite(flux)
    wavelength  = wavelength[mask_finite]
    flux        = flux[mask_finite]

    if len(wavelength) < 10:
        return None

    wl_min, wl_max = wavelength_range
    mask = (wavelength >= wl_min) & (wavelength <= wl_max)
    if mask.sum() < 10:
        return None

    grid = np.linspace(wl_min, wl_max, interp_len)
    fx_grid = interp1d(wavelength[mask], flux[mask], kind="linear",
                       bounds_error=False, fill_value=np.nan)(grid)

    if np.isnan(fx_grid).mean() > 0.20:
        return None

    fx_grid = pd.Series(fx_grid).ffill().bfill().to_numpy(dtype=np.float32)

    if not np.isfinite(fx_grid).all():
        return None

    if remove_cont:
        win = max(5, (interp_len // 10) | 1)
        fx_grid = remove_continuum(fx_grid, window=win)

    scaler = robust_scale if scale_method == "robust" else zscore_scale
    fx_grid = scaler(fx_grid).astype(np.float32)

    if not np.isfinite(fx_grid).all():
        return None

    return fx_grid


def process_single_spectrum(
    file_path: str,
    spectra_dir: str,
    wavelength_range: tuple = (WL_MIN, WL_MAX),
    interp_len: int = INTERP_LEN,
    redshift: float = None,
    phase: float = None,
    instrument: str = "unknown",
    label_class: str = "unknown",
    taxonomy_I: str = "unknown",
    taxonomy_II: str = "unknown",
    remove_cont: bool = True,
    scale_method: str = "robust",
) -> "dict | None":
    """
    Process a single spectrum file: load, interpolate, normalize.

    Thin file-reading wrapper around `process_spectrum_array`.
    Returns dict with 'file_path', 'flux', 'metadata' or None on failure.
    """
    full_path = os.path.join(spectra_dir, file_path)
    if not os.path.exists(full_path):
        return None

    try:
        raw = pd.read_csv(
            full_path, comment="#", sep=r"\s+",
            engine="python", header=None, usecols=[0, 1],
        )
        wavelength = pd.to_numeric(raw.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        flux       = pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    except Exception:
        return None

    fx_grid = process_spectrum_array(
        wavelength, flux,
        wavelength_range=wavelength_range, interp_len=interp_len,
        remove_cont=remove_cont, scale_method=scale_method,
    )
    if fx_grid is None:
        return None

    return {
        "file_path": file_path,
        "flux":      fx_grid,
        "metadata": {
            "redshift":    redshift,
            "phase":       phase,
            "instrument":  instrument,
            "class":       label_class,
            "taxonomy_I":  taxonomy_I,
            "taxonomy_II": taxonomy_II,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def batch_process(
    spectra_dir_good: str,
    output_dir: str,
    label_df = None,
    label_path = None,
    wavelength_range: tuple = (WL_MIN, WL_MAX),
    interp_len: int = INTERP_LEN,
    remove_cont: bool = True,
    scale_method: str = "robust",
) -> "str | None":
    """
    Process all spectrum files and save a .pt file.

    Output .pt contains: obj_id, file_path, flux, redshift, phase,
    instrument, class, taxonomy_I, taxonomy_II, num_samples, wl_range,
    interp_len, remove_cont.
    """
    os.makedirs(output_dir, exist_ok=True)

    if label_df is not None:
        df = label_df.copy()
    elif label_path is not None:
        df = pd.read_csv(label_path)
    else:
        raise ValueError("Provide either label_df or label_path.")

    if "obj_id" not in df.columns:
        raise ValueError("Label DataFrame must contain an 'obj_id' column.")

    df = df.set_index("obj_id")

    phase_cols = [c for c in ("phase", "phase_phot", "phase_snid", "snid_age") if c in df.columns]
    has_z     = "redshift" in df.columns
    has_inst  = "instrument_name" in df.columns
    has_class = "latest_classification" in df.columns
    has_taxI  = "taxonomy_I" in df.columns
    has_taxII = "taxonomy_II" in df.columns
    print(f"Redshift column       : {'redshift' if has_z else 'MISSING — z=0 for all'}")
    print(f"Phase columns         : {phase_cols or 'MISSING — phase=None for all'}")
    print(f"Instrument column     : {'instrument_name' if has_inst else 'MISSING — unknown for all'}")
    print(f"Classification column : {'latest_classification' if has_class else 'MISSING — unknown for all'}")
    print(f"Taxonomy I column     : {'taxonomy_I' if has_taxI else 'MISSING — unknown for all'}")
    print(f"Taxonomy II column    : {'taxonomy_II' if has_taxII else 'MISSING — unknown for all'}")

    all_files = sorted(
        f for f in os.listdir(spectra_dir_good)
        if f.endswith(".csv") or f.endswith(".dat")
    )
    print(f"Spectrum files found: {len(all_files)}")

    all_files = [
        f for f in all_files
        if os.path.splitext(f)[0].split("_")[0] in df.index
    ]
    print(f"Spectrum files matching label_df: {len(all_files)}")

    processed, failed = [], []

    for file_name in tqdm(all_files, desc="Processing spectra"):
        obj_id = os.path.splitext(file_name)[0].split("_")[0]

        if obj_id in df.index:
            row = df.loc[obj_id]
            row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
            meta = _parse_metadata_row(row)
        else:
            meta = {
                "redshift": None, "phase": None, "instrument": "unknown",
                "class": "unknown", "taxonomy_I": "unknown", "taxonomy_II": "unknown",
            }

        result = process_single_spectrum(
            file_path=file_name,
            spectra_dir=spectra_dir_good,
            wavelength_range=wavelength_range,
            interp_len=interp_len,
            redshift=meta["redshift"],
            phase=meta["phase"],
            instrument=meta["instrument"],
            label_class=meta["class"],
            taxonomy_I=meta["taxonomy_I"],
            taxonomy_II=meta["taxonomy_II"],
            remove_cont=remove_cont,
            scale_method=scale_method,
        )

        if result is not None:
            result["obj_id"] = obj_id
            processed.append(result)
        else:
            failed.append(file_name)

    print(f"\nProcessing complete:")
    print(f"  Success : {len(processed)}")
    print(f"  Failed  : {len(failed)}")

    if not processed:
        print("Warning: No spectra were successfully processed.")
        return None

    obj_ids      = [r["obj_id"]                  for r in processed]
    file_paths   = [r["file_path"]               for r in processed]
    flux         = torch.stack([torch.from_numpy(r["flux"]) for r in processed])
    redshifts    = [r["metadata"]["redshift"]     for r in processed]
    phases       = [r["metadata"]["phase"]        for r in processed]
    instruments  = [r["metadata"]["instrument"]   for r in processed]
    latest_class = [r["metadata"]["class"]        for r in processed]
    taxI         = [r["metadata"]["taxonomy_I"]   for r in processed]
    taxII        = [r["metadata"]["taxonomy_II"]  for r in processed]

    n       = len(processed)
    n_z     = sum(z is not None for z in redshifts)
    n_phase = sum(p is not None for p in phases)
    n_class = sum(c != "unknown" for c in latest_class)
    inst_counts  = pd.Series(instruments).value_counts().to_dict()
    class_counts = pd.Series(latest_class).value_counts().to_dict()
    print(f"\nMetadata coverage:")
    print(f"  redshift  known : {n_z}/{n} ({100*n_z/n:.1f}%)")
    print(f"  phase     known : {n_phase}/{n} ({100*n_phase/n:.1f}%)")
    print(f"  class     known : {n_class}/{n} ({100*n_class/n:.1f}%)")
    print(f"  instruments     : {inst_counts}")
    print(f"  classes         : {class_counts}")

    wl_min, wl_max = wavelength_range
    tag      = f"{wl_min}_{wl_max}_{'cont' if remove_cont else 'raw'}_{scale_method}_{interp_len}"
    out_path = os.path.join(output_dir, f"processed_spectra_{tag}.pt")

    torch.save({
        "obj_id":      obj_ids,
        "file_path":   file_paths,
        "flux":        flux,
        "redshift":    redshifts,
        "phase":       phases,
        "instrument":  instruments,
        "class":       latest_class,
        "taxonomy_I":  taxI,
        "taxonomy_II": taxII,
        "num_samples": n,
        "wl_range":    wavelength_range,
        "interp_len":  interp_len,
        "remove_cont": remove_cont,
    }, out_path)

    print(f"\nSaved -> {out_path}")
    print(f"  flux shape : {flux.shape}")

    if failed:
        failed_path = os.path.join(output_dir, "failed_files.csv")
        pd.DataFrame({"file_path": failed}).to_csv(failed_path, index=False)
        print(f"  failed list -> {failed_path}")

    return out_path
