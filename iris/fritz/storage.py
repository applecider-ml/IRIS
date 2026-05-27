"""
iris.fritz.storage
==================
Persist pipeline results to disk: .dat files and metadata CSV.
"""

import csv
import os


def save_object(result: dict, output_dir: str = "fritz_data") -> "list[str]":
    """
    Save all spectra for one object and append metadata to CSV.

    Parameters
    ----------
    result     : dict returned by build_object_result()
    output_dir : root folder (created if missing)

    Returns
    -------
    List of .dat file paths written.
    """
    os.makedirs(output_dir, exist_ok=True)

    meta      = result["meta"]
    obj_id    = meta["obj_id"]
    n_spectra = meta["n_spectra"]
    dat_paths = []

    for i, spec in enumerate(result["spectra"], start=1):
        waves  = spec.get("wavelengths") or []
        fluxes = spec.get("fluxes")      or []
        errors = spec.get("errors")      or [None] * len(waves)

        fname = f"{obj_id}_{i}.dat"
        fpath = os.path.join(output_dir, fname)

        with open(fpath, "w") as f:
            f.write("# wavelength\tflux\terror\n")
            for w, fl, e in zip(waves, fluxes, errors):
                e_str = str(e) if e is not None else "nan"
                f.write(f"{w}\t{fl}\t{e_str}\n")

        dat_paths.append(fpath)

    csv_path    = os.path.join(output_dir, "metadata.csv")
    file_exists = os.path.isfile(csv_path)

    rows = []
    for i, spec in enumerate(result["spectra"], start=1):
        latest = meta.get("latest_classification") or {}
        row = {
            "obj_id":                obj_id,
            "n_spectra":             n_spectra,
            "dat_file":              f"{obj_id}_{i}.dat",
            "redshift":              meta.get("redshift"),
            "peak_mjd":              meta.get("peak_mjd"),
            "unique_classes":        "|".join(meta.get("unique_classes") or []),
            "latest_classification": latest.get("classification"),
            "latest_class_prob":     latest.get("probability"),
            "spectrum_id":           spec.get("spectrum_id"),
            "instrument_name":      spec.get("instrument_name"),
            "observed_at":          spec.get("observed_at"),
            "observed_at_mjd":      spec.get("observed_at_mjd"),
            "phase":                spec.get("phase"),
            "phase_source":         spec.get("phase_source"),
            "phase_phot":           spec.get("phase_phot"),
            "phase_snid":           spec.get("phase_snid"),
            "snid_age":             spec.get("snid_age"),
            "snid_age_err":         spec.get("snid_age_err"),
            "snid_redshift":        spec.get("snid_redshift"),
        }
        rows.append(row)

    if rows:
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    return dat_paths
