"""
iris.fritz.data_processing
==========================
Pure functions for transforming raw Fritz API data into structured records.
"""

from datetime import datetime
from astropy.time import Time


def to_mjd(obs) -> "float | None":
    if obs is None:
        return None
    if isinstance(obs, (int, float)):
        return float(obs)
    return float(Time(obs).mjd)


def peak_mjd_from_mag(phot: list, preferred_filter: str = "ztfr") -> "float | None":
    """Return MJD of brightest (lowest mag) detection."""
    if not phot:
        return None

    det = [p for p in phot if p.get("mag") is not None and p.get("filter") == preferred_filter]
    if not det:
        det = [p for p in phot if p.get("mag") is not None]
    if not det:
        return None

    return float(min(det, key=lambda x: x["mag"])["mjd"])


def extract_snid(altdata: dict) -> dict:
    if not isinstance(altdata, dict):
        return {}
    return {
        "snid_age":      altdata.get("SNIDAGEM"),
        "snid_age_err":  altdata.get("SNIDAGEMERR"),
        "snid_redshift": altdata.get("SNIDZMED"),
    }


def get_latest_classification(classifications: list) -> "dict | None":
    if not classifications:
        return None
    latest = max(classifications, key=lambda c: datetime.fromisoformat(c["created_at"]))
    return {
        "classification": latest.get("classification"),
        "probability":    latest.get("probability"),
    }


def build_spectrum_record(spectrum: dict, obj_id: str, peak_mjd: float | None) -> dict:
    obs_mjd   = to_mjd(spectrum.get("observed_at"))
    snid      = extract_snid(spectrum.get("altdata"))
    phase_phot = float(obs_mjd - peak_mjd) if (obs_mjd is not None and peak_mjd is not None) else None
    phase_snid = snid.get("snid_age") if isinstance(snid.get("snid_age"), (int, float)) else None

    return {
        "obj_id":          obj_id,
        "spectrum_id":     spectrum.get("id"),
        "instrument_name": spectrum.get("instrument_name"),
        "observed_at":     spectrum.get("observed_at"),
        "observed_at_mjd": obs_mjd,
        "wavelengths":     spectrum.get("wavelengths"),
        "fluxes":          spectrum.get("fluxes"),
        "errors":          spectrum.get("errors") or spectrum.get("fluxerrs"),
        "phase":           phase_phot,
        "phase_source":    "photometry" if phase_phot is not None else None,
        "phase_phot":      phase_phot,
        "phase_snid":      phase_snid,
        **snid,
    }


def build_object_result(source: dict, phot: list, spectra: list, obj_id: str) -> dict:
    """Combine raw API responses into a single structured dict."""
    peak_mjd = peak_mjd_from_mag(phot, preferred_filter="ztfr")
    source_z = source.get("redshift")
    classes  = source.get("classifications", [])

    meta = {
        "obj_id":                obj_id,
        "redshift":              source_z,
        "peak_mjd":              peak_mjd,
        "n_spectra":             len(spectra),
        "unique_classes":        list({c.get("classification") for c in classes if c.get("classification")}),
        "latest_classification": get_latest_classification(classes),
        "classifications": [
            {"classification": c.get("classification"), "probability": c.get("probability")}
            for c in classes
        ],
    }

    spec_records = [build_spectrum_record(s, obj_id, peak_mjd) for s in spectra]
    return {"meta": meta, "spectra": spec_records}
