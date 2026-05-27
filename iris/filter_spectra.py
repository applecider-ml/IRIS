"""
iris.filter_spectra
===================
Spectral filtering: safe classification, deduplication, taxonomy normalization.

Usage:
    from iris.filter_spectra import filter_spectra

    df_filtered = filter_spectra("metadata.csv", "filtered.csv")
"""

import argparse
import numpy as np
import pandas as pd


# ============================================================
# TAXONOMY (timedomain-taxonomy / skyportal)
# ============================================================

TAXONOMY = {
    # Type Ia
    "Ia": ("Supernova", "Ia"), "Ia-norm": ("Supernova", "Ia"),
    "Ia-pec": ("Supernova", "Ia-pec"), "Ia-91T": ("Supernova", "Ia-91T"),
    "Ia-91bg": ("Supernova", "Ia-91bg"), "Ia-02cx": ("Supernova", "Ia-02cx"),
    "Ia-02es": ("Supernova", "Ia-02es"), "Ia-CSM": ("Supernova", "Ia-CSM"),
    "Ia-SC": ("Supernova", "Ia-SC"),
    # Type Ib/c
    "Ib": ("Supernova", "Ib"), "Ic": ("Supernova", "Ic"),
    "Ic-BL": ("Supernova", "Ic-BL"), "Ib/c": ("Supernova", "Ib/c"),
    "Ibn": ("Supernova", "Ibn"), "Icn": ("Supernova", "Icn"),
    # Type II
    "II": ("Supernova", "II"), "Type II": ("Supernova", "II"),
    "IIP": ("Supernova", "IIP"), "IIL": ("Supernova", "IIL"),
    "IIn": ("Supernova", "IIn"), "IIb": ("Supernova", "IIb"),
    # SLSNe
    "SLSN-I": ("Supernova", "SLSN-I"), "SLSN-II": ("Supernova", "SLSN-II"),
    "I-SLSN": ("Supernova", "SLSN-I"), "II-SLSN": ("Supernova", "SLSN-II"),
    # TDE
    "Tidal Disruption Event": ("TDE", "TDE"), "TDE": ("TDE", "TDE"),
    "TDE-H": ("TDE", "TDE-H"), "TDE-He": ("TDE", "TDE-He"),
    "TDE-H+He": ("TDE", "TDE-H+He"),
    # AGN / Nuclear
    "AGN": ("AGN", "AGN"), "QSO": ("AGN", "QSO"), "Blazar": ("AGN", "Blazar"),
    # GRB / afterglow
    "afterglow": ("GRB", "afterglow"), "long GRB": ("GRB", "long GRB"),
    "short GRB": ("GRB", "short GRB"),
    # Variable stars
    "CV": ("Variable", "CV"), "Nova": ("Variable", "Nova"),
    "LBV": ("Variable", "LBV"), "RR Lyrae": ("Variable", "RR Lyrae"),
    "Cepheid": ("Variable", "Cepheid"),
    # Other
    "galaxy": ("Galaxy", "galaxy"), "star": ("Star", "star"),
}

ALIASES = {k.lower(): k for k in TAXONOMY}
ALIASES.update({
    "snia": "Ia", "snib": "Ib", "snic": "Ic", "snic-bl": "Ic-BL",
    "snii": "II", "sniip": "IIP", "sniin": "IIn", "sniib": "IIb",
    "tde": "Tidal Disruption Event", "type ii": "II", "type i": "Ia",
    "ia-91t": "Ia-91T", "ia-91bg": "Ia-91bg",
})


def normalize_class(raw):
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if s in TAXONOMY:
        return s
    sl = s.lower()
    if sl in ALIASES:
        return ALIASES[sl]
    return None


def get_taxonomy_info(canonical):
    if canonical in TAXONOMY:
        return TAXONOMY[canonical]
    return ("Unknown", canonical)


def is_safe_classification(row):
    cls  = row.get("latest_classification")
    prob = row.get("latest_class_prob")
    if pd.isna(cls) or not isinstance(cls, str):
        return False
    if "|" in cls:
        return False
    if pd.notna(prob) and float(prob) < 0.5:
        return False
    if normalize_class(cls) is None:
        return False
    return True


def pick_best_spectrum(group):
    if "snr_proxy" in group.columns and group["snr_proxy"].notna().any():
        return group.loc[group["snr_proxy"].idxmax()]
    return group.iloc[0]


def remove_duplicates(df, time_window_days=0.5):
    results = []
    for obj_id, obj_df in df.groupby("obj_id"):
        obj_df = obj_df.sort_values("observed_at_mjd").reset_index(drop=True)
        if len(obj_df) == 1:
            results.append(obj_df)
            continue

        used = [False] * len(obj_df)
        for i in range(len(obj_df)):
            if used[i]:
                continue
            mjd_i = obj_df.loc[i, "observed_at_mjd"]
            group_idx = [i]
            for j in range(i + 1, len(obj_df)):
                if used[j]:
                    continue
                mjd_j = obj_df.loc[j, "observed_at_mjd"]
                if pd.notna(mjd_i) and pd.notna(mjd_j):
                    if abs(mjd_j - mjd_i) <= time_window_days:
                        group_idx.append(j)
                        used[j] = True
            used[i] = True
            group = obj_df.loc[group_idx]
            best = pick_best_spectrum(group)
            results.append(best.to_frame().T)

    return pd.concat(results, ignore_index=True)


def filter_spectra(metadata_csv, output_csv, time_window_days=0.5, min_probability=0.5):
    """Main filtering pipeline."""
    df = pd.read_csv(metadata_csv)
    print(f"Total spectra loaded:       {len(df)}")

    df["is_safe"] = df.apply(is_safe_classification, axis=1)
    df_safe = df[df["is_safe"]].copy()
    print(f"After safe classification:  {len(df_safe)}")

    df_safe["canonical_class"] = df_safe["latest_classification"].apply(normalize_class)
    df_safe[["broad_class", "tax_label"]] = df_safe["canonical_class"].apply(
        lambda c: pd.Series(get_taxonomy_info(c) if c else ("Unknown", "Unknown"))
    )

    class_per_obj = df_safe.groupby("obj_id")["canonical_class"].nunique()
    unique_obj = class_per_obj[class_per_obj == 1].index
    df_unique = df_safe[df_safe["obj_id"].isin(unique_obj)].copy()
    print(f"After unique class/object:  {len(df_unique)}")

    df_dedup = remove_duplicates(df_unique, time_window_days=time_window_days)
    print(f"After duplicate removal:    {len(df_dedup)}")

    print("\nClass distribution:")
    print(df_dedup.groupby(["broad_class", "tax_label"]).size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
          .to_string(index=False))

    df_dedup.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")
    return df_dedup


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--time-window", type=float, default=0.5)
    p.add_argument("--min-prob",    type=float, default=0.5)
    args = p.parse_args()
    filter_spectra(args.metadata, args.output, args.time_window, args.min_prob)
