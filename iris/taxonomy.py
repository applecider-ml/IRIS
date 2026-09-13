"""
iris.taxonomy
=============
Taxonomy definitions and pipeline for spectral classification.

Usage:
    from iris.taxonomy import (
        FINE_10, COARSE_6,
        apply_taxonomy, split_anomaly_sets,
        clean_metadata, build_all_pt_files,
        validate_taxonomy, print_config,
    )
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Dict, Optional


# =============================================================================
# TAXONOMY DEFINITIONS
# =============================================================================

FINE_10 = {
    "name": "fine_10cls",
    "class_names": [
        "SN_Ia", "SN_II", "SN_IIn", "SN_IIb",
        "SN_Ibc", "SN_Ibn", "SLSN",
        "AGN", "TDE", "CV",
    ],
    "hierarchy": {
        "SN_Thermonuclear": ["SN_Ia"],
        "SN_CC_H":          ["SN_II", "SN_IIn", "SN_IIb"],
        "SN_CC_SE":         ["SN_Ibc", "SN_Ibn"],
        "SN_Luminous":      ["SLSN"],
        "NonSN":            ["AGN", "TDE", "CV"],
    },
    "mapping": {
        # --- SN_Ia ---
        "Ia": "SN_Ia", "Ia-norm": "SN_Ia", "Ia-91T": "SN_Ia",
        "Ia-91bg": "SN_Ia", "Ia-02cx": "SN_Ia", "Ia-03fg": "SN_Ia",
        "Ia-pec": "SN_Ia", "Ia-99aa": "SN_Ia",
        # --- SN_II ---
        "Type II": "SN_II", "II-norm": "SN_II", "IIP": "SN_II",
        "IIL": "SN_II", "II-pec": "SN_II",
        # --- SN_IIn (includes Ia-CSM) ---
        "IIn": "SN_IIn", "Ia-CSM": "SN_IIn",
        # --- SN_IIb ---
        "IIb": "SN_IIb",
        # --- SN_Ibc ---
        "Ib": "SN_Ibc", "Ic": "SN_Ibc", "Ib/c": "SN_Ibc",
        "Ic-BL": "SN_Ibc", "Ib-pec": "SN_Ibc", "Icn": "SN_Ibc",
        "Ien": "SN_Ibc",
        # --- SN_Ibn ---
        "Ibn": "SN_Ibn",
        # --- SLSN ---
        "Ic-SLSN": "SLSN-I", "II-SLSN": "SLSN-II", "SLSN I": "SLSN-I",
        "Ic.5-SLSN": "SLSN-I",
        # --- AGN ---
        "AGN": "AGN", "Seyfert": "AGN", "Galactic Nuclei": "AGN",
        # --- TDE ---
        "Tidal Disruption Event": "TDE",
        # --- CV ---
        "Cataclysmic": "CV", "U Gem": "CV", "AM CVn": "CV",
        "Nova-like": "CV",
    },
}

COARSE_6 = {
    "name": "coarse_6cls",
    "class_names": [
        "SN_Ia", "SN_II", "SN_Ibc", "SLSN", "AGN", "NonSN_Other",
    ],
    "hierarchy": {
        "SN_Thermonuclear": ["SN_Ia"],
        "SN_CC":            ["SN_II", "SN_Ibc"],
        "SN_Luminous":      ["SLSN"],
        "NonSN":            ["AGN", "NonSN_Other"],
    },
    "mapping": {
        # --- SN_Ia ---
        "Ia": "SN_Ia", "Ia-norm": "SN_Ia", "Ia-91T": "SN_Ia",
        "Ia-91bg": "SN_Ia", "Ia-02cx": "SN_Ia", "Ia-03fg": "SN_Ia",
        "Ia-pec": "SN_Ia", "Ia-99aa": "SN_Ia",
        # --- SN_II (all H-rich merged) ---
        "Type II": "SN_II", "II-norm": "SN_II", "IIP": "SN_II",
        "IIL": "SN_II", "II-pec": "SN_II", "IIn": "SN_II",
        "IIb": "SN_II", "Ia-CSM": "SN_II",
        # --- SN_Ibc (all stripped-envelope merged) ---
        "Ib": "SN_Ibc", "Ic": "SN_Ibc", "Ib/c": "SN_Ibc",
        "Ic-BL": "SN_Ibc", "Ib-pec": "SN_Ibc", "Icn": "SN_Ibc",
        "Ien": "SN_Ibc", "Ibn": "SN_Ibc",
        # --- SLSN ---
        "Ic-SLSN": "SLSN", "II-SLSN": "SLSN", "SLSN I": "SLSN",
        "Ic.5-SLSN": "SLSN",
        # --- AGN ---
        "AGN": "AGN", "Seyfert": "AGN", "Galactic Nuclei": "AGN",
        # --- NonSN_Other ---
        "Tidal Disruption Event": "NonSN_Other",
        "Cataclysmic": "NonSN_Other", "U Gem": "NonSN_Other",
        "AM CVn": "NonSN_Other", "Nova-like": "NonSN_Other",
    },
}


# =============================================================================
# ANOMALY / RARE / UNKNOWN LABEL SETS
# =============================================================================

RARE_LABELS = {
    "ILRT": "Rare",
    "FBOT": "Rare",
    "Ca-rich": "Rare",
    "Luminous Red Nova": "Rare",
    "afterglow": "Rare",
    "long GRB": "Rare",
}

UNKNOWN_LABELS = {
    "QSO": "QSO(AGN)",
    "Blazar": "Blazar(AGN)",
    "BL Lac": "BL Lac(Blazar-AGN)",
    "Novae": "Novae-Cataclysmic",
    "Classical Nova": "Novae-Cataclysmic",
}

VAGUE_LABELS = {"Type I", "Supernova", "Eruptive"}

VARIABLE_KEYWORDS = {
    "varstar", "stellar variable", "variable", "periodic", "eclipsing",
    "pulsating", "mira", "delta scuti", "yso", "fu ori", "s doradus",
    "eruptive", "be", "ew", "w ursae maj", "microlensing",
}

BOGUS_KEYWORDS = ["bogus", "duplicate", "roid"]

CHALLENGING_KEYWORDS = {
    "non-variable", "orphan", "hosted", "nuclear",
    "transient", "anomolous",
}


# =============================================================================
# STEP 0: EXTRACT CHALLENGING SPECTRA
# =============================================================================

def extract_challenging(
    df: pd.DataFrame,
    class_col: str = "unique_classes",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract challenging spectra: unclassified and multi-label objects.

    These are removed by clean_metadata() but may be useful for evaluation.
    Multi-label objects keep all their labels in the class column.
    """
    unclassified = df[df["latest_classification"].isna()].copy()
    classified = df[df["latest_classification"].notna()]
    multi_label = classified[classified[class_col].str.contains(r"\|", na=False)].copy()

    challenging = pd.concat([unclassified, multi_label], ignore_index=True)

    if verbose:
        print(f"[challenging] Unclassified: {len(unclassified)}")
        print(f"[challenging] Multi-label:  {len(multi_label)}")
        print(f"[challenging] Total:        {len(challenging)}")
        if len(multi_label) > 0:
            print(f"\n[challenging] Multi-label examples:")
            for label, count in multi_label[class_col].value_counts().head(10).items():
                print(f"    {label:40s}: {count}")

    return challenging


# =============================================================================
# STEP 1: CLEAN METADATA
# =============================================================================

def clean_metadata(
    df: pd.DataFrame,
    class_col: str = "unique_classes",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Clean raw Fritz metadata: remove unclassified, variables, bogus,
    multi-label, vague, and challenging objects.

    Returns cleaned DataFrame ready for taxonomy mapping.
    """
    n0 = len(df)

    df = df[df["latest_classification"].notna()].copy()
    if verbose:
        print(f"[clean] Unclassified removed: {n0 - len(df)}")

    vague_mask = df[class_col].isin(VAGUE_LABELS)
    if verbose:
        print(f"[clean] Vague labels removed: {vague_mask.sum()}")
    df = df[~vague_mask].copy()

    df["_split_cls"] = df[class_col].str.split("|")

    def _is_variable(class_list):
        if isinstance(class_list, list):
            return any(c.strip().lower() in VARIABLE_KEYWORDS for c in class_list)
        return False

    var_mask = df["_split_cls"].apply(_is_variable)
    if verbose:
        print(f"[clean] Variables removed: {var_mask.sum()}")
    df = df[~var_mask].copy()

    multi_mask = df[class_col].str.contains(r"\|", na=False)
    if verbose:
        print(f"[clean] Multi-label removed: {multi_mask.sum()}")
    df = df[~multi_mask].copy()

    bogus_pattern = "|".join(BOGUS_KEYWORDS)
    bogus_mask = df[class_col].str.lower().str.contains(bogus_pattern, na=False)
    if verbose:
        print(f"[clean] Bogus removed: {bogus_mask.sum()}")
    df = df[~bogus_mask].copy()

    challenge_mask = df[class_col].str.lower().isin(CHALLENGING_KEYWORDS)
    if verbose:
        print(f"[clean] Challenging/uninformative removed: {challenge_mask.sum()}")
    df = df[~challenge_mask].copy()

    df = df.drop(columns=["_split_cls"], errors="ignore")

    if verbose:
        print(f"\n[clean] {n0} -> {len(df)} samples retained")
        print(f"[clean] {df[class_col].nunique()} unique labels remaining")

    return df.reset_index(drop=True)


# =============================================================================
# STEP 2: APPLY TAXONOMY
# =============================================================================

def apply_taxonomy(
    df: pd.DataFrame,
    taxonomy: dict,
    raw_col: str = "unique_classes",
    target_col: str = "taxonomy_class",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Apply a taxonomy mapping to a label DataFrame."""
    mapping     = taxonomy["mapping"]
    class_names = taxonomy["class_names"]
    hierarchy   = taxonomy["hierarchy"]
    tax_name    = taxonomy["name"]

    df = df.copy()
    df[target_col] = df[raw_col].map(mapping)

    unmapped = df[df[target_col].isna()]
    n_before = len(df)

    if verbose and len(unmapped) > 0:
        print(f"[{tax_name}] Dropped {len(unmapped)} unmapped samples:")
        print(unmapped[raw_col].value_counts().to_string())
        print()

    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    if verbose:
        print(f"[{tax_name}] {n_before} -> {len(df)} samples")
        print(f"[{tax_name}] {len(class_names)} classes\n")
        counts = df[target_col].value_counts()
        for cls in class_names:
            n = counts.get(cls, 0)
            pct = 100 * n / len(df)
            bar = "#" * int(pct / 2)
            print(f"  {cls:15s}: {n:6d}  ({pct:5.1f}%) {bar}")
        print()

    return df, class_names, hierarchy


# =============================================================================
# STEP 3: SPLIT INTO TRAIN / RARE / UNKNOWN / BOGUS
# =============================================================================

def split_anomaly_sets(
    df: pd.DataFrame,
    taxonomy: dict,
    raw_col: str = "unique_classes",
    verbose: bool = True,
) -> dict:
    """
    Split cleaned metadata into training data + anomaly evaluation sets.

    Returns dict with keys: train_df, rare_df, outliers_df, bogus_df,
    class_names, hierarchy.
    """
    mapping     = taxonomy["mapping"]
    class_names = taxonomy["class_names"]
    hierarchy   = taxonomy["hierarchy"]

    df = df.copy()

    def _classify(label):
        if label in mapping:
            return "train"
        elif label in RARE_LABELS:
            return "rare"
        elif label in UNKNOWN_LABELS:
            return "outliers"
        else:
            return "bogus"

    df["_split"] = df[raw_col].map(_classify)

    train_df = df[df["_split"] == "train"].copy()
    train_df["taxonomy_class"] = train_df[raw_col].map(mapping)
    train_df = train_df.drop(columns=["_split"])

    rare_df    = df[df["_split"] == "rare"].drop(columns=["_split"]).copy()
    outliers_df = df[df["_split"] == "outliers"].drop(columns=["_split"]).copy()
    bogus_df   = df[df["_split"] == "bogus"].drop(columns=["_split"]).copy()

    if verbose:
        print(f"Split summary:")
        print(f"  Train   : {len(train_df):6d} samples ({len(class_names)} classes)")
        print(f"  Rare    : {len(rare_df):6d} samples (anomaly eval)")
        print(f"  Outliers: {len(outliers_df):6d} samples (OOD eval)")
        print(f"  Bogus   : {len(bogus_df):6d} samples (discarded)")

        if len(rare_df) > 0:
            print(f"\n  Rare breakdown:")
            for label, count in rare_df[raw_col].value_counts().items():
                print(f"    {label:25s}: {count}")

        if len(outliers_df) > 0:
            print(f"\n  Outliers breakdown:")
            for label, count in outliers_df[raw_col].value_counts().items():
                print(f"    {label:25s}: {count}")
        print()

    return {
        "train_df":    train_df,
        "rare_df":     rare_df,
        "outliers_df": outliers_df,
        "bogus_df":    bogus_df,
        "class_names": class_names,
        "hierarchy":   hierarchy,
    }


# =============================================================================
# STEP 4: BUILD .PT FILES
# =============================================================================

def build_all_pt_files(
    splits: dict,
    spectra_dir: str,
    output_dir: str,
    remove_cont: bool = False,
    scale_method: str = "robust",
    verbose: bool = True,
) -> dict:
    """
    Process spectra and save .pt files for each split.

    Creates:
        output_dir/main.pt      — all training data
        output_dir/rare.pt      — rare transients for anomaly eval
        output_dir/outliers.pt  — outlier types for OOD eval
    """
    from iris.processing import batch_process

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    sets_to_build = [
        ("main",    splits["train_df"]),
        ("rare",    splits["rare_df"]),
        ("outliers", splits["outliers_df"]),
    ]

    for name, df in sets_to_build:
        if len(df) == 0:
            if verbose:
                print(f"[build] {name}: empty, skipping")
            results[name] = None
            continue

        if verbose:
            print(f"\n{'='*50}")
            print(f"[build] Processing {name} ({len(df)} samples)")
            print(f"[build] scale_method={scale_method}, remove_cont={remove_cont}")
            print(f"{'='*50}")

        out_path = batch_process(
            spectra_dir_good=spectra_dir,
            output_dir=output_dir,
            label_df=df,
            remove_cont=remove_cont,
            scale_method=scale_method,
        )

        if out_path and os.path.exists(out_path):
            final_path = os.path.join(output_dir, f"{name}.pt")
            os.rename(out_path, final_path)
            results[name] = final_path
            if verbose:
                data = torch.load(final_path, map_location="cpu", weights_only=False)
                print(f"[build] Saved: {final_path}")
                print(f"[build]   flux shape: {data['flux'].shape}")
                print(f"[build]   samples:    {data['num_samples']}")
        else:
            results[name] = None
            if verbose:
                print(f"[build] {name}: processing failed")

    return results


def build_single_pt(
    df: pd.DataFrame,
    spectra_dir: str,
    output_path: str,
    remove_cont: bool = False,
    scale_method: str = "robust",
    verbose: bool = True,
) -> Optional[str]:
    """Process spectra from a single DataFrame and save as .pt file."""
    from iris.processing import batch_process

    if len(df) == 0:
        if verbose:
            print(f"[build] Empty DataFrame, skipping.")
        return None

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"[build] Processing {len(df)} samples -> {output_path}")
        print(f"[build] scale_method={scale_method}, remove_cont={remove_cont}")

    out = batch_process(
        spectra_dir_good=spectra_dir,
        output_dir=output_dir,
        label_df=df,
        remove_cont=remove_cont,
        scale_method=scale_method,
    )

    if out and os.path.exists(out):
        if out != output_path:
            os.rename(out, output_path)
        if verbose:
            data = torch.load(output_path, map_location="cpu", weights_only=False)
            print(f"[build] Saved: {output_path} ({data['num_samples']} samples)")
        return output_path

    return None


# =============================================================================
# GATHER SPECTRUM FILES
# =============================================================================

def gather_spectra(
    df: pd.DataFrame,
    base_path: str,
    output_folder: str,
    dat_col: str = "dat_file",
    folder_prefix: str = "fritz_data_",
    verbose: bool = True,
) -> Tuple[int, List[str]]:
    """Copy .dat spectrum files referenced in df into a single flat directory."""
    os.makedirs(output_folder, exist_ok=True)

    clean_files = set(df[dat_col].dropna().astype(str))
    if verbose:
        print(f"[gather] Spectra to find: {len(clean_files)}")

    search_folders = sorted([
        d for d in os.listdir(base_path)
        if d.startswith(folder_prefix) and os.path.isdir(os.path.join(base_path, d))
    ])
    if verbose:
        print(f"[gather] Source folders: {search_folders}")

    copied = 0
    not_found = set(clean_files)

    for folder_name in search_folders:
        folder = os.path.join(base_path, folder_name)
        for root, _, files in os.walk(folder):
            for f in files:
                if not f.endswith(".dat") or f not in clean_files:
                    continue
                src = os.path.join(root, f)
                dst = os.path.join(output_folder, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied += 1
                not_found.discard(f)

    not_found = sorted(not_found)

    if verbose:
        print(f"[gather] Copied:    {copied}")
        print(f"[gather] Not found: {len(not_found)}")
        if not_found and len(not_found) <= 10:
            for f in not_found:
                print(f"  - {f}")
        elif not_found:
            print(f"  (showing first 10)")
            for f in not_found[:10]:
                print(f"  - {f}")

    return copied, not_found


# =============================================================================
# SAVE / LOAD SPLITS
# =============================================================================

def save_splits(
    splits: dict,
    output_dir: str = "fritz_metadata",
    prefix: str = "split",
    verbose: bool = True,
):
    """Save all split DataFrames to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    for name in ("train_df", "rare_df", "outliers_df", "bogus_df"):
        df = splits[name]
        short = name.replace("_df", "")
        path = os.path.join(output_dir, f"{prefix}_{short}.csv")
        df.to_csv(path, index=False)
        if verbose:
            print(f"[save] {path} ({len(df)} rows)")

    meta = {
        "class_names": splits["class_names"],
        "hierarchy": splits["hierarchy"],
    }
    meta_path = os.path.join(output_dir, f"{prefix}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"[save] {meta_path}")


def load_splits(
    input_dir: str = "fritz_metadata",
    prefix: str = "split",
    verbose: bool = True,
) -> dict:
    """Load splits from CSV files saved by save_splits()."""
    splits = {}

    for name in ("train", "rare", "outliers", "bogus"):
        path = os.path.join(input_dir, f"{prefix}_{name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            splits[f"{name}_df"] = df
            if verbose:
                print(f"[load] {path} ({len(df)} rows)")
        else:
            splits[f"{name}_df"] = pd.DataFrame()
            if verbose:
                print(f"[load] {path} (not found, empty)")

    meta_path = os.path.join(input_dir, f"{prefix}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        splits["class_names"] = meta["class_names"]
        splits["hierarchy"]   = meta["hierarchy"]
        if verbose:
            print(f"[load] {meta_path}")
    else:
        splits["class_names"] = []
        splits["hierarchy"]   = {}

    return splits


# =============================================================================
# VALIDATION & CONFIG UTILITIES
# =============================================================================

def validate_taxonomy(taxonomy: dict) -> bool:
    """Check internal consistency. Raises AssertionError if invalid."""
    class_names = taxonomy["class_names"]
    hierarchy   = taxonomy["hierarchy"]
    mapping     = taxonomy["mapping"]

    mapped_targets = set(mapping.values())
    assert mapped_targets.issubset(set(class_names)), (
        f"Mapped targets not in class_names: {mapped_targets - set(class_names)}"
    )

    all_fine = []
    for fine_list in hierarchy.values():
        all_fine.extend(fine_list)
    assert sorted(all_fine) == sorted(class_names), (
        f"Hierarchy mismatch:\n"
        f"  In hierarchy but not class_names: {set(all_fine) - set(class_names)}\n"
        f"  In class_names but not hierarchy: {set(class_names) - set(all_fine)}"
    )

    for cls in class_names:
        assert cls in mapped_targets, f"Class '{cls}' has no mapping entries"

    print("All checks passed.")
    return True


def print_config(taxonomy: dict):
    """Print config lines to paste into ConfigV5."""
    class_names = taxonomy["class_names"]
    hierarchy   = taxonomy["hierarchy"]
    print("# --- Paste into config_new.py ---")
    print(f"num_classes: int = {len(class_names)}")
    print(f"class_names: List[str] = field(default_factory=lambda: {class_names})")
    print(f"hierarchy: Dict = field(default_factory=lambda: {{")
    for k, v in hierarchy.items():
        print(f'    "{k}": {v},')
    print("})")
