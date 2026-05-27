#!/usr/bin/env python3
"""
IRIS Pipeline — Main entry point.

Usage:
    python run_pipeline.py --data-dir /path/to/data
    python run_pipeline.py --data-dir /path/to/data --scale-method zscore
    python run_pipeline.py --data-dir /path/to/data --skip-qc
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from iris.taxonomy import (
    FINE_10, clean_metadata, extract_challenging, gather_spectra,
    split_anomaly_sets, build_single_pt,
)
from iris.quality_control import compute_qc, apply_qc_filter


def step_concat_metadata(data_dir: str, output_path: str) -> pd.DataFrame:
    """Step 1: Concatenate metadata from all fritz_data_*/metadata.csv."""
    print("\n" + "=" * 60)
    print("STEP 1: Concatenate metadata")
    print("=" * 60)

    pattern = os.path.join(data_dir, "fritz_data_*/metadata.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No metadata CSVs found matching {pattern}. "
            "Run the Fritz data fetcher first."
        )

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Total: {len(df)} spectra from {len(csv_files)} folders")
    print(f"Saved: {output_path}")
    return df


def step_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Clean metadata."""
    print("\n" + "=" * 60)
    print("STEP 2: Clean metadata")
    print("=" * 60)
    return clean_metadata(df)


def step_gather(clean_df: pd.DataFrame, data_dir: str, output_folder: str):
    """Step 3: Gather .dat files into a single clean directory."""
    print("\n" + "=" * 60)
    print("STEP 3: Gather spectra")
    print("=" * 60)
    return gather_spectra(clean_df, base_path=data_dir, output_folder=output_folder)


def step_qc(clean_df: pd.DataFrame, spectra_dir: str):
    """Step 4-5: Compute QC metrics and apply filter."""
    print("\n" + "=" * 60)
    print("STEP 4: Quality control")
    print("=" * 60)
    df_qc = compute_qc(clean_df, spectra_dir=spectra_dir)
    return apply_qc_filter(df_qc)


def step_split(df_pass: pd.DataFrame):
    """Step 6-7: Split into train/rare/outliers and add taxonomy columns."""
    print("\n" + "=" * 60)
    print("STEP 6: Split & taxonomy")
    print("=" * 60)

    splits = split_anomaly_sets(df_pass, FINE_10)

    reverse_hier = {
        cls: group
        for group, classes in FINE_10["hierarchy"].items()
        for cls in classes
    }
    splits["train_df"]["taxonomy_II"] = (
        splits["train_df"]["taxonomy_class"].map(reverse_hier)
    )
    splits["train_df"]["taxonomy_I"] = splits["train_df"]["taxonomy_class"]

    return splits


def step_save_csvs(splits: dict, metadata_dir: str):
    """Step 8: Save split CSVs."""
    print("\n" + "=" * 60)
    print("STEP 8: Save CSVs")
    print("=" * 60)

    os.makedirs(metadata_dir, exist_ok=True)
    for name, key in [("train", "train_df"), ("rare", "rare_df"),
                      ("outliers", "outliers_df"), ("bogus", "bogus_df")]:
        path = os.path.join(metadata_dir, f"{name}.csv")
        splits[key].to_csv(path, index=False)
        print(f"  {path}: {len(splits[key])} rows")


def step_build_pt(splits: dict, spectra_dir: str, output_dir: str,
                  scale_method: str = "robust"):
    """Step 9: Build .pt files for all splits."""
    print("\n" + "=" * 60)
    print(f"STEP 9: Build .pt files (scale={scale_method})")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    for name, key in [("main", "train_df"), ("rare", "rare_df"),
                      ("outliers", "outliers_df")]:
        df = splits[key]
        if len(df) == 0:
            print(f"  {name}: empty, skipping")
            continue
        suffix = f"_{scale_method[0]}" if scale_method != "robust" else ""
        out_path = os.path.join(output_dir, f"{name}{suffix}.pt")
        build_single_pt(df, spectra_dir, out_path, scale_method=scale_method)

    # Combined rare + outliers
    rare_df = splits["rare_df"]
    outliers_df = splits["outliers_df"]
    if len(rare_df) > 0 or len(outliers_df) > 0:
        combined = pd.concat([rare_df, outliers_df], ignore_index=True)
        suffix = f"_{scale_method[0]}" if scale_method != "robust" else ""
        out_path = os.path.join(output_dir, f"rare_outliers{suffix}.pt")
        build_single_pt(combined, spectra_dir, out_path, scale_method=scale_method)


def main():
    parser = argparse.ArgumentParser(
        description="IRIS preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root directory containing fritz_data_* folders and .dat files",
    )
    parser.add_argument(
        "--metadata-dir", default=None,
        help="Output directory for CSVs (default: <data-dir>/fritz_metadata)",
    )
    parser.add_argument(
        "--clean-dir", default=None,
        help="Directory for gathered clean .dat files (default: <data-dir>/fritz_data_clean)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for output .pt files (default: <data-dir>/spectra_pt)",
    )
    parser.add_argument(
        "--scale-method", choices=["robust", "zscore", "both"], default="both",
        help="Scaling method for .pt files",
    )
    parser.add_argument("--skip-fetch",  action="store_true",
                        help="Skip metadata concatenation")
    parser.add_argument("--skip-gather", action="store_true",
                        help="Skip spectra gathering")
    parser.add_argument("--skip-qc",     action="store_true",
                        help="Skip quality control")
    args = parser.parse_args()

    data_dir     = os.path.abspath(args.data_dir)
    metadata_dir = args.metadata_dir or os.path.join(data_dir, "fritz_metadata")
    clean_dir    = args.clean_dir or os.path.join(data_dir, "fritz_data_clean")
    output_dir   = args.output_dir or os.path.join(data_dir, "spectra_pt")
    metadata_csv = os.path.join(metadata_dir, "metadata.csv")

    # Step 1
    if args.skip_fetch:
        print(f"\n[skip] Using existing metadata: {metadata_csv}")
        df = pd.read_csv(metadata_csv, low_memory=False)
    else:
        df = step_concat_metadata(data_dir, metadata_csv)

    # Step 1b: Extract challenging (unclassified + multi-label)
    print("\n" + "=" * 60)
    print("STEP 1b: Extract challenging spectra")
    print("=" * 60)
    challenging_df = extract_challenging(df)

    # Step 2
    clean_df = step_clean(df)

    # Step 3
    if args.skip_gather:
        print(f"\n[skip] Using existing clean spectra: {clean_dir}")
    else:
        # Gather both clean and challenging spectra
        all_for_gather = pd.concat([clean_df, challenging_df], ignore_index=True).drop_duplicates(subset=["dat_file"])
        step_gather(all_for_gather, data_dir, clean_dir)

    # Step 4-5
    if args.skip_qc:
        print("\n[skip] Skipping QC — using all cleaned spectra")
        df_pass = clean_df
    else:
        df_pass, _ = step_qc(clean_df, clean_dir)

    # Step 6-7
    splits = step_split(df_pass)

    # Step 8
    step_save_csvs(splits, metadata_dir)

    # Save challenging CSV
    if len(challenging_df) > 0:
        chal_path = os.path.join(metadata_dir, "challenging.csv")
        challenging_df.to_csv(chal_path, index=False)
        print(f"  {chal_path}: {len(challenging_df)} rows")

    # Step 9
    if args.scale_method == "both":
        step_build_pt(splits, clean_dir, output_dir, scale_method="robust")
        step_build_pt(splits, clean_dir, output_dir + "_zscore", scale_method="zscore")
    else:
        step_build_pt(splits, clean_dir, output_dir, scale_method=args.scale_method)

    # Build challenging .pt
    if len(challenging_df) > 0:
        scale = "robust" if args.scale_method != "zscore" else "zscore"
        chal_pt = os.path.join(output_dir, "challenging.pt")
        print(f"\n[build] Building challenging.pt ({len(challenging_df)} samples)")
        build_single_pt(challenging_df, clean_dir, chal_pt, scale_method=scale)

    print("\n" + "=" * 60)
    print("IRIS pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
