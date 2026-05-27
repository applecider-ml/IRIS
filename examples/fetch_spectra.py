#!/usr/bin/env python3
"""
Fetch spectra from Fritz API.

The token is read from the FRITZ_TOKEN environment variable.

Setup:
    # Option 1: Export in your shell
    export FRITZ_TOKEN="your-token-here"

    # Option 2: Create a .env file (git-ignored)
    echo "FRITZ_TOKEN=your-token-here" > .env

    # Option 3: Pass via sbatch
    sbatch --export=ALL,FRITZ_TOKEN=your-token examples/submit_fetch.sh

Usage:
    python examples/fetch_spectra.py
    python examples/fetch_spectra.py --output-dir fritz_data_new --max-workers 2
"""

import argparse
import os
import sys


def load_token() -> str:
    """Load Fritz token from environment or .env file."""
    token = os.environ.get("FRITZ_TOKEN")

    if not token:
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("FRITZ_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip("\"'")
                        break

    if not token:
        print("ERROR: FRITZ_TOKEN not found.")
        print("Set it via:")
        print("  export FRITZ_TOKEN='your-token'")
        print("  or create a .env file with FRITZ_TOKEN=your-token")
        sys.exit(1)

    return token


def main():
    parser = argparse.ArgumentParser(description="Fetch spectra from Fritz")
    parser.add_argument("--output-dir", default="fritz_data", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--ids-file", default=None,
                        help="File with obj_ids (one per line). If not set, discovers all.")
    parser.add_argument("--request-delay", type=float, default=0.5,
                        help="Seconds between API calls per object")
    parser.add_argument("--slice-index", type=int, default=None,
                        help="0-based slice index (for SLURM array jobs)")
    parser.add_argument("--n-slices", type=int, default=None,
                        help="Total number of slices")
    parser.add_argument("--discover-only", default=None, metavar="FILE",
                        help="Only discover source IDs and save to FILE (no download)")
    args = parser.parse_args()

    token = load_token()

    from iris.fritz.client import FritzClient
    from iris.fritz.pipeline import run_pipeline

    client = FritzClient(token=token)

    # Discover-only mode: save IDs to file and exit
    if args.discover_only:
        from iris.fritz.pipeline import discover_source_ids
        print("Discovering all sources with spectra...")
        obj_ids = discover_source_ids(client)
        with open(args.discover_only, "w") as f:
            for oid in obj_ids:
                f.write(oid + "\n")
        print(f"Saved {len(obj_ids)} IDs to {args.discover_only}")
        return

    if args.ids_file:
        with open(args.ids_file) as f:
            obj_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(obj_ids)} IDs from {args.ids_file}")
    else:
        print("Discovering all sources with spectra...")
        from iris.fritz.pipeline import discover_source_ids
        obj_ids = discover_source_ids(client)
        print(f"Discovered {len(obj_ids)} sources")

    # Slice for SLURM array jobs
    if args.slice_index is not None and args.n_slices is not None:
        obj_ids = obj_ids[args.slice_index::args.n_slices]
        print(f"Slice {args.slice_index}/{args.n_slices}: {len(obj_ids)} IDs")

    print(f"Fetching {len(obj_ids)} objects...")
    run_pipeline(client, obj_ids, output_dir=args.output_dir,
                 max_workers=args.max_workers, request_delay=args.request_delay)


if __name__ == "__main__":
    main()
