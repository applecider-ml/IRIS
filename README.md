# IRIS — Identification and Reduction of Interesting Spectra

> *Named after Iris (Ἶρις), Greek goddess of the rainbow and messenger between the heavens and mortals. A spectrum is light unfolded into its rainbow; IRIS carries the messages of cosmic transients across that bridge — from raw photons to ML-ready tensors.*

Spectral preprocessing pipeline for transient classification. Downloads, cleans, and processes astronomical spectra from [Fritz/SkyPortal](https://fritz.science) into ML-ready `.pt` tensors.

## Pipeline Overview

```
Fritz API → .dat files → QC filtering → taxonomy mapping → .pt tensors
```

**Steps:**
1. **Fetch** — Download spectra from Fritz (or use pre-fetched `fritz_data_*` folders)
2. **Clean** — Remove unclassified, variable stars, bogus, multi-label objects
3. **Gather** — Collect `.dat` files into a single flat directory
4. **QC** — Compute spectral quality metrics (SNR, coverage, flatness) and filter
5. **Split** — Partition into train / rare (anomaly) / outliers (OOD) sets
6. **Build** — Interpolate onto uniform grid, scale, save as PyTorch `.pt` files

## Taxonomy

Two classification schemes:

| Scheme | Classes | Use case |
|--------|---------|----------|
| **FINE_10** | SN_Ia, SN_II, SN_IIn, SN_IIb, SN_Ibc, SN_Ibn, SLSN, AGN, TDE, CV | Full classification |
| **COARSE_6** | SN_Ia, SN_II, SN_Ibc, SLSN, AGN, NonSN_Other | Broad grouping |

Hierarchy: `SN_Thermonuclear > SN_CC_H > SN_CC_SE > SN_Luminous > NonSN`

## Installation

```bash
pip install .
```

Or for development:
```bash
pip install -e .
```

## Usage

### On Delta (SLURM)

```bash
# 1. Quick test — fetch 10 spectra
cd /projects/bcrv/asasli/IRIS
sbatch --export=FRITZ_TOKEN=your-token-here \
  --job-name=iris-test \
  --account=bcrv-delta-cpu \
  --partition=cpu \
  --ntasks=1 --cpus-per-task=4 --mem=8G --time=00:30:00 \
  --output=$HOME/iris_test_%j.out \
  --error=$HOME/iris_test_%j.err \
  --wrap='
mkdir -p ~/iris_test
cd /projects/bcrv/asasli/IRIS
python -c "
from iris.fritz.client import FritzClient
from iris.fritz.pipeline import run_pipeline, discover_source_ids
import os
client = FritzClient(token=os.environ[\"FRITZ_TOKEN\"])
ids = discover_source_ids(client, page_size=10)[:10]
print(f\"Fetching {len(ids)} sources...\")
run_pipeline(client, ids, output_dir=os.path.expanduser(\"~/iris_test\"), max_workers=2)
"
'

# 2. Monitor
squeue -u $USER
tail -f ~/iris_test_*.out
```

See [`examples/`](examples/) for more SLURM recipes (full pipeline, GPU jobs, interactive sessions).

### Full pipeline

```bash
python run_pipeline.py --data-dir /path/to/data --scale-method both
```

Options:
```
--data-dir       Root directory with fritz_data_* folders (required)
--scale-method   robust | zscore | both (default: both)
--skip-fetch     Use existing metadata.csv
--skip-gather    Use existing fritz_data_clean/
--skip-qc        Skip quality control filtering
```

### As a library

```python
from iris.taxonomy import FINE_10, clean_metadata, split_anomaly_sets, build_single_pt
from iris.quality_control import compute_qc, apply_qc_filter

# Clean and split
df = pd.read_csv("fritz_metadata/metadata.csv")
clean_df = clean_metadata(df)
splits = split_anomaly_sets(clean_df, FINE_10)

# Build .pt files
build_single_pt(splits["train_df"], "fritz_data_clean", "spectra_pt/main.pt")
```

### Fetching data from Fritz

```python
from iris.fritz.client import FritzClient
from iris.fritz.pipeline import run_full_pipeline

client = FritzClient(token="your-fritz-token")
run_full_pipeline(client, output_dir="fritz_data", max_workers=4)
```

## Output Format

Each `.pt` file contains:

| Key | Type | Description |
|-----|------|-------------|
| `flux` | Tensor (N, 4096) | Interpolated, scaled spectra |
| `obj_id` | list[str] | ZTF object identifiers |
| `redshift` | list[float] | Catalog redshifts |
| `phase` | list[float] | Days from peak brightness |
| `instrument` | list[str] | Instrument name (SEDM, LRIS, ...) |
| `class` | list[str] | Raw classification label |
| `taxonomy_I` | list[str] | Fine class (FINE_10) |
| `taxonomy_II` | list[str] | Hierarchy group |

Wavelength grid: 3850–9000 A, 4096 points.

## GitHub Actions

The workflow runs manually via `workflow_dispatch`:

```
Actions → IRIS → Run workflow → select options
```

Produces artifacts: `.pt` files + split CSVs.

## Examples

The [`examples/`](examples/) directory contains notebooks and scripts demonstrating how to query and use IRIS with Delta, including data retrieval and inference workflows.

## Project Structure

```
iris/
├── taxonomy.py          # FINE_10/COARSE_6 definitions, clean/split/build
├── processing.py        # Spectral interpolation, scaling, batch processing
├── quality_control.py   # QC metrics and filtering
├── filter_spectra.py    # Safe classification, deduplication
└── fritz/
    ├── client.py        # Fritz API HTTP client
    ├── data_processing.py  # API response parsing
    ├── pipeline.py      # Batch download orchestration
    └── storage.py       # .dat and CSV persistence
```
