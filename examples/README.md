# Examples

## Quick test: fetch a few spectra

```bash
# 1. Submit a test fetch job to SLURM
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

# 3. Check results
ls -lh ~/iris_test/
cat ~/iris_test/metadata.csv
```

## Discover source IDs first

Save all available IDs to a file, review/filter them, then fetch only the ones you want.

```bash
# Step 1: Discover and save all IDs
export FRITZ_TOKEN="your-token-here"
python examples/fetch_spectra.py --discover-only all_ids.txt

# Step 2: Review / filter
wc -l all_ids.txt          # how many?
head -20 all_ids.txt        # preview

# Step 3: Fetch only the ones you want
python examples/fetch_spectra.py --ids-file all_ids.txt --output-dir fritz_data

# Or submit as array job
sbatch --export=ALL,ID_FILE=all_ids.txt examples/fetch_array.sh
```

## Parallel fetch with SLURM array jobs

Split a large fetch across multiple parallel jobs, each downloading its own slice.

### Basic (10 parallel jobs, auto-discover sources)

```bash
export FRITZ_TOKEN="your-token-here"
sbatch --export=ALL examples/fetch_array.sh
```

### With a pre-built ID file

```bash
export FRITZ_TOKEN="your-token-here"
sbatch --export=ALL,ID_FILE=all_ids.txt examples/fetch_array.sh
```

### Custom number of slices

```bash
# 5 parallel jobs instead of 10
sbatch --export=ALL --array=0-4 examples/fetch_array.sh
```

### Custom output location

```bash
sbatch --export=ALL,OUTPUT_BASE=/scratch/$USER/fritz_data examples/fetch_array.sh
```

Each task writes to `fritz_data_0/`, `fritz_data_1/`, ..., `fritz_data_9/`.

## Process already-downloaded data

If spectra are already fetched in `fritz_data_*` folders, run the pipeline from gather onwards.
Output goes to `~/iris_clean` (.dat files) and `~/iris_metadata` (CSVs).

```bash
cd /projects/bcrv/asasli/IRIS

# Full pipeline (gather + QC + split + .pt)
python run_pipeline.py \
    --data-dir /projects/bcrv/asasli/later/Spectra \
    --clean-dir ~/iris_clean \
    --metadata-dir ~/iris_metadata \
    --output-dir ~/iris_metadata/spectra_pt \
    --scale-method both

# Skip QC if you want all cleaned spectra without filtering
python run_pipeline.py \
    --data-dir /projects/bcrv/asasli/later/Spectra \
    --clean-dir ~/iris_clean \
    --metadata-dir ~/iris_metadata \
    --output-dir ~/iris_metadata/spectra_pt \
    --scale-method both \
    --skip-qc

# Results
ls ~/iris_clean/           # gathered .dat files
ls ~/iris_metadata/        # train.csv, rare.csv, outliers.csv, bogus.csv, challenging.csv
ls ~/iris_metadata/spectra_pt/  # .pt tensors
```

### Via SLURM

```bash
sbatch --export=ALL,DATA_DIR=/projects/bcrv/asasli/later/Spectra,SKIP_FETCH=1 \
    --wrap='
cd /projects/bcrv/asasli/IRIS
python run_pipeline.py \
    --data-dir /projects/bcrv/asasli/later/Spectra \
    --clean-dir ~/iris_clean \
    --metadata-dir ~/iris_metadata \
    --output-dir ~/iris_metadata/spectra_pt \
    --scale-method both
' \
    --job-name=iris-process \
    --account=bcrv-delta-cpu \
    --partition=cpu \
    --cpus-per-task=16 --mem=64G --time=02:00:00 \
    --output=logs/iris_process_%j.out \
    --error=logs/iris_process_%j.err
```

## Full pipeline (processing)

### Basic submission

```bash
sbatch examples/submit_delta.sh
```

### Custom data directory

```bash
sbatch --export=DATA_DIR=/path/to/data examples/submit_delta.sh
```

### Skip steps for faster runs

```bash
# Skip QC (use all cleaned spectra without filtering)
sbatch --export=SKIP_QC=1 examples/submit_delta.sh

# Skip everything except .pt building (data already prepared)
sbatch --export=ALL,SKIP_FETCH=1,SKIP_GATHER=1,SKIP_QC=1 examples/submit_delta.sh
```

### GPU job (for future training steps)

```bash
sbatch --partition=gpuA100x4 --gpus=1 --account=bcrv-delta-gpu examples/submit_delta.sh
```

### Monitor your job

```bash
# Check queue status
squeue -u $USER

# Watch output in real-time
tail -f logs/iris_<JOB_ID>.out

# Job history
sacct -j <JOB_ID> --format=JobID,Elapsed,MaxRSS,State
```

## Interactive session (for debugging)

```bash
srun --account=bcrv-delta-cpu --partition=cpu-interactive \
     --nodes=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash

# Then run manually:
export FRITZ_TOKEN="your-token-here"
cd /projects/bcrv/asasli/IRIS
python examples/fetch_spectra.py --output-dir ~/iris_test
```
