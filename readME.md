# Benchmarking project

## Requirements

### pip
You may need to install more than these based on your environment.

    pip install easydict tqdm seaborn

### dash (optional)
**You can just skip this part** because this part is just for visualizing plots on web browsers (You can check plots the without Dash, and it is more simple.) If this is needed,
you will need to install [Dash](https://plot.ly/dash/).

    pip install dash==0.18.3  # The core dash backend
    pip install dash-renderer==0.11.0  # The dash front-end
    pip install dash-html-components==0.8.0  # HTML components
    pip install dash-core-components==0.13.0  # Supercharged components
    pip install plotly --upgrade  # Plotly graphing library used in examples

## Folders

ROOT
|- detection/                  # scripts for detection evaluation
|- segmentation/               # scripts for segmentation evaluation
|- data-preparation.ipynb      # code for test-data preparation. you can assume it is already done
|- readME.md                   # this file


## Detection Benchmark

There is `setting.json` in the `detection` folder.

1. Open `setting.json`, and change `benchmark_root` for your case. E.g, `/BENCHMARK_ROOT`
2. Put your result file, e.g., `trial203.out` to `/BENCHMARK_ROOT/detection/trial203.out`.
3. Run `python evaluate_performance.py`

   - Once you run `evaluate_performance.py`, it evaluates all `.out` files in the directory, and generate `.cache` files of the same name.
   - It skips `.out` files if their `.cache` files already exist.

4. Run `python core.py`

   - The figure will show precision-recall curves from the all `.cache` files.
