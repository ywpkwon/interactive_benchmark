# Benchmarking project

## Requirements

### pip
There could be more.

    pip install easydict tqdm

### Dash
You will need to install [Dash](https://plot.ly/dash/). This is basically for web-launching plots. I am also thinking to visualizing results without Dash for development stages, but it is not done yet.

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

0. `cd detection/`
1. Open `setting.json`
2. Change `benchmark_root` for your case. E.g, `/BENCHMARK_ROOT`
3. `/BENCHMARK_ROOT/images` contains images to test.
4. Run your detection for those images, and put your result file, e.g., `trial203.out` to `/BENCHMARK_ROOT/detection/trial203.out`.
5. Run `python evaluate_performance.py`

Once you run `evaluate_performance.py`, it evaluates all `.out` files in the directory, and generate `.cache` files of the same name. Since it skips a `.out` file if its `.cache` file already exists.
When you plot, the figure will show precision-recall curves from the all `.cache` files.



