# Stock Trend Classifier

The notebook workflow is split into a small layered Python project:

- `src/stock_trend/config.py`: relative paths and training settings.
- `src/stock_trend/data.py`: CSV input and submission output.
- `src/stock_trend/preprocessing.py`: cleaning, feature engineering, scaling, and test sequences.
- `src/stock_trend/datasets.py`: ticker splits and TensorFlow sequence datasets.
- `src/stock_trend/model.py`: focal loss and the CNN/BiLSTM model.
- `src/stock_trend/pipeline.py`: training and prediction orchestration.
- `main.py`: command-line entry point.

## Data layout

Place the competition files here. The paths are intentionally relative to this project and contain no machine-specific locations:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
```

Install dependencies with `pip install -r requirements.txt`, then run:

```powershell
$env:PYTHONPATH = "src"
python main.py
```

The best model is saved in `artifacts/best_model.keras` and the submission is saved in `outputs/submission.csv`. To use another data directory:

```powershell
python main.py --data-dir path\to\data
```
