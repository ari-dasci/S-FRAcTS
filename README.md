# S-FRAcTS

FRAcTS: Federated Robust Attention for forecasting Time Series

Official implementation accompanying our paper (see [Citation](#citation)).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run experiments through `main.py`. Train a model, then test/validate it:

```bash
python main.py --mode train --model transformerFedFRAcTS --agg_mode FRAcTS --data_name <dataset_name> \
    --data_root ./data --output_root ./results

python main.py --mode test --data_name <dataset_name> \
    --data_root ./data --output_root ./results
```

`--agg_alpha` (the balance between server-similarity and centrality in the FRAcTS aggregator) defaults to `0.75`, the value used for the paper's reported results; pass `--agg_alpha <value>` to try any other weighting.

See `python main.py --help` for the full list of options (training mode, model/aggregation strategy, attention mechanism, federated learning hyperparameters, etc.).

Each training run is saved under `<output_root>/<dataset_name>/alpha_<alpha>/checkpoints/<run_name>/`, together with a `meta.json` describing the run (model, attention type, number of clients, sequence/prediction length, alpha). `--mode test` and `--mode validate` scan that folder and evaluate every checkpoint found there, appending results to `test_results.json` / `val_results.json` in the same `alpha_<alpha>` directory.

## License

This project is licensed under the GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite:


