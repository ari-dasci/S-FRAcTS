
import torch
from model.transformer import TransformerModel
from model.utils.timeseries_dataset import TimeSeriesDataset
from model.utils.tools import EarlyStopping
from flex.data import FedDatasetConfig, FedDataDistribution, Dataset
from torch.utils.data import DataLoader
from model.attn import FullAttention, ProbAttention, LocalAttention
from flex.pool import FlexPool
from federated.primitives import build_server_model
from federated.primitives import train_n_rounds
import torch.nn as nn
from torch import optim
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
import datetime as dt
import argparse
from pathlib import Path

def load_data_csv(csv_path, time_column="date", train_percentage=0.7):
	"""
	Load a time series dataset from a CSV file and split it into train/test.

	Parameters
	----------
	csv_path : str
		Path to the CSV file.
	time_column : str
		Name of the column containing the timestamps.
	train_percentage : float
		Fraction of the data used for training.

	Returns
	-------
	data_train, timestamps_train, data_test, timestamps_test
	"""
	df = pd.read_csv(csv_path)
	df[time_column] = pd.to_datetime(df[time_column])
	df = df.sort_values(by=time_column).reset_index(drop=True)

	timestamps = df[time_column].values
	data_values = df.drop(columns=[time_column]).values

	# Chronological split (no shuffling)
	split_idx = int(len(data_values) * train_percentage)

	data_train = data_values[:split_idx]
	data_test = data_values[split_idx:]

	timestamps_train = timestamps[:split_idx]
	timestamps_test = timestamps[split_idx:]

	return data_train, timestamps_train, data_test, timestamps_test
   

def load_model(name,path_model,seq_len,pred_len,attn,feature):
	"""
	Build a model from `name`/`seq_len`/`pred_len`/`attn`/`feature` and load its
	weights from a checkpoint file.

	Parameters
	----------
	name : str
		Name of the model architecture to build (see `init_model`).
	path_model : str
		Path to the checkpoint.pth file to load.
	seq_len : int
		Input sequence length the model was trained with.
	pred_len : int
		Prediction length the model was trained with.
	attn : str
		Attention mechanism: "local", "prob" or "full".
	feature : int
		Number of input features/variables.

	Returns
	-------
	model : torch.nn.Module
		Model with the checkpoint's weights loaded.
	"""
	model = init_model(name,seq_len,pred_len,attn,feature)
	if not os.path.exists(path_model):
		raise FileNotFoundError(f"Model checkpoint not found: {path_model}")
	model.load_state_dict(torch.load(path_model))
	return model


def init_model(model,seq_len,pred_len,attn, features):
	"""
	Build a new (randomly initialized) TransformerModel.

	Parameters
	----------
	model : str
		Name of the model architecture (currently only TransformerModel is built;
		kept as a parameter for the naming convention used elsewhere in main.py).
	seq_len : int
		Input sequence length.
	pred_len : int
		Prediction length.
	attn : str
		Attention mechanism: "local", "prob" or "full".
	features : int
		Number of input features/variables.

	Returns
	-------
	model : torch.nn.Module
		Newly initialized model.
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return TransformerModel(feats=features, n_window=seq_len,
						loss = torch.nn.MSELoss(), 
      					d_model=None,	
           				n_head=8,
           				dim_feedforward = None,
               			num_encoder_layers = 2,
						num_decoder_layers = 2,
						optimizers = torch.optim.Adam,
                        n_pred = pred_len,
                        attn= LocalAttention if attn == "local" else (FullAttention if attn == "full" else ProbAttention),
						attn_params = {"output_attention":True},
						metrics = {"loss": torch.nn.MSELoss(),
		   							"val_loss": torch.nn.MSELoss()}, lr=1e-3).double().to(device)



def vali(model,vali_loader, criterion,output_attention):
        """
        Evaluate `model` on `vali_loader` and return the average validation loss.

        Parameters
        ----------
        model : torch.nn.Module
            Model to evaluate.
        vali_loader : torch.utils.data.DataLoader
            Validation data loader.
        criterion : torch.nn.Module
            Loss function.
        output_attention : bool
            If True, also return the last batch's attention maps.

        Returns
        -------
        total_loss : float
            Average validation loss.
        enc_attns, dec_self_attns, dec_cross_attns, lastself_attn, lastcross_attn
            Only returned if output_attention is True (attention maps from the last batch).
        """
        model.eval()
        total_loss = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for batch_x, batch_y, labels in tqdm(vali_loader):
            batch_x, batch_y, labels = batch_x.to(device), batch_y.to(device), labels.to(device)
            pred, enc_attns, dec_self_attns, dec_cross_attns,lastself_attn, lastcross_attn = model(batch_x,batch_y)
            loss = criterion(pred.detach().cpu(), labels.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        model.train()
        if output_attention:
            return total_loss, enc_attns, dec_self_attns, dec_cross_attns,lastself_attn, lastcross_attn
        else:
            return total_loss
        
def select_criterion():
        criterion = nn.MSELoss()
        return criterion        
    
        
def train(train_epochs,model,train_loader,output_attention,early_stopping,vali_loader,path,model_optim):
	"""
	Train the non-federated baseline model for `train_epochs` epochs, with
	early stopping on the validation loss computed via `vali()`.

	Parameters
	----------
	train_epochs : int
		Maximum number of epochs to train for.
	model : torch.nn.Module
		Model to train.
	train_loader : torch.utils.data.DataLoader
		Training data loader.
	output_attention : bool
		Whether the model returns attention maps (passed through to `vali`).
	early_stopping : EarlyStopping
		Early stopping tracker; training stops once it flags `early_stop`.
	vali_loader : torch.utils.data.DataLoader
		Validation data loader, used for early stopping.
	path : str
		Directory where the best checkpoint is saved.
	model_optim : torch.optim.Optimizer
		Optimizer used to update `model`.

	Returns
	-------
	model : torch.nn.Module
		Model with the best (lowest validation loss) checkpoint loaded.
	"""
	criterion = select_criterion()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	for epoch in range(train_epochs):
		train_loss = []
		model.train()
		for batch_x, batch_y, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
			batch_x, batch_y, labels = batch_x.to(device), batch_y.to(device), labels.to(device)
			model_optim.zero_grad()
			pred, *_ = model(batch_x, batch_y)
			loss = criterion(pred, labels)
			train_loss.append(loss.item())

			loss.backward()
			model_optim.step()

		train_loss = np.average(train_loss)
		val_loss = vali(model, vali_loader, criterion, output_attention)
		if output_attention:
			val_loss = val_loss[0]
		print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")
		print("=" * 100)

		early_stopping(val_loss, model, path)
		if early_stopping.early_stop:
			print(f"Early stopping at epoch {epoch+1}")
			break
				
	best_model_path = path + "/" + "checkpoint.pth"
	torch.cuda.empty_cache()
	model.load_state_dict(torch.load(best_model_path))
	return model


def evaluate_checkpoint(model, dataloader, device, desc="Evaluating"):
	"""
	Run `model` over `dataloader` and compute forecasting error metrics.

	Parameters
	----------
	model : torch.nn.Module
		Model to evaluate (already on `device`).
	dataloader : torch.utils.data.DataLoader
		Data loader yielding (src, tgt, y, ...) batches.
	device : torch.device
		Device to run inference on.
	desc : str, optional
		Label shown on the tqdm progress bar.

	Returns
	-------
	metrics : dict
		Dictionary with keys "mae", "mse", "rmse", "mape", "mspe".
	"""
	model.eval()
	sum_abs_error = sum_squared_error = 0.0
	sum_abs_percentage_error = sum_squared_percentage_error = 0.0
	total_count = 0

	with torch.no_grad():
		for src, tgt, y, *_ in tqdm(dataloader, desc=desc):
			y_hat, *_ = model(src.to(device), tgt.to(device))
			preds = y_hat.detach().cpu().numpy()
			trues = y.detach().cpu().numpy()

			abs_error = np.abs(preds - trues)
			squared_error = (preds - trues) ** 2
			abs_percentage_error = abs_error / (np.abs(trues) + 1e-8)  # avoid division by zero

			sum_abs_error += abs_error.sum()
			sum_squared_error += squared_error.sum()
			sum_abs_percentage_error += abs_percentage_error.sum()
			sum_squared_percentage_error += (abs_percentage_error ** 2).sum()
			total_count += preds.size

	return {
		"mae": sum_abs_error / total_count,
		"mse": sum_squared_error / total_count,
		"rmse": np.sqrt(sum_squared_error / total_count),
		"mape": sum_abs_percentage_error / total_count,
		"mspe": sum_squared_percentage_error / total_count,
	}


def append_json_result(json_path: Path, entry: dict):
	"""Append `entry` to the JSON list stored at `json_path`, creating it if needed.
	Re-reads the file first so concurrent runs writing to the same file don't clobber
	each other's results."""
	results = json.loads(json_path.read_text()) if json_path.exists() else []
	results.append(entry)
	json_path.write_text(json.dumps(results, indent=4))


def save_checkpoint_with_meta(folder: Path, model, meta: dict):
	"""Save a model checkpoint together with a meta.json describing the run
	(model/attn/n_clients/seq_len/pred_len/alpha/...), so it can be reloaded
	later without having to parse any of this back out of the folder name."""
	folder.mkdir(parents=True, exist_ok=True)
	torch.save(model.state_dict(), folder / "checkpoint.pth")
	(folder / "meta.json").write_text(json.dumps(meta, indent=4))


if __name__ == '__main__':
	# Parse arguments
	parser = argparse.ArgumentParser()

	# Run mode and model
	parser.add_argument("--mode", choices=["train", "train_notFed", "test", "validate"], default="train")
	parser.add_argument("--model", choices=["transformer", "transformerNotFed", "transformerFedAvg", "transformerFedFRAcTS", "transformerFedAtt"], default="transformer")
	parser.add_argument("--attn", choices=["local", "prob", "full"], default="prob", help="Attention mechanism. 'prob' is the one used to produce the paper's reported results. 'local' is experimental/unstable, see known issues in model/attn.py.")

	# Model / training hyperparameters
	parser.add_argument("--seq_len", type=int, default=24)
	parser.add_argument("--pred_len", type=int, default=24)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
	parser.add_argument("--learning_rate", type=float, default=1e-3)
	parser.add_argument("--weight_decay", type=float, default=1e-4)

	# Federation setup
	parser.add_argument("--n_clients", type=int, default=10)
	parser.add_argument("--n_rounds", type=int, default=30)
	parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for quantity skew. None=equal split (default). Lower=more heterogeneous (e.g. 0.1, 0.5, 1.0)")

	# Aggregation strategy
	parser.add_argument("--agg_mode", choices=["", "Avg", "FRAcTS", "FedAtt"], default="")
	parser.add_argument("--agg_alpha", type=float, default=0.75, help="Alpha weight for the FRAcTS aggregator: balances server-similarity (alpha) vs centrality (1-alpha). Default=0.75")
	parser.add_argument("--max_repeats", type=int, default=None)
	parser.add_argument("--fedatt_stepsize", type=float, default=1.0, help="Stepsize (epsilon) for the FedAtt aggregator. Default=1.0")
	parser.add_argument("--fedatt_metric", type=int, default=2, help="Norm order used by FedAtt to measure server-client layer distance. Default=2")

	# Data
	parser.add_argument("--data_name", type=str)
	parser.add_argument("--data_root", type=str, default="./data", help="Directory containing the input <data_name>.csv file")
	parser.add_argument("--output_root", type=str, default="./results", help="Directory where checkpoints, results and JSON logs are written")
	parser.add_argument("--time_unit", type=str, default="h")
	parser.add_argument("--freq", type=int, default=1)
	parser.add_argument("--features", type=int, default=7)

	args = parser.parse_args()

	print("Parameters chosen: ")
	print("Mode: ", args.mode)
	print("Model: ", args.model)
	print("Data: ", args.data_name)
	print("Attn: ", args.attn)
	print("Features",args.features)
   
	name = args.model
	mode = args.mode
	window_size = args.seq_len
	features = args.features 
 
	output_attention = True
	for_forecasting = True
	for_transformer = True 
	for_tsfedl = False
	 
    # Load dataset
	data_train, timestamps_train, data_test, timestamps_test = None, None, None, None
	csv_path = f"{args.data_root}/{args.data_name}.csv"
 
 
	if mode=="train":
				path = f"{args.n_clients}_{args.seq_len}_{args.pred_len}"
				checkpoints_path =f"./checkpoints_{name}_{args.attn}_{args.data_name}/"
				if not os.path.exists(checkpoints_path):
					os.makedirs(checkpoints_path)
  
				checkpoints = checkpoints_path + f"checkpoints_{path}/"

				data_train, timestamps_train, _, _ = load_data_csv(csv_path)
				model = init_model(name,args.seq_len,args.pred_len,args.attn,features)
     			# Split train and validation
				validation_percentage = 0.2
				data_val, timestamps_val = data_train[:int(validation_percentage*len(data_train))], timestamps_train[:int(validation_percentage*len(timestamps_train))]
				data_train, timestamps_train = data_train[int(validation_percentage*len(data_train)):], timestamps_train[int(validation_percentage*len(timestamps_train)):]
				
				print("Data shape:", data_train.shape)
				print("Timestamps shape:", len(timestamps_train))

				timestamps_train_dt = pd.to_datetime(timestamps_train.astype(str)).to_numpy()
				timestamps_val_dt = pd.to_datetime(timestamps_val.astype(str)).to_numpy()

				# Small slice of the validation split used to compute the reference Q,K,V
				# activations on the server (see aggregate_fracts).
				qkv_ref_frac = 0.1
				n_qkv = max(1, int(len(data_val) * qkv_ref_frac))
				n_qkv = min(n_qkv, len(data_val) - 1)  # keep at least 1 point for evaluation

				qkv_ref_data = data_val[:n_qkv]
				qkv_ref_timestamps_dt = timestamps_val_dt[:n_qkv]

				# Remaining part of data_val, used for server-side evaluation
				data_val_eval = data_val[n_qkv:]
				timestamps_val_eval_dt = timestamps_val_dt[n_qkv:]

				# Create Flex Pool and Flex Dataset
				data = Dataset.from_array(data_train, timestamps_train_dt)
				if args.alpha is not None:
					# Dirichlet quantity skew: each client gets a different amount of data
					np.random.seed(42)
					dir_weights = np.random.dirichlet([args.alpha] * args.n_clients)
					# Ensure minimum viable size per client (at least seq_len + pred_len rows)
					min_frac = (args.seq_len + args.pred_len + 1) / len(data_train)
					dir_weights = np.maximum(dir_weights, min_frac)
					dir_weights = dir_weights / dir_weights.sum()  # re-normalize
					print(f"Dirichlet alpha={args.alpha} | weights: {np.round(dir_weights, 4)}")
					print(f"Samples per client: {(dir_weights * len(data_train)).astype(int)}")
					config = FedDatasetConfig(n_nodes=args.n_clients, weights=dir_weights.tolist())
					flex_dataset = FedDataDistribution.from_config(data, config)
				else:
					config = FedDatasetConfig(n_nodes=args.n_clients)
					flex_dataset = FedDataDistribution.from_config(data, config)
                
				# Assign the evaluation slice (excluding the QKV reference subset) to the server
				server_id = "server"
				flex_dataset[server_id] = Dataset.from_array(data_val_eval, timestamps_val_eval_dt)

				pool = FlexPool.client_server_pool(
					fed_dataset=flex_dataset,
					server_id="server",
					init_func=build_server_model,
					model=model,
				)
                
    
				output_model = train_n_rounds(
						pool,
						args.n_rounds,
						train_epochs=args.epochs,
						batch_size=args.batch_size,
						output_attention=output_attention,
						n_clients=args.n_clients,
						seq_len=args.seq_len,
						num_workers = args.num_workers,
						pred_len = args.pred_len,
						checkpoints=checkpoints,
						attn = args.attn,
						patience =args.patience,
						data_name = args.data_name,
                        freq = args.freq,
                        time_unit = args.time_unit,
						qkv_ref_data = qkv_ref_data,
						qkv_ref_timestamps = qkv_ref_timestamps_dt,
						agg_mode = args.agg_mode,
						agg_alpha = args.agg_alpha,
						max_repeats = args.max_repeats,
						fedatt_stepsize = args.fedatt_stepsize,
						fedatt_metric = args.fedatt_metric,
						)
				run_name = f"{name}_{args.attn}_{path}_{dt.date.today()}"
				checkpoint_dir = Path(args.output_root) / args.data_name / f"alpha_{args.alpha}" / "checkpoints" / run_name
				save_checkpoint_with_meta(checkpoint_dir, output_model, meta={
					"model": name,
					"attn": args.attn,
					"n_clients": args.n_clients,
					"seq_len": args.seq_len,
					"pred_len": args.pred_len,
					"alpha": args.alpha,
					"data_name": args.data_name,
					"date": str(dt.date.today()),
				})
				torch.cuda.empty_cache()

	elif mode =="test":
		exp_dir = Path(args.output_root) / args.data_name / f"alpha_{args.alpha}"
		checkpoints_dir = exp_dir / "checkpoints"
		json_result_path = exp_dir / "test_results.json"

		print(f"Testing checkpoints in: {checkpoints_dir}")
		print(f"Output JSON path: {json_result_path}")

		if not checkpoints_dir.exists():
			raise FileNotFoundError(f"Input checkpoints folder not found: {checkpoints_dir}")

		data_train, _, data_test, timestamps_test = load_data_csv(csv_path)
		timestamps_test_dt = pd.to_datetime(timestamps_test.astype(str)).to_numpy()

		validation_percentage = 0.2
		train_scaler = StandardScaler().fit(data_train[int(validation_percentage*len(data_train)):])
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		for run_dir in sorted(checkpoints_dir.iterdir()):
			meta_path = run_dir / "meta.json"
			if not meta_path.exists():
				print(f"Warning: no meta.json found in {run_dir}, skipping.")
				continue
			meta = json.loads(meta_path.read_text())

			print("Testing:", run_dir.name)
			model = load_model(meta["model"], str(run_dir / "checkpoint.pth"), meta["seq_len"], meta["pred_len"], meta["attn"], features).double().to(device)

			dataset_test = TimeSeriesDataset(torch.from_numpy(data_test), timestamps_test_dt,
											sequence_length=meta["seq_len"], npred=meta["pred_len"], for_forecasting=for_forecasting,
											for_transformer=for_transformer, for_tsfedl=for_tsfedl, for_test=True,time_unit=args.time_unit,freq = args.freq,
											scaler=train_scaler)
			dataloader_test =DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

			metrics = evaluate_checkpoint(model, dataloader_test, device, desc="Testing")
			print(f"MAE: {metrics['mae']}, MSE: {metrics['mse']}, RMSE: {metrics['rmse']}, MAPE: {metrics['mape']}, MSPE: {metrics['mspe']}")

			append_json_result(json_result_path, {
				"model": meta["model"], "attn": meta["attn"], "n_clients": meta["n_clients"],
				"seq_len": meta["seq_len"], "alpha": meta["alpha"],
				"mse": metrics["mse"], "mae": metrics["mae"],
			})


	elif mode == "validate":
		exp_dir = Path(args.output_root) / args.data_name / f"alpha_{args.alpha}"
		checkpoints_dir = exp_dir / "checkpoints"
		json_result_path = exp_dir / "val_results.json"

		print(f"Validating checkpoints in: {checkpoints_dir}")
		print(f"Output JSON path: {json_result_path}")

		if not checkpoints_dir.exists():
			raise FileNotFoundError(f"Input checkpoints folder not found: {checkpoints_dir}")

		data_train, timestamps_train, _, _ = load_data_csv(csv_path)
		validation_percentage = 0.2
		data_val, timestamps_val = data_train[:int(validation_percentage*len(data_train))], timestamps_train[:int(validation_percentage*len(timestamps_train))]
		timestamps_val_dt = pd.to_datetime(timestamps_val.astype(str)).to_numpy()

		qkv_ref_frac = 0.1
		n_qkv = max(1, int(len(data_val) * qkv_ref_frac))
		n_qkv = min(n_qkv, len(data_val) - 1)

		data_val_eval = data_val[n_qkv:]
		timestamps_val_eval_dt = timestamps_val_dt[n_qkv:]

		print("Validation data shape:", data_val_eval.shape)
		print("Validation timestamps shape:", len(timestamps_val_eval_dt))

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		for run_dir in sorted(checkpoints_dir.iterdir()):
			meta_path = run_dir / "meta.json"
			if not meta_path.exists():
				print(f"Warning: no meta.json found in {run_dir}, skipping.")
				continue
			meta = json.loads(meta_path.read_text())

			print("Validating:", run_dir.name)
			model = load_model(meta["model"], str(run_dir / "checkpoint.pth"), meta["seq_len"], meta["pred_len"], meta["attn"], features).double().to(device)

			dataset_val = TimeSeriesDataset(torch.from_numpy(data_val_eval), timestamps_val_eval_dt,
											sequence_length=meta["seq_len"], npred=meta["pred_len"], for_forecasting=for_forecasting,
											for_transformer=for_transformer, for_tsfedl=for_tsfedl, for_test=True,time_unit=args.time_unit,freq = args.freq)
			dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

			metrics = evaluate_checkpoint(model, dataloader_val, device, desc="Validating")
			print(f"MAE: {metrics['mae']}, MSE: {metrics['mse']}, RMSE: {metrics['rmse']}, MAPE: {metrics['mape']}, MSPE: {metrics['mspe']}")

			append_json_result(json_result_path, {
				"model": meta["model"], "attn": meta["attn"], "n_clients": meta["n_clients"],
				"seq_len": meta["seq_len"], "alpha": meta["alpha"],
				"mse": metrics["mse"], "mae": metrics["mae"],
			})

	elif mode == "train_notFed":
		path = f"{args.n_clients}_{args.seq_len}_{args.pred_len}"
		checkpoints_path =f"./checkpoints_{name}_{args.attn}/"
		if not os.path.exists(checkpoints_path):
					os.makedirs(checkpoints_path)

		checkpoints = checkpoints_path + f"checkpoints_{path}/" 
		if not os.path.exists(checkpoints):
			os.makedirs(checkpoints)
                    
		data_train, timestamps_train, _, _ = load_data_csv(csv_path)
		model = init_model(name,args.seq_len,args.pred_len,args.attn,features).double()
		# Split train and validation
		validation_percentage = 0.2
		data_val, timestamps_val = data_train[:int(validation_percentage*len(data_train))], timestamps_train[:int(validation_percentage*len(timestamps_train))]
		data_train, timestamps_train = data_train[int(validation_percentage*len(data_train)):], timestamps_train[int(validation_percentage*len(timestamps_train)):]
		
		print("Data shape:", data_train.shape)
		print("Timestamps shape:", len(timestamps_train))

		timestamps_train_dt = pd.to_datetime(timestamps_train.astype(str)).to_numpy()
		timestamps_val_dt = pd.to_datetime(timestamps_val.astype(str)).to_numpy()  	 	

		# Create dataset for train and validation
		dataset_train = TimeSeriesDataset(torch.from_numpy(data_train), timestamps_train_dt, 
											sequence_length=window_size, npred= args.pred_len, for_forecasting=for_forecasting, 
											for_transformer=for_transformer, for_tsfedl=for_tsfedl,time_unit=args.time_unit,freq = args.freq)
		
		dataset_val = TimeSeriesDataset(torch.from_numpy(data_val), timestamps_val_dt, 
										sequence_length=window_size, npred=args.pred_len, for_forecasting=for_forecasting, 
										for_transformer=for_transformer, for_tsfedl=for_tsfedl,time_unit = args.time_unit,freq = args.freq)
		
		# Create dataloader
		dataloader_train = DataLoader(dataset_train, batch_size= args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
		dataloader_val = DataLoader(dataset_val, batch_size= args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
		
		
		# Create callbacks
		early_stopping = EarlyStopping(patience=args.patience, verbose=True)
		model_optim = optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
		model_out = train(args.epochs,model,dataloader_train,output_attention,early_stopping,dataloader_val,checkpoints,model_optim)   
		       
		run_name = f"{name}_{args.attn}_{path}_{dt.date.today()}"
		checkpoint_dir = Path(args.output_root) / args.data_name / f"alpha_{args.alpha}" / "checkpoints" / run_name
		save_checkpoint_with_meta(checkpoint_dir, model_out, meta={
			"model": name,
			"attn": args.attn,
			"n_clients": args.n_clients,
			"seq_len": args.seq_len,
			"pred_len": args.pred_len,
			"alpha": args.alpha,
			"data_name": args.data_name,
			"date": str(dt.date.today()),
		})
		torch.cuda.empty_cache()