import torch
import torch.nn as nn
import gc
import copy
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from model.utils.tools import EarlyStopping
from model.utils.timeseries_dataset import TimeSeriesDataset
from flex.data import Dataset
from flex.model import FlexModel
from flex.pool import (
    init_server_model,
    deploy_server_model,
    collect_clients_weights,
    set_aggregated_weights,
)
from federated.aggregators import (
    aggregate_with_fedavg,
    aggregate_fracts,
    aggregate_fedatt,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


@init_server_model
def build_server_model(model, lr=1e-3, weight_decay=0.0001):
    """Initialize the server's FlexModel: applies Xavier weight init and wraps
    the model together with its optimizer and loss for the FL pool."""
    model.apply(init_weights)
    flex_model = FlexModel()
    flex_model["model"] = model.to(device)
    flex_model["optimizer"] = Adam
    flex_model["optimizer_kwargs"] = {"lr": lr}
    flex_model["criterion"] = nn.MSELoss()
    return flex_model


@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel):
    """Deep-copy the server's FlexModel so each client starts the round from
    the current global model."""
    return copy.deepcopy(server_flex_model.copy())


def train(client_model, client_data, checkpoints, **kwargs):
    """Train a single client's (or the server's) local model for `kwargs['train_epochs']`
    epochs, with early stopping, and return the trained model."""
    if client_model.actor_id == "server":
        print("Training model at server.")
    else:
        print("Training model at client.", client_model.actor_id)
    
    data_train = client_data.X_data.to_numpy()
    timestamps_train = client_data.y_data.to_numpy()
    
    
    dataset_train = TimeSeriesDataset(torch.from_numpy(data_train), timestamps_train, 
												sequence_length=kwargs["seq_len"], npred= kwargs["pred_len"], for_forecasting=True, 
												for_transformer= True,time_unit=kwargs["time_unit"],freq = kwargs["freq"])
    dataloader_train = DataLoader(dataset_train, batch_size= kwargs["batch_size"], shuffle=True, num_workers=0, pin_memory=False)
    
    path = os.path.join(checkpoints, str(client_model.actor_id))
    if not os.path.exists(path):
        os.makedirs(path)
        
    model = client_model["model"].to(device)
    early_stopping = EarlyStopping(patience=kwargs["patience"], verbose=True)       
    criterion = client_model["criterion"].to(device)
    optimizer = client_model["optimizer"](
        model.parameters(), **client_model["optimizer_kwargs"]
    )
    round = 1
        
    for epoch in range(kwargs["train_epochs"]):
            train_loss = []
            model = model.train()
            
            for batch_x, batch_y, labels in tqdm(dataloader_train, desc=f"Epoch {epoch+1}"):
                batch_x, batch_y, labels = batch_x.to(device), batch_y.to(device), labels.to(device)
                optimizer.zero_grad()
                pred, *_ = model(batch_x, batch_y)
                loss = criterion(pred, labels)
                train_loss.append(loss.item())
                
                loss.backward()
                optimizer.step()
            
            train_loss = np.average(train_loss)
            print(f"Epoch: {epoch+1}, Loss: {train_loss}")
            print("=" * 100)

            early_stopping(train_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping:round",round)
                break
       
    best_model_path = path + "/" + "checkpoint.pth"	
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(best_model_path))

    # Free memory: move model to CPU, delete training artifacts
    model.cpu()
    del dataloader_train, dataset_train, optimizer, criterion, early_stopping
    torch.cuda.empty_cache()
    gc.collect()
        
    return model



@collect_clients_weights
def get_clients_weights(client_flex_model: FlexModel):
    """Collect a client's full state_dict, keyed by parameter name (used by
    aggregators that need to identify specific layers, e.g. FRAcTS/FedAtt)."""
    with torch.no_grad():
        weight_dict = client_flex_model["model"].state_dict()
    return weight_dict


@collect_clients_weights
def get_clients_weights_fedavg(client_flex_model: FlexModel):
    """Collect a client's state_dict as an ordered list of tensors (the format
    expected by aggregate_with_fedavg)."""
    weight_dict = client_flex_model["model"].state_dict()
    return [weight_dict[name] for name in weight_dict]


@set_aggregated_weights
def set_agreggated_weights_to_server(server_flex_model: FlexModel, aggregated_weights):
    """Copy the aggregated weights produced by an aggregator back into the
    server's model, in place."""
    with torch.no_grad():
        weight_dict = server_flex_model["model"].state_dict()
        for layer_key, new in zip(weight_dict, aggregated_weights):
            weight_dict[layer_key].copy_(new)


def evaluate_global_model(server_flex_model: FlexModel, val_data: Dataset, **kwargs):
    """Evaluate the current server (global) model on the held-out validation
    set and return the average loss for the round."""
    model = server_flex_model["model"].to(device)
    criterion = server_flex_model["criterion"].to(device)
    model.eval()
    total_loss = []
    data_val = val_data.X_data.to_numpy()
    timestamps_val = val_data.y_data.to_numpy()

    dataset_val = TimeSeriesDataset(torch.from_numpy(data_val), timestamps_val, 
											sequence_length=kwargs["seq_len"], npred=kwargs["pred_len"], for_forecasting=True, 
											for_transformer=True,time_unit=kwargs["time_unit"],freq = kwargs["freq"])
    dataloader_val = DataLoader(dataset_val, batch_size= kwargs["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
    
    
    for batch_x, batch_y, labels in tqdm(dataloader_val):
            batch_x, batch_y, labels = batch_x.to(device), batch_y.to(device), labels.to(device)
            pred, _, _, _,_, _ = model(batch_x,batch_y)
            loss = criterion(pred.detach().cpu(), labels.detach().cpu())
            total_loss.append(loss)
            
    total_loss = np.average(total_loss)
    model.train()

    return float(total_loss)

def _free_client_models(pool):
    """Delete client models from pool to free RAM. They will be
    re-created by copy_server_model_to_clients at the next round."""
    for cid in list(pool.clients._models.keys()):
        fm = pool.clients._models[cid]
        if "model" in fm:
            fm["model"].cpu()
            del fm["model"]
        if "criterion" in fm:
            del fm["criterion"]
        if "optimizer" in fm:
            del fm["optimizer"]
        if "optimizer_kwargs" in fm:
            del fm["optimizer_kwargs"]
    torch.cuda.empty_cache()
    gc.collect()


def _free_aggregator_state(pool):
    """Clear aggregator intermediate state to free RAM."""
    for aid in list(pool.aggregators._models.keys()):
        am = pool.aggregators._models[aid]
        if "weights" in am:
            del am["weights"]
        if "aggregated_weights" in am:
            del am["aggregated_weights"]
    gc.collect()


def train_n_rounds(pool, n_rounds, **kwargs):
    """Run the federated training loop for up to `n_rounds` rounds: deploy the
    server model to clients, train locally, aggregate with the strategy given
    in kwargs['agg_mode'] (Avg/FRAcTS/FedAtt), and evaluate the global model
    each round. Stops early if the validation metric does not improve for
    `patience` consecutive rounds."""
    patience = 10  # Number of rounds without improvement before stopping
    best_metric = float("inf")
    patience_counter = 0
    print("Aggregation Mode",kwargs["agg_mode"])

    for i in range(n_rounds):
        kwargs["round"] = i + 1
        print("+" * 100)
        print(f"\nRunning round: {i+1} of {n_rounds}")
        pool.servers.map(copy_server_model_to_clients, pool.clients)
        pool.clients.map(train, **kwargs)  # clients or selected_clients

        if kwargs["agg_mode"] == "Avg":
            pool.aggregators.map(get_clients_weights_fedavg, pool.clients)
            _free_client_models(pool)  # Free client model copies before aggregation
            pool.aggregators.map(aggregate_with_fedavg)
        elif kwargs["agg_mode"] == "FRAcTS":
            pool.aggregators.map(get_clients_weights, pool.clients)
            _free_client_models(pool)  # Free client model copies before aggregation
            pool.aggregators.map(
                aggregate_fracts,
                pool.servers,
                qkv_ref_data=kwargs.get("qkv_ref_data", None),
                qkv_ref_timestamps=kwargs.get("qkv_ref_timestamps", None),
                seq_len=kwargs["seq_len"],
                pred_len=kwargs["pred_len"],
                batch_size=kwargs["batch_size"],
                num_workers=kwargs["num_workers"],
                time_unit=kwargs["time_unit"],
                freq=kwargs["freq"],
                round=kwargs["round"],
                max_batches=1,
                max_repeats=kwargs.get("max_repeats", None),
                alpha=kwargs.get("agg_alpha", 0.75),
            )
        elif kwargs["agg_mode"] == "FedAtt":
            pool.aggregators.map(get_clients_weights, pool.clients)
            _free_client_models(pool)  # Free client model copies before aggregation
            pool.aggregators.map(
                aggregate_fedatt,
                pool.servers,
                round=kwargs["round"],
                stepsize=kwargs.get("fedatt_stepsize", 1.0),
                metric=kwargs.get("fedatt_metric", 2),
            )
        else:    
            raise ValueError(f"Unsupported aggregation mode: {kwargs['agg_mode']}")
        
        pool.aggregators.map(set_agreggated_weights_to_server, pool.servers)
        _free_aggregator_state(pool)  # Clear aggregator intermediate weights
        current_metric = pool.servers.map(evaluate_global_model, **kwargs)
        
        # Free memory between rounds
        torch.cuda.empty_cache()
        gc.collect()
        
        if current_metric[0] < best_metric:
            best_metric = current_metric[0]
            patience_counter = 0  # Restart the patience counter if there's an improvement
        else:
            patience_counter += 1
            print(f"No improvement in the metric. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Stopping training after {i+1} rounds. No improvement in {patience} consecutive rounds.")
            break  # Stop training if patience is exceeded
    return pool.servers._models["server"]["model"]
