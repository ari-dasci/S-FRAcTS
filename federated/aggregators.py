import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
import copy
import math
from torch.utils.data import DataLoader

from flex.pool import aggregate_weights
from model.attn import AttentionLayer
from model.utils.timeseries_dataset import TimeSeriesDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


@aggregate_weights
def aggregate_with_fedavg(list_of_weights: list):
    """Standard FedAvg: element-wise mean of each layer's weights across clients."""
    agg_weights = []
    for layer_index in range(len(list_of_weights[0])):
        weights_per_layer = [weights[layer_index].cpu() for weights in list_of_weights]
        weights_per_layer = tl.stack(weights_per_layer)
        agg_layer = tl.mean(weights_per_layer, axis=0)
        agg_weights.append(torch.tensor(agg_layer))
    return agg_weights


def _build_qkv_ref_dataloader(
    qkv_ref_data,
    qkv_ref_timestamps,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int,
    time_unit: str,
    freq: str,
):
    """
    Build a DataLoader for the control dataset using the same settings as train/eval.
    """
    if qkv_ref_data is None or qkv_ref_timestamps is None:
        raise ValueError(
            "qkv_ref_data and qkv_ref_timestamps must be provided for QKVCosine aggregation."
        )

    data_np = (
        qkv_ref_data.to_numpy() if hasattr(qkv_ref_data, "to_numpy") else qkv_ref_data
    )
    ts_np = (
        qkv_ref_timestamps.to_numpy()
        if hasattr(qkv_ref_timestamps, "to_numpy")
        else qkv_ref_timestamps
    )

    dataset = TimeSeriesDataset(
        torch.from_numpy(data_np),
        ts_np,
        sequence_length=seq_len,
        npred=pred_len,
        for_forecasting=True,
        for_transformer=True,
        time_unit=time_unit,
        freq=freq,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return loader


def _collect_qkv_activations(model: nn.Module, dataloader: DataLoader, max_batches: int = None):
    """
    Run `model` over `dataloader` and collect Q, K, V activations for each AttentionLayer.

    Args:
        max_batches: if set, stop after collecting this many batches (saves RAM for
                     large datasets like IHEP where processing the full dataloader
                     would accumulate tens of GB of activation tensors).

    Returns:
        q_acts, k_acts, v_acts: dict[attn_idx] -> torch.Tensor of shape [T, H, D]
    """
    # Identify attention layers in a stable order
    attn_modules = []
    for name, module in model.named_modules():
        if isinstance(module, AttentionLayer):
            attn_modules.append((name, module))

    # Storage for activations: list per attention index, later concatenated
    q_lists = {idx: [] for idx in range(len(attn_modules))}
    k_lists = {idx: [] for idx in range(len(attn_modules))}
    v_lists = {idx: [] for idx in range(len(attn_modules))}

    hooks = []

    def make_hook(attn_idx: int):
        def hook(module, inputs, output):
            # inputs: (queries, keys, values, ...)
            queries, keys, values = inputs[0], inputs[1], inputs[2]

            with torch.no_grad():
                B, L, _ = queries.shape
                _, S, _ = keys.shape
                H = module.n_heads

                Q = module.query_projection(queries).view(B * L, H, -1)
                K = module.key_projection(keys).view(B * S, H, -1)
                V = module.value_projection(values).view(B * S, H, -1)

                q_lists[attn_idx].append(Q.detach().cpu())
                k_lists[attn_idx].append(K.detach().cpu())
                v_lists[attn_idx].append(V.detach().cpu())

        return hook

    # Register hooks
    for idx, (_, module) in enumerate(attn_modules):
        hooks.append(module.register_forward_hook(make_hook(idx)))

    try:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                # train/eval use (batch_x, batch_y, labels)
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    batch_x = batch[0].to(device)
                    batch_y = batch[1].to(device)
                else:
                    raise ValueError(
                        "Unexpected batch format in control dataloader for QKVCosine."
                    )
                _ = model(batch_x, batch_y)
                del batch_x, batch_y
    finally:
        for h in hooks:
            h.remove()

    # Concatenate lists per attention layer
    q_acts = {}
    k_acts = {}
    v_acts = {}
    for idx in range(len(attn_modules)):
        if q_lists[idx]:
            q_acts[idx] = torch.cat(q_lists[idx], dim=0)  # [T, H, D]
            k_acts[idx] = torch.cat(k_lists[idx], dim=0)
            v_acts[idx] = torch.cat(v_lists[idx], dim=0)
        else:
            # No activations collected: return empty tensors
            q_acts[idx] = torch.empty(0)
            k_acts[idx] = torch.empty(0)
            v_acts[idx] = torch.empty(0)
    # Free intermediate lists
    del q_lists, k_lists, v_lists
    torch.cuda.empty_cache()

    return q_acts, k_acts, v_acts, attn_modules


def _extract_norm_vecs(q_acts, k_acts, v_acts, n_heads_layer):
    """Extract normalized mean-pooled per-head activation vectors from raw
    Q/K/V activation tensors.

    For each head h the raw activations have shape (T, D), where T is the
    total number of tokens across all batches. They are averaged over T and
    L2-normalized, giving one (D,) vector per head that is used to compute
    cosine similarities between clients and server.

    Returns:
        q_vecs, k_vecs, v_vecs: each is a list of length n_heads_layer
            containing 1-D float32 numpy arrays of shape (D,).
    """
    def _norm(x):
        return x / (x.norm() + 1e-8)

    q_vecs, k_vecs, v_vecs = [], [], []
    for h in range(n_heads_layer):
        # Mean-pool across tokens -> (D,), then normalize
        q_vecs.append(_norm(q_acts[:, h, :].mean(dim=0)).float().numpy())
        k_vecs.append(_norm(k_acts[:, h, :].mean(dim=0)).float().numpy())
        v_vecs.append(_norm(v_acts[:, h, :].mean(dim=0)).float().numpy())
    return q_vecs, k_vecs, v_vecs


def aggregate_fracts(
    aggregator_flex_model,
    servers_flex_models,
    *,
    qkv_ref_data,
    qkv_ref_timestamps,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int,
    time_unit: str,
    freq: str,
    round: int = 1,
    max_batches: int = 100,
    max_repeats: int = None,
    alpha: float = 0.75,
):
    """
    QKV aggregation combining pairwise centrality and server-guided similarity.

    Round 1 - pairwise centrality only:
        For each attention layer and head position, compute pairwise cosine
        similarity of Q/K/V activations across all clients (no server reference
        since the server model is random at this stage).  Select the client
        head with the highest centrality score (most similar to the rest of
        the pool). Non-attention layers use FedAvg.

    Round >= 2 - combined score:
        For every (server_head i, client_head j) pair the assignment score is:

            score[i, j] = alpha * sim_server[i, j]
                        + (1 - alpha) * centrality_norm[j]

        where:
        - sim_server[i, j]: cosine similarity between the Q/K/V activations
          of server head i and client head j.
        - centrality_norm[j]: pairwise centrality of client head j across
          the client pool, normalized to [0, 1]. Client heads far from the
          consensus receive low centrality.

        Assignment uses a capacity-aware greedy algorithm
        (max_repeats defaults to max(1, N//2)).

    Parameters
    ----------
    alpha : float, default 0.75
        Weight for the server-similarity term.  (1 - alpha) weights the
        centrality term.  Set alpha=1.0 to select purely by server similarity;
        set alpha=0.0 for pure centrality selection guided by the server shape.
    """
    list_of_weights_dict = aggregator_flex_model["weights"]
    if not list_of_weights_dict:
        raise ValueError("No client weights found in aggregator_flex_model['weights'].")

    state_keys = list(list_of_weights_dict[0].keys())
    n_clients = len(list_of_weights_dict)

    if "server" not in servers_flex_models:
        raise ValueError(
            "Expected a single server with key 'server' in servers_flex_models."
        )
    server_model = servers_flex_models["server"]["model"].to(device)

    control_loader = _build_qkv_ref_dataloader(
        qkv_ref_data=qkv_ref_data,
        qkv_ref_timestamps=qkv_ref_timestamps,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
        num_workers=num_workers,
        time_unit=time_unit,
        freq=freq,
    )

    # Discover attention modules once (from server architecture)
    attn_modules = []
    for name, module in server_model.named_modules():
        if isinstance(module, AttentionLayer):
            attn_modules.append((name, module))

    attn_param_names = []
    for name, module in attn_modules:
        q_name = f"{name}.query_projection.weight"
        k_name = f"{name}.key_projection.weight"
        v_name = f"{name}.value_projection.weight"
        if (
            q_name not in state_keys
            or k_name not in state_keys
            or v_name not in state_keys
        ):
            raise ValueError(
                f"Could not find Q/K/V weights for attention module '{name}' in state_dict."
            )
        attn_param_names.append((q_name, k_name, v_name, module.n_heads))

    # Collect client activations (both rounds need them)
    client_norm_vecs = []
    for ci, client_state in enumerate(list_of_weights_dict):
        print(f"  Collecting QKV activations for client {ci+1}/{n_clients} (Round {round})")
        client_model = copy.deepcopy(server_model)
        client_model.load_state_dict(client_state, strict=True)

        q_acts, k_acts, v_acts, _ = _collect_qkv_activations(client_model, control_loader, max_batches=max_batches)
        del client_model
        torch.cuda.empty_cache()

        per_layer = {}
        for attn_idx, (_, _, _, n_heads_layer) in enumerate(attn_param_names):
            if attn_idx in q_acts and q_acts[attn_idx].numel() > 0:
                per_layer[attn_idx] = _extract_norm_vecs(
                    q_acts[attn_idx], k_acts[attn_idx], v_acts[attn_idx], n_heads_layer
                )
            else:
                per_layer[attn_idx] = None
        del q_acts, k_acts, v_acts
        gc.collect()

        client_norm_vecs.append(per_layer)

    # Round 1: pairwise centrality only (server not yet trained)
    if round == 1:
        print("Round 1: pairwise centrality head selection")

        aggregated_state = {}
        for key in state_keys:
            aggregated_state[key] = torch.zeros_like(list_of_weights_dict[0][key])

        for attn_idx, (q_name, k_name, v_name, n_heads_layer) in enumerate(attn_param_names):
            if client_norm_vecs[0][attn_idx] is None:
                for wname in (q_name, k_name, v_name):
                    aggregated_state[wname].copy_(
                        torch.stack([w[wname] for w in list_of_weights_dict], dim=0).mean(0)
                    )
                continue

            d_q = list_of_weights_dict[0][q_name].shape[0] // n_heads_layer
            d_k = list_of_weights_dict[0][k_name].shape[0] // n_heads_layer
            d_v = list_of_weights_dict[0][v_name].shape[0] // n_heads_layer

            Wq_agg = list_of_weights_dict[0][q_name].clone()
            Wk_agg = list_of_weights_dict[0][k_name].clone()
            Wv_agg = list_of_weights_dict[0][v_name].clone()

            for h in range(n_heads_layer):
                if n_clients == 1:
                    selected_client = 0
                else:
                    centrality_scores = np.zeros(n_clients, dtype=float)
                    for ci in range(n_clients):
                        qi = client_norm_vecs[ci][attn_idx][0][h]
                        ki = client_norm_vecs[ci][attn_idx][1][h]
                        vi = client_norm_vecs[ci][attn_idx][2][h]
                        for cj in range(ci + 1, n_clients):
                            qj = client_norm_vecs[cj][attn_idx][0][h]
                            kj = client_norm_vecs[cj][attn_idx][1][h]
                            vj = client_norm_vecs[cj][attn_idx][2][h]
                            sim = (
                                float(np.dot(qi, qj))
                                + float(np.dot(ki, kj))
                                + float(np.dot(vi, vj))
                            ) / 3.0
                            centrality_scores[ci] += sim
                            centrality_scores[cj] += sim
                    centrality_scores /= max(1, n_clients - 1)
                    selected_client = int(np.argmax(centrality_scores))

                print(f"Round 1 | layer={q_name} | head={h} | selected_client={selected_client}")

                q_s, q_e = h * d_q, (h + 1) * d_q
                k_s, k_e = h * d_k, (h + 1) * d_k
                v_s, v_e = h * d_v, (h + 1) * d_v

                Wq_agg[q_s:q_e, :] = list_of_weights_dict[selected_client][q_name][q_s:q_e, :]
                Wk_agg[k_s:k_e, :] = list_of_weights_dict[selected_client][k_name][k_s:k_e, :]
                Wv_agg[v_s:v_e, :] = list_of_weights_dict[selected_client][v_name][v_s:v_e, :]

            aggregated_state[q_name].copy_(Wq_agg)
            aggregated_state[k_name].copy_(Wk_agg)
            aggregated_state[v_name].copy_(Wv_agg)

        attn_weight_keys = (
            {n[0] for n in attn_param_names}
            | {n[1] for n in attn_param_names}
            | {n[2] for n in attn_param_names}
        )
        for key in state_keys:
            if key not in attn_weight_keys:
                stacked = torch.stack([w[key] for w in list_of_weights_dict], dim=0)
                aggregated_state[key].copy_(stacked.mean(0))

        aggregator_flex_model["aggregated_weights"] = [
            aggregated_state[key] for key in state_keys
        ]
        aggregator_flex_model["weights"] = []
        del client_norm_vecs, aggregated_state
        torch.cuda.empty_cache()
        gc.collect()
        print("Round 1 done: representative heads selected")
        return

    # Round >= 2: combined score of server similarity and centrality
    print(f"Round {round}: combined score, alpha={alpha}")

    # Collect server activations -> norm vecs -> free raw tensors
    server_q, server_k, server_v, _ = _collect_qkv_activations(server_model, control_loader, max_batches=max_batches)
    server_norm_vecs = {}
    for attn_idx, (_, _, _, n_heads_layer) in enumerate(attn_param_names):
        if attn_idx in server_q and server_q[attn_idx].numel() > 0:
            server_norm_vecs[attn_idx] = _extract_norm_vecs(
                server_q[attn_idx], server_k[attn_idx], server_v[attn_idx], n_heads_layer
            )
        else:
            server_norm_vecs[attn_idx] = None
    del server_q, server_k, server_v
    torch.cuda.empty_cache()
    gc.collect()

    aggregated_state = {}
    for key in state_keys:
        aggregated_state[key] = torch.zeros_like(list_of_weights_dict[0][key])

    for attn_idx, (q_name, k_name, v_name, n_heads_layer) in enumerate(attn_param_names):
        # Fallback to average if server or client activations unavailable
        if server_norm_vecs[attn_idx] is None or client_norm_vecs[0][attn_idx] is None:
            for wname in (q_name, k_name, v_name):
                aggregated_state[wname].copy_(
                    torch.stack([w[wname] for w in list_of_weights_dict], dim=0).mean(0)
                )
            continue

        srv_q_vecs, srv_k_vecs, srv_v_vecs = server_norm_vecs[attn_idx]
        C = n_clients
        N = n_heads_layer
        M = C * N  # total client-head columns

        # Server-similarity matrix  [N x M]
        sim_server = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for c in range(C):
                cli_q, cli_k, cli_v = client_norm_vecs[c][attn_idx]
                for h in range(N):
                    j = c * N + h
                    sim_server[i, j] = (
                        float(np.dot(srv_q_vecs[i], cli_q[h]))
                        + float(np.dot(srv_k_vecs[i], cli_k[h]))
                        + float(np.dot(srv_v_vecs[i], cli_v[h]))
                    ) / 3.0

        # Per-column centrality score  [M]
        # centrality_raw[j] = mean pairwise cosine similarity of client-head j
        # with every other client-head in the same head-position h = j % N.
        centrality_raw = np.zeros(M, dtype=np.float32)
        for c in range(C):
            cli_q, cli_k, cli_v = client_norm_vecs[c][attn_idx]
            for h in range(N):
                j = c * N + h
                score = 0.0
                count = 0
                for c2 in range(C):
                    if c2 == c:
                        continue
                    cli_q2, cli_k2, cli_v2 = client_norm_vecs[c2][attn_idx]
                    score += (
                        float(np.dot(cli_q[h], cli_q2[h]))
                        + float(np.dot(cli_k[h], cli_k2[h]))
                        + float(np.dot(cli_v[h], cli_v2[h]))
                    ) / 3.0
                    count += 1
                centrality_raw[j] = score / max(1, count)

        # Min-max normalize both terms to [0, 1] so they are on the same scale
        def _minmax(arr):
            lo, hi = arr.min(), arr.max()
            return np.ones_like(arr) if hi <= lo else (arr - lo) / (hi - lo)

        sim_server_norm = _minmax(sim_server)        # [N, M]
        centrality_norm = _minmax(centrality_raw)    # [M]

        # Combined score
        # Broadcast centrality_norm across rows (same for all server heads)
        score_matrix = alpha * sim_server_norm + (1.0 - alpha) * centrality_norm[np.newaxis, :]

        # Capacity-aware greedy assignment
        if max_repeats is None:
            cap = max(1, N // 2)
        else:
            cap = int(max_repeats)
        if cap * M < N:
            cap = math.ceil(N / M)

        capacities = np.full(M, cap, dtype=int)
        assignment = {}
        row_order = np.argsort(-score_matrix.max(axis=1))
        for i in row_order:
            available = np.where(capacities > 0)[0]
            if available.size == 0:
                break
            col_scores = score_matrix[i, available] + np.random.rand(available.size) * 1e-8
            chosen = int(available[np.argmax(col_scores)])
            assignment[int(i)] = chosen
            capacities[chosen] -= 1

        # Fill any unassigned rows
        for r in range(N):
            if r not in assignment:
                assignment[r] = int(np.argmax(score_matrix[r, :]))

        # Copy selected head slices into aggregated weights
        d_q = list_of_weights_dict[0][q_name].shape[0] // N
        d_k = list_of_weights_dict[0][k_name].shape[0] // N
        d_v = list_of_weights_dict[0][v_name].shape[0] // N

        Wq_agg = list_of_weights_dict[0][q_name].clone()
        Wk_agg = list_of_weights_dict[0][k_name].clone()
        Wv_agg = list_of_weights_dict[0][v_name].clone()

        for h in range(N):
            col = assignment[h]
            client_idx = col // N
            head_idx   = col % N

            q_s, q_e = head_idx * d_q, (head_idx + 1) * d_q
            k_s, k_e = head_idx * d_k, (head_idx + 1) * d_k
            v_s, v_e = head_idx * d_v, (head_idx + 1) * d_v

            Wq_agg[q_s:q_e, :] = list_of_weights_dict[client_idx][q_name][q_s:q_e, :]
            Wk_agg[k_s:k_e, :] = list_of_weights_dict[client_idx][k_name][k_s:k_e, :]
            Wv_agg[v_s:v_e, :] = list_of_weights_dict[client_idx][v_name][v_s:v_e, :]

            print(
                f"Round {round} | layer={q_name} | server_head={h} | "
                f"client={client_idx} head={head_idx} | "
                f"score={score_matrix[h, col]:.4f} "
                f"(srv_norm={sim_server_norm[h, col]:.4f}, cent_norm={centrality_norm[col]:.4f})"
            )

        aggregated_state[q_name].copy_(Wq_agg)
        aggregated_state[k_name].copy_(Wk_agg)
        aggregated_state[v_name].copy_(Wv_agg)

    # Non-attention layers: FedAvg
    attn_weight_keys = (
        {n[0] for n in attn_param_names}
        | {n[1] for n in attn_param_names}
        | {n[2] for n in attn_param_names}
    )
    for key in state_keys:
        if key not in attn_weight_keys:
            stacked = torch.stack([w[key] for w in list_of_weights_dict], dim=0)
            aggregated_state[key].copy_(stacked.mean(0))

    aggregator_flex_model["aggregated_weights"] = [
        aggregated_state[key] for key in state_keys
    ]
    aggregator_flex_model["weights"] = []
    del client_norm_vecs, server_norm_vecs, aggregated_state
    torch.cuda.empty_cache()
    gc.collect()


def aggregate_fedatt(
    aggregator_flex_model,
    servers_flex_models,
    *,
    round: int = 1,
    stepsize: float = 1.0,
    metric: int = 2,
):
    """
    FedAtt (Ji et al., IJCNN 2019) - attention-based aggregator, used as a
    comparison baseline in the paper.

    General idea (the "aggregate_att" algorithm from their official
    repository): instead of averaging client weights with equal weight
    (FedAvg), for each layer we measure how "far" each client is from the
    current global model (norm of the weight difference) and use a softmax
    over those distances to weight the server update: clients that are more
    different receive more "attention" in the update.

    Parameters
    ----------
    stepsize: step size of the server update (epsilon).
    metric: order of the norm used as the server-client distance (the `ord`
        argument of torch.linalg.norm; 2 = Euclidean/spectral norm, as in the paper).
    """
    w_clients = aggregator_flex_model["weights"]
    if not w_clients:
        raise ValueError("No client weights found in aggregator_flex_model['weights'].")
    if "server" not in servers_flex_models:
        raise ValueError("Expected a single server with key 'server' in servers_flex_models.")

    w_server = servers_flex_models["server"]["model"].state_dict()
    state_keys = list(w_server.keys())
    m = len(w_clients)
    att = {}

    # Step 1: server-client distance (norm) per layer -> "attention energy"
    for k in state_keys:
        w_srv = w_server[k]
        if not w_srv.is_floating_point():
            continue
        att[k] = torch.zeros(m, device=w_srv.device)
        for i in range(m):
            w_cli = w_clients[i][k].to(w_srv.device)
            diff = w_srv - w_cli
            # Flatten weights with more than 2 dimensions (e.g. Conv1d)
            att[k][i] = torch.linalg.norm(diff if diff.dim() <= 2 else diff.reshape(-1), ord=metric)

    # Step 2: softmax over the distances + server update
    w_next = {}
    for k in state_keys:
        w_srv = w_server[k]
        if not w_srv.is_floating_point():
            w_next[k] = w_srv.clone()
            continue

        alphas = F.softmax(att[k], dim=0)
        att_weight = torch.zeros_like(w_srv)
        for i in range(m):
            w_cli = w_clients[i][k].to(w_srv.device)
            att_weight += (w_srv - w_cli) * alphas[i]

        w_next[k] = w_srv - att_weight * stepsize

    aggregator_flex_model["aggregated_weights"] = [w_next[key] for key in state_keys]
    aggregator_flex_model["weights"] = []
    del w_next
    torch.cuda.empty_cache()
    gc.collect()

