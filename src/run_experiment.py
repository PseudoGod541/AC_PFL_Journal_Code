import os
import argparse, csv, warnings

# Now that 'os' is imported, we can set the environment variables safely
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import numpy as np
import tensorflow as tf
import flwr as fl
import pickle
import shutil
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from config import DATASETS, ALL_FEATURE_COLS, SEQUENCE_LENGTH, RUL_CAP, NUM_ROUNDS, NUM_CLIENTS, NUM_CLUSTERS, RECLUSTER_EVERY
from utils import (
    get_engine_clusters, create_hard_partitions, compute_cluster_assignments,
    load_cmapss_data, assign_test_engines_to_clusters, evaluate_clustered_model, compute_acpfl_cluster_assignments,
    get_condition_labels, filter_engines_by_condition
)
from preprocess import calculate_rul, create_sequences

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def already_completed(output_file, method, dataset, seed):
    """Check if this exact run already exists in results CSV."""
    if not os.path.exists(output_file):
        return False
    import pandas as pd
    try:
        df = pd.read_csv(output_file)
        return any(
            (df['method'] == method) &
            (df['dataset'] == dataset) &
            (df['seed'] == seed)
        )
    except Exception:
        return False

HEAD_REGISTRY_DIR = "head_registry"
os.makedirs(HEAD_REGISTRY_DIR, exist_ok=True)

def save_head(cid, weights):
    with open(f"{HEAD_REGISTRY_DIR}/client_{cid}.pkl", "wb") as f:
        pickle.dump(weights, f)

def load_head(cid):
    path = f"{HEAD_REGISTRY_DIR}/client_{cid}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def nasa_rul_score(y_true, y_pred):
    error = y_pred - y_true
    scores = np.where(error < 0, np.exp(-error/13)-1, np.exp(error/10)-1)
    return float(np.sum(scores))

# ── Model ── (now with LayerNormalization and correct base/head split)
def build_model(input_shape):
    base = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.LayerNormalization(),
    ])
    head = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1),
    ])
    inp = tf.keras.Input(shape=input_shape)
    out = head(base(inp))
    model = tf.keras.Model(inp, out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model, base, head

# ── Routing helper ──────────────────────────────────────────────────────────
def route_predictions(client_preds_dict, test_engine_clusters,
                      client_to_cluster, num_clients, y_test):
    all_preds = np.zeros(len(y_test))
    for idx, cluster_id in test_engine_clusters.items():
        clients_in_cluster = [
            cid for cid in range(num_clients)
            if client_to_cluster.get(str(cid)) == cluster_id
        ]
        if clients_in_cluster:
            all_preds[idx] = np.mean([
                client_preds_dict[cid][idx]
                for cid in clients_in_cluster
            ])
        else:
            all_preds[idx] = np.mean([
                client_preds_dict[cid][idx]
                for cid in range(num_clients)
            ])
    return all_preds

# ── Data ────────────────────────────────────────────────────────────────────
def load_dataset(dataset_key, seed, condition_subset=None):
    """
    condition_subset: optional set/list of condition-IDs (0..5) to restrict
    this dataset to, e.g. {0,1,2} for a C=3 run. None = unfiltered (default,
    byte-for-byte identical behavior to before this param existed).
    Condition IDs are recovered by clustering setting1/2/3 -- a SEPARATE
    clustering from the sensor-based client-partitioning clusters below.
    """
    paths = DATASETS[dataset_key]
    df = load_cmapss_data(paths["train"])
    df = calculate_rul(df, RUL_CAP)
    for col in ALL_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # -- Condition-regime filter (must run on raw, pre-MinMax setting columns,
    # BEFORE scaler.fit_transform overwrites them below) --
    cond_kmeans = cond_scaler = cond_remap = None
    if condition_subset is not None:
        unit_to_condition, cond_kmeans, cond_scaler, cond_remap = get_condition_labels(df)
        df = filter_engines_by_condition(df, unit_to_condition, set(condition_subset))

    scaler = MinMaxScaler()
    df[ALL_FEATURE_COLS] = scaler.fit_transform(df[ALL_FEATURE_COLS])

    unit_to_cluster, kmeans_model, cluster_scaler, sensor_cols = \
        get_engine_clusters(df, NUM_CLUSTERS[dataset_key])
    partitions, client_to_cluster = create_hard_partitions(unit_to_cluster, NUM_CLIENTS, NUM_CLUSTERS[dataset_key])

    client_data = []
    for part in partitions:
        part = list(part)
        if len(part) < 2:
            train_units, val_units = part, part
        else:
            train_units, val_units = train_test_split(part, test_size=0.2, random_state=seed)
        X_tr, y_tr = create_sequences(df[df['unit_id'].isin(train_units)], ALL_FEATURE_COLS, SEQUENCE_LENGTH)
        X_val, y_val = create_sequences(df[df['unit_id'].isin(val_units)], ALL_FEATURE_COLS, SEQUENCE_LENGTH)
        client_data.append((X_tr, y_tr, X_val, y_val))

    # -- Test set. rul_all[i] is RUL for unit_id (i+1) in the ORIGINAL,
    # unfiltered test file -- must re-index by original unit_id, not by
    # position in a (possibly filtered) unit list. --
    df_test = load_cmapss_data(paths["test"])
    rul_all = np.clip(np.loadtxt(paths["rul"]).flatten(), 0, RUL_CAP)
    orig_test_ids = sorted(df_test['unit_id'].unique())
    assert orig_test_ids == list(range(1, len(orig_test_ids) + 1)), \
        "Expected 1-indexed contiguous unit_ids in the raw test file -- RUL alignment assumption violated."
    assert len(rul_all) == len(orig_test_ids), \
        "RUL file length doesn't match number of test engines -- alignment assumption violated."

    if condition_subset is not None:
        unit_to_condition_test, _, _, _ = get_condition_labels(
            df_test, kmeans=cond_kmeans, scaler=cond_scaler, remap=cond_remap
        )
        df_test = filter_engines_by_condition(df_test, unit_to_condition_test, set(condition_subset))

    df_test[ALL_FEATURE_COLS] = scaler.transform(
        df_test[[c for c in ALL_FEATURE_COLS if c in df_test.columns]]
        .reindex(columns=ALL_FEATURE_COLS, fill_value=0)
    )

    test_unit_ids = sorted(df_test['unit_id'].unique())
    y_test = rul_all[[uid - 1 for uid in test_unit_ids]]

    X_test = []
    for uid in test_unit_ids:
        seq = df_test[df_test['unit_id'] == uid][ALL_FEATURE_COLS].values
        if len(seq) >= SEQUENCE_LENGTH:
            X_test.append(seq[-SEQUENCE_LENGTH:])
        else:
            pad = np.zeros((SEQUENCE_LENGTH - len(seq), len(ALL_FEATURE_COLS)))
            X_test.append(np.vstack([pad, seq]))
    X_test = np.array(X_test)

    test_engine_clusters_by_id = assign_test_engines_to_clusters(
        df_test, kmeans_model, cluster_scaler, sensor_cols
    )
    test_engine_clusters = {
        idx: test_engine_clusters_by_id[uid]
        for idx, uid in enumerate(test_unit_ids)
    }

    return client_data, X_test, y_test, scaler, test_engine_clusters, client_to_cluster

# ── Client ────────────────────────────────────────────────────────────────────
class RULClient(fl.client.NumPyClient):
    def __init__(self, model, base, method, data, head, cid):
        self.model, self.base, self.method, self.head = model, base, method, head
        self.X_tr, self.y_tr, self.X_val, self.y_val = data
        self.cid = cid

    def get_parameters(self, config):
        if self.method == "fedper":
            all_weights = self.model.get_weights()
            head_size = len(self.head.get_weights())
            return all_weights[:-head_size]
        return self.model.get_weights()

    def fit(self, parameters, config):
        if self.method == "fedper":
            head_size = len(self.head.get_weights())
            head_weights = load_head(self.cid)
            if head_weights is None:
                head_weights = self.model.get_weights()[-head_size:]
            self.model.set_weights(list(parameters) + head_weights)
        elif self.method == "fedprox":
            self.model.set_weights(parameters)
        else:
            self.model.set_weights(parameters)

        if self.method == "fedprox":
            self.model.set_weights(parameters)
            global_weights_tf = [tf.constant(v.numpy(), dtype=tf.float32)
                                 for v in self.model.trainable_variables]
            mu = 0.01
            def make_prox_loss(global_w_tf):
                def prox_loss(y_true, y_pred):
                    mse = tf.reduce_mean(tf.square(y_pred - tf.cast(y_true, tf.float32)))
                    reg = tf.add_n([
                        tf.reduce_sum(tf.square(w - gw))
                        for w, gw in zip(self.model.trainable_variables, global_w_tf)
                    ])
                    return mse + (mu / 2.0) * reg
                return prox_loss
            self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                               loss=make_prox_loss(global_weights_tf),
                               metrics=['mae'])
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            self.model.fit(self.X_tr, self.y_tr, epochs=20, batch_size=32,
                           validation_data=(self.X_val, self.y_val),
                           callbacks=[es], verbose=0)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        elif self.method == "ditto":
            self.model.set_weights(parameters)
            global_vars_tf = [tf.constant(v.numpy(), dtype=tf.float32)
                              for v in self.model.trainable_variables]
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            self.model.fit(self.X_tr, self.y_tr, epochs=20, batch_size=32,
                           validation_data=(self.X_val, self.y_val),
                           callbacks=[es], verbose=0)
            global_weights = self.model.get_weights()
            local_model, _, _ = build_model((self.X_tr.shape[1], self.X_tr.shape[2]))
            local_path = f"{HEAD_REGISTRY_DIR}/ditto_local_{self.cid}.pkl"
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    local_model.set_weights(pickle.load(f))
            else:
                local_model.set_weights(global_weights)
            def make_ditto_loss(global_v):
                def ditto_loss(y_true, y_pred):
                    mse = tf.reduce_mean(tf.square(y_pred - tf.cast(y_true, tf.float32)))
                    reg = tf.add_n([
                        tf.reduce_sum(tf.square(w - gv))
                        for w, gv in zip(local_model.trainable_variables, global_v)
                    ])
                    return mse + (0.1 / 2.0) * reg
                return ditto_loss
            local_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                                loss=make_ditto_loss(global_vars_tf),
                                metrics=['mae'])
            local_model.fit(self.X_tr, self.y_tr, epochs=20, batch_size=32,
                            validation_data=(self.X_val, self.y_val),
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)],
                            verbose=0)
            with open(local_path, "wb") as f:
                pickle.dump(local_model.get_weights(), f)
            comm_bytes = int(sum(p.nbytes for p in global_weights))
            return global_weights, len(self.X_tr), {"comm_bytes": comm_bytes}

        else:
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            self.model.fit(self.X_tr, self.y_tr, epochs=20, batch_size=32,
                           validation_data=(self.X_val, self.y_val),
                           callbacks=[es], verbose=0)

        if self.method == "fedper":
            save_head(self.cid, self.head.get_weights())
            all_weights = self.model.get_weights()
            head_size = len(self.head.get_weights())
            weights = all_weights[:-head_size]
        else:
            weights = self.model.get_weights()

        comm_bytes = int(sum(p.nbytes for p in weights))
        return weights, len(self.X_tr), {"comm_bytes": comm_bytes}

    def evaluate(self, parameters, config):
        if self.method == "fedper":
            all_weights = self.model.get_weights()
            head_size = len(self.head.get_weights())
            head_weights = load_head(self.cid)
            if head_weights is not None:
                new_weights = list(parameters) + head_weights
            else:
                new_weights = list(parameters) + all_weights[-head_size:]
            self.model.set_weights(new_weights)
            loss, mae = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        elif self.method == "ditto":
            local_path = f"{HEAD_REGISTRY_DIR}/ditto_local_{self.cid}.pkl"
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    local_weights = pickle.load(f)
                local_model, _, _ = build_model((self.X_val.shape[1], self.X_val.shape[2]))
                local_model.set_weights(local_weights)
                loss, mae = local_model.evaluate(self.X_val, self.y_val, verbose=0)
                del local_model
            else:
                self.model.set_weights(parameters)
                loss, mae = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        else:
            self.model.set_weights(parameters)
            loss, mae = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        return loss, len(self.X_val), {"mae": float(mae)}

# ── Strategy ──────────────────────────────────────────────────────────────────
class ThesisStrategy(fl.server.strategy.FedAvg):
    def __init__(self, eval_fn, method, **kwargs):
        super().__init__(**kwargs)
        self.eval_fn = eval_fn
        self.method = method
        self.final_weights = None
        self.total_comm_bytes = 0

    def aggregate_fit(self, server_round, results, failures):
        agg_params, metrics = super().aggregate_fit(server_round, results, failures)
        if agg_params is not None:
            self.final_weights = parameters_to_ndarrays(agg_params)
            self.total_comm_bytes += sum(r.metrics.get("comm_bytes", 0) for _, r in results)
        return agg_params, metrics

    def evaluate(self, server_round, parameters):
        if self.method in ("fedper", "ditto"):
            return None
        weights = parameters_to_ndarrays(parameters)
        mae, nasa = self.eval_fn(weights)
        print(f"  [Round {server_round}] Test MAE: {mae:.4f} | NASA: {nasa:.2f}")
        return 0.0, {"mae": mae, "nasa_score": nasa}

# ── Simulation ────────────────────────────────────────────────────────────────
def run_simulation(method, dataset_key, seed, condition_subset=None, result_label=None):
    if os.path.exists(HEAD_REGISTRY_DIR):
        shutil.rmtree(HEAD_REGISTRY_DIR)
    os.makedirs(HEAD_REGISTRY_DIR, exist_ok=True)

    set_seed(seed)
    client_data, X_test, y_test, _, test_engine_clusters, client_to_cluster = \
        load_dataset(dataset_key, seed, condition_subset=condition_subset)
    input_shape = (SEQUENCE_LENGTH, len(ALL_FEATURE_COLS))

    temp_model, _, temp_head = build_model(input_shape)
    if method == "fedper":
        head_size = len(temp_head.get_weights())
        initial_params = ndarrays_to_parameters(temp_model.get_weights()[:-head_size])
    else:
        initial_params = ndarrays_to_parameters(temp_model.get_weights())
    tf.keras.backend.clear_session()

    def eval_fn(weights):
        model, base, head = build_model(input_shape)
        model.set_weights(weights)
        preds = model.predict(X_test, verbose=0).flatten()
        mae = float(np.mean(np.abs(preds - y_test)))
        nasa = nasa_rul_score(y_test, preds)
        tf.keras.backend.clear_session()
        return mae, nasa

    def client_fn(cid):
        cid = int(cid)
        model, base, head = build_model(input_shape)
        return RULClient(model, base, method, client_data[cid], head, cid)

    strategy = ThesisStrategy(
        initial_parameters=initial_params,
        eval_fn=eval_fn,
        method=method,
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=lambda m: {
            "mae": sum(n * v["mae"] for n, v in m) / sum(n for n, _ in m)
        },
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.5}
    )

    if method == "fedper":
        if strategy.final_weights is None:
            raise RuntimeError("No weights aggregated — all clients failed. Check FedPer weight mismatch.")
        head_size = len(build_model(input_shape)[2].get_weights())
        client_preds_dict = {}
        for cid in range(NUM_CLIENTS):
            model, base, head = build_model(input_shape)
            h = load_head(cid)
            if h:
                new_weights = list(strategy.final_weights) + h
            else:
                new_weights = list(strategy.final_weights) + \
                              model.get_weights()[-head_size:]
            model.set_weights(new_weights)
            client_preds_dict[cid] = model.predict(
                X_test, verbose=0
            ).flatten()
            tf.keras.backend.clear_session()

        all_preds = route_predictions(
            client_preds_dict, test_engine_clusters,
            client_to_cluster, NUM_CLIENTS, y_test
        )
        final_mae = float(np.mean(np.abs(all_preds - y_test)))
        final_nasa = nasa_rul_score(y_test, all_preds)

    elif method == "ditto":
        client_preds_dict = {}
        for cid in range(NUM_CLIENTS):
            local_path = f"{HEAD_REGISTRY_DIR}/ditto_local_{cid}.pkl"
            model, _, _ = build_model(input_shape)
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    model.set_weights(pickle.load(f))
            else:
                model.set_weights(strategy.final_weights)
            client_preds_dict[cid] = model.predict(
                X_test, verbose=0
            ).flatten()
            tf.keras.backend.clear_session()

        all_preds = route_predictions(
            client_preds_dict, test_engine_clusters,
            client_to_cluster, NUM_CLIENTS, y_test
        )
        final_mae = float(np.mean(np.abs(all_preds - y_test)))
        final_nasa = nasa_rul_score(y_test, all_preds)

    else:
        # FedAvg or FedProx
        final_mae, final_nasa = eval_fn(strategy.final_weights)

    return {
        "method": method,
        "dataset": result_label or dataset_key,
        "seed": seed,
        "test_mae": round(final_mae, 4),
        "nasa_score": round(final_nasa, 2),
        "comm_kb": round(strategy.total_comm_bytes / 1024, 2),
    }

# ── CFL ──────────────────────────────────────────────────────────────────────
def run_cfl(dataset_key, seed):
    if os.path.exists(HEAD_REGISTRY_DIR):
        shutil.rmtree(HEAD_REGISTRY_DIR)
    os.makedirs(HEAD_REGISTRY_DIR, exist_ok=True)

    set_seed(seed)
    paths = DATASETS[dataset_key]
    num_clusters = NUM_CLUSTERS[dataset_key]

    df_full = load_cmapss_data(paths["train"])
    df_full = calculate_rul(df_full, RUL_CAP)
    for col in ALL_FEATURE_COLS:
        if col not in df_full.columns:
            df_full[col] = 0.0

    scaler = MinMaxScaler()
    df_full[ALL_FEATURE_COLS] = scaler.fit_transform(df_full[ALL_FEATURE_COLS])

    unit_to_cluster, kmeans_model, cluster_scaler, sensor_cols = \
        get_engine_clusters(df_full, num_clusters)
    client_partitions, _ = create_hard_partitions(unit_to_cluster, NUM_CLIENTS, num_clusters)

    client_data = []
    for part in client_partitions:
        part = list(part)
        if len(part) < 2:
            train_units, val_units = part, part
        else:
            train_units, val_units = train_test_split(part, test_size=0.2, random_state=seed)
        X_tr, y_tr = create_sequences(df_full[df_full['unit_id'].isin(train_units)], ALL_FEATURE_COLS, SEQUENCE_LENGTH)
        X_val, y_val = create_sequences(df_full[df_full['unit_id'].isin(val_units)], ALL_FEATURE_COLS, SEQUENCE_LENGTH)
        client_data.append((X_tr, y_tr, X_val, y_val))

    df_test = load_cmapss_data(paths["test"])
    df_test[ALL_FEATURE_COLS] = scaler.transform(
        df_test[[c for c in ALL_FEATURE_COLS if c in df_test.columns]]
        .reindex(columns=ALL_FEATURE_COLS, fill_value=0)
    )
    y_test = np.clip(np.loadtxt(paths["rul"]).flatten(), 0, RUL_CAP)
    test_unit_ids = sorted(df_test['unit_id'].unique())
    X_test = []
    for uid in test_unit_ids:
        seq = df_test[df_test['unit_id'] == uid][ALL_FEATURE_COLS].values
        if len(seq) >= SEQUENCE_LENGTH:
            X_test.append(seq[-SEQUENCE_LENGTH:])
        else:
            pad = np.zeros((SEQUENCE_LENGTH - len(seq), len(ALL_FEATURE_COLS)))
            X_test.append(np.vstack([pad, seq]))
    X_test = np.array(X_test)

    test_engine_clusters_by_id = assign_test_engines_to_clusters(
        df_test, kmeans_model, cluster_scaler, sensor_cols
    )
    test_engine_clusters = {
        idx: test_engine_clusters_by_id[uid]
        for idx, uid in enumerate(test_unit_ids)
    }

    input_shape = (SEQUENCE_LENGTH, len(ALL_FEATURE_COLS))

    m, _, _ = build_model(input_shape)
    global_weights = m.get_weights()
    tf.keras.backend.clear_session()

    total_comm_bytes = 0
    client_to_cluster = None

    for round_num in range(1, NUM_ROUNDS + 1):
        client_updates = []
        client_sizes = []
        client_delta_vectors = {}

        for cid in range(NUM_CLIENTS):
            if client_to_cluster is None:
                params = global_weights
            else:
                params = cluster_weights[client_to_cluster[cid]]

            model, base, head = build_model(input_shape)
            model.set_weights(params)
            prev_base = [w.copy() for w in base.get_weights()]

            X_tr, y_tr, X_val, y_val = client_data[cid]
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            model.fit(X_tr, y_tr, epochs=20, batch_size=32,
                      validation_data=(X_val, y_val),
                      callbacks=[es], verbose=0)

            updated_base = base.get_weights()
            delta = np.concatenate([
                (updated_base[i] - prev_base[i]).flatten()
                for i in range(len(updated_base))
            ])
            delta = delta / (np.linalg.norm(delta) + 1e-8)
            client_delta_vectors[cid] = delta

            client_updates.append(model.get_weights())
            client_sizes.append(len(X_tr))
            total_comm_bytes += sum(p.nbytes for p in base.get_weights())
            tf.keras.backend.clear_session()

        if client_to_cluster is None:
            total = sum(client_sizes)
            global_weights = [
                sum(client_sizes[i] * client_updates[i][j] / total
                    for i in range(NUM_CLIENTS))
                for j in range(len(client_updates[0]))
            ]
            if round_num == RECLUSTER_EVERY:
                assignments = compute_cluster_assignments(client_delta_vectors, num_clusters)
                client_to_cluster = {cid: assignments[cid] for cid in range(NUM_CLIENTS)}
                print(f"  [Round {round_num}] CFL one-shot clustering: {client_to_cluster}")
                cluster_weights = {c: [w.copy() for w in global_weights]
                                   for c in range(num_clusters)}
        else:
            cluster_updates = {c: [] for c in range(num_clusters)}
            cluster_sizes = {c: [] for c in range(num_clusters)}
            for cid in range(NUM_CLIENTS):
                c = client_to_cluster[cid]
                cluster_updates[c].append(client_updates[cid])
                cluster_sizes[c].append(client_sizes[cid])
            for c in range(num_clusters):
                if not cluster_updates[c]:
                    continue
                total = sum(cluster_sizes[c])
                cluster_weights[c] = [
                    sum(cluster_sizes[c][i] * cluster_updates[c][i][j] / total
                        for i in range(len(cluster_updates[c])))
                    for j in range(len(cluster_updates[c][0]))
                ]

        if client_to_cluster is None:
            m, _, _ = build_model(input_shape)
            m.set_weights(global_weights)
            preds = m.predict(X_test, verbose=0).flatten()
            mae = float(np.mean(np.abs(preds - y_test)))
            tf.keras.backend.clear_session()
            print(f"  [Round {round_num}] Global MAE: {round(mae, 4)}")
        else:
            round_maes = []
            for c in range(num_clusters):
                m, _, _ = build_model(input_shape)
                m.set_weights(cluster_weights[c])
                preds = m.predict(X_test, verbose=0).flatten()
                round_maes.append(float(np.mean(np.abs(preds - y_test))))
                tf.keras.backend.clear_session()
            print(f"  [Round {round_num}] Cluster MAEs: {[round(x,2) for x in round_maes]} | Avg: {round(np.mean(round_maes),4)}")

    assert client_to_cluster is not None, "FATAL: CFL clustering never triggered."

    final_mae, final_nasa, _ = evaluate_clustered_model(
        cluster_weights=cluster_weights,
        test_engine_clusters=test_engine_clusters,
        X_test=X_test,
        y_test=y_test,
        input_shape=input_shape,
        num_clusters=num_clusters,
        build_model_func=build_model,
        nasa_score_func=nasa_rul_score
    )

    return {
        "method": "cfl",
        "dataset": dataset_key,
        "seed": seed,
        "test_mae": round(final_mae, 4),
        "nasa_score": round(final_nasa, 2),
        "comm_kb": round(total_comm_bytes / 1024, 2),
    }

# ── AC-PFL ──────────────────────────────────────────────────────────────────
def run_acpfl(dataset_key, seed, num_clusters=None, alpha=1.0,
               condition_subset=None, result_label=None):
    """
    AC-PFL v2 — Full version with NASASim
    
    Core contributions:
    1. Private prediction head per client — never shared
    2. Base-only aggregation within clusters
    3. Dynamic re-clustering via spectral clustering on gradient similarity
    4. Risk-aware re-clustering via NASASim (combined with GradSim)
    5. Weighted ensemble evaluation per cluster

    condition_subset: optional set/list of condition-IDs (0..5) to restrict
    this dataset to, e.g. {0,1,2} for a C=3 run. None = unfiltered.
    NOTE: this data-loading block is intentionally kept in lockstep with
    load_dataset() in this file — run_acpfl does NOT call load_dataset,
    it has its own copy, so any filtering change must be made in both places.
    """
    if num_clusters is None:
        num_clusters = NUM_CLUSTERS[dataset_key]

    if os.path.exists(HEAD_REGISTRY_DIR):
        shutil.rmtree(HEAD_REGISTRY_DIR)
    os.makedirs(HEAD_REGISTRY_DIR, exist_ok=True)

    set_seed(seed)
    paths = DATASETS[dataset_key]

    # ── Data loading ──────────────────────────────────────────────────────
    df_full = load_cmapss_data(paths["train"])
    df_full = calculate_rul(df_full, RUL_CAP)
    for col in ALL_FEATURE_COLS:
        if col not in df_full.columns:
            df_full[col] = 0.0

    # -- Condition-regime filter (raw, pre-MinMax setting columns) --
    cond_kmeans = cond_scaler = cond_remap = None
    if condition_subset is not None:
        unit_to_condition, cond_kmeans, cond_scaler, cond_remap = get_condition_labels(df_full)
        df_full = filter_engines_by_condition(df_full, unit_to_condition, set(condition_subset))

    scaler = MinMaxScaler()
    df_full[ALL_FEATURE_COLS] = scaler.fit_transform(df_full[ALL_FEATURE_COLS])

    unit_to_cluster, kmeans_model, cluster_scaler, sensor_cols = \
        get_engine_clusters(df_full, num_clusters)
    client_partitions, client_to_cluster = create_hard_partitions(
        unit_to_cluster, NUM_CLIENTS, num_clusters
    )

    client_data = []
    for part in client_partitions:
        part = list(part)
        if len(part) < 2:
            train_units, val_units = part, part
        else:
            train_units, val_units = train_test_split(
                part, test_size=0.2, random_state=seed
            )
        X_tr, y_tr = create_sequences(
            df_full[df_full['unit_id'].isin(train_units)],
            ALL_FEATURE_COLS, SEQUENCE_LENGTH
        )
        X_val, y_val = create_sequences(
            df_full[df_full['unit_id'].isin(val_units)],
            ALL_FEATURE_COLS, SEQUENCE_LENGTH
        )
        client_data.append((X_tr, y_tr, X_val, y_val))

    # ── Test set. rul_all[i] is RUL for unit_id (i+1) in the ORIGINAL,
    # unfiltered test file -- re-index by original unit_id, not list position. ──
    df_test = load_cmapss_data(paths["test"])
    rul_all = np.clip(np.loadtxt(paths["rul"]).flatten(), 0, RUL_CAP)
    orig_test_ids = sorted(df_test['unit_id'].unique())
    assert orig_test_ids == list(range(1, len(orig_test_ids) + 1)), \
        "Expected 1-indexed contiguous unit_ids in the raw test file -- RUL alignment assumption violated."
    assert len(rul_all) == len(orig_test_ids), \
        "RUL file length doesn't match number of test engines -- alignment assumption violated."

    if condition_subset is not None:
        unit_to_condition_test, _, _, _ = get_condition_labels(
            df_test, kmeans=cond_kmeans, scaler=cond_scaler, remap=cond_remap
        )
        df_test = filter_engines_by_condition(df_test, unit_to_condition_test, set(condition_subset))

    df_test[ALL_FEATURE_COLS] = scaler.transform(
        df_test[[c for c in ALL_FEATURE_COLS if c in df_test.columns]]
        .reindex(columns=ALL_FEATURE_COLS, fill_value=0)
    )
    test_unit_ids = sorted(df_test['unit_id'].unique())
    y_test = rul_all[[uid - 1 for uid in test_unit_ids]]

    X_test = []
    for uid in test_unit_ids:
        seq = df_test[df_test['unit_id'] == uid][ALL_FEATURE_COLS].values
        if len(seq) >= SEQUENCE_LENGTH:
            X_test.append(seq[-SEQUENCE_LENGTH:])
        else:
            pad = np.zeros((SEQUENCE_LENGTH - len(seq), len(ALL_FEATURE_COLS)))
            X_test.append(np.vstack([pad, seq]))
    X_test = np.array(X_test)

    test_engine_clusters_by_id = assign_test_engines_to_clusters(
        df_test, kmeans_model, cluster_scaler, sensor_cols
    )
    test_engine_clusters = {
        idx: test_engine_clusters_by_id[int(uid)]
        for idx, uid in enumerate(test_unit_ids)
    }

    input_shape = (SEQUENCE_LENGTH, len(ALL_FEATURE_COLS))

    # ── Initialize cluster base weights ───────────────────────────────────
    cluster_base_weights = {}
    for c in range(num_clusters):
        _, base_init, _ = build_model(input_shape)
        cluster_base_weights[c] = [w.copy() for w in base_init.get_weights()]
        tf.keras.backend.clear_session()

    # Private state per client
    client_head_weights = {}

    total_comm_bytes = 0

    # ── Federation rounds ─────────────────────────────────────────────────
    for round_num in range(1, NUM_ROUNDS + 1):

        cluster_base_updates = {c: [] for c in range(num_clusters)}
        cluster_sizes = {c: [] for c in range(num_clusters)}
        client_delta_vectors = {}
        client_val_nasa = {}
        client_risk_profiles = {}

        for cid in range(NUM_CLIENTS):
            cluster_id = client_to_cluster[str(cid)]

            # Build model and load weights
            model, base, head = build_model(input_shape)
            base.set_weights(cluster_base_weights[cluster_id])
            if cid in client_head_weights:
                head.set_weights(client_head_weights[cid])

            # Snapshot base before training
            prev_base = [w.copy() for w in base.get_weights()]

            # Local training
            X_tr, y_tr, X_val, y_val = client_data[cid]
            es = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            )
            model.fit(
                X_tr, y_tr, epochs=20, batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[es], verbose=0
            )

            # Extract updated weights
            updated_base = [w.copy() for w in base.get_weights()]
            client_head_weights[cid] = [w.copy() for w in head.get_weights()]

            # Base-only gradient delta for re-clustering
            delta = np.concatenate([
                (updated_base[i] - prev_base[i]).flatten()
                for i in range(len(updated_base))
            ])
            delta = delta / (np.linalg.norm(delta) + 1e-8)
            client_delta_vectors[cid] = delta

            # Validation NASA for monitoring
            val_preds = model.predict(X_val, verbose=0).flatten()
            client_val_nasa[cid] = nasa_rul_score(y_val, val_preds)

            # Risk profile for NASASim
            errors = val_preds - y_val
            client_risk_profiles[cid] = np.array([
                client_val_nasa[cid],
                float(np.mean(errors > 0)),
                float(np.mean(errors)),
                float(np.std(errors))
            ])

            # Accumulate base updates
            cluster_base_updates[cluster_id].append(updated_base)
            cluster_sizes[cluster_id].append(len(X_tr))
            total_comm_bytes += sum(p.nbytes for p in updated_base)

            tf.keras.backend.clear_session()

        # ── Aggregate base weights ────────────────────────────────────────
        for c in range(num_clusters):
            if not cluster_base_updates[c]:
                print(f"  ⚠️ Round {round_num}: Cluster {c} has no clients.")
                continue
            total = sum(cluster_sizes[c])
            cluster_base_weights[c] = [
                sum(
                    cluster_sizes[c][i] * cluster_base_updates[c][i][j] / total
                    for i in range(len(cluster_base_updates[c]))
                )
                for j in range(len(cluster_base_updates[c][0]))
            ]

        # ── Dynamic re-clustering with NASASim ──────────────────────────
        if round_num % RECLUSTER_EVERY == 0 and round_num < NUM_ROUNDS:

            cids = sorted(client_delta_vectors.keys())
            n = len(cids)

            # Gradient similarity matrix
            grad_sim = np.zeros((n, n))
            for i, ci in enumerate(cids):
                for j, cj in enumerate(cids):
                    grad_sim[i, j] = np.dot(
                        client_delta_vectors[ci],
                        client_delta_vectors[cj]
                    )
            grad_sim = (grad_sim + 1) / 2  # normalize to [0,1]

            # Delayed alpha: pure GradSim for first reclustering
            if round_num <= RECLUSTER_EVERY:
                effective_alpha = 1.0
            else:
                effective_alpha = alpha

            if effective_alpha < 1.0 and len(client_risk_profiles) == NUM_CLIENTS:
                # NASASim on normalized risk profiles
                from sklearn.preprocessing import StandardScaler as SS
                risk_array = np.array([client_risk_profiles[c] for c in cids])
                if risk_array.shape[0] > 1:
                    risk_norm = SS().fit_transform(risk_array)
                else:
                    risk_norm = risk_array

                nasa_sim = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        dot = np.dot(risk_norm[i], risk_norm[j])
                        norm_val = (np.linalg.norm(risk_norm[i]) *
                                    np.linalg.norm(risk_norm[j]) + 1e-8)
                        nasa_sim[i, j] = (dot / norm_val + 1) / 2  # normalize to [0,1]

                combined = effective_alpha * grad_sim + (1 - effective_alpha) * nasa_sim
            else:
                combined = grad_sim

            new_assignments = compute_acpfl_cluster_assignments(
                combined, cids, num_clusters
            )

            changes = {
                cid: (client_to_cluster[str(cid)], new_assignments[cid])
                for cid in range(NUM_CLIENTS)
                if client_to_cluster[str(cid)] != new_assignments[cid]
            }
            client_to_cluster = {
                str(cid): new_assignments[cid]
                for cid in range(NUM_CLIENTS)
            }
            print(f"  [Round {round_num}] Re-clustered (α={effective_alpha:.1f}). "
                  f"Changes: {changes}")
            print(f"  [Round {round_num}] Assignments: {client_to_cluster}")

            for c in range(num_clusters):
                clients_c = [
                    cid for cid in range(NUM_CLIENTS)
                    if client_to_cluster[str(cid)] == c
                ]
                if clients_c:
                    avg = np.mean([client_val_nasa[cid] for cid in clients_c])
                    print(f"  [Round {round_num}] Cluster {c} val NASA: {avg:.2f}")

        # ── Round monitoring ──────────────────────────────────────────────
        print(f"  [Round {round_num}] Val NASA per client: "
              f"{[round(client_val_nasa.get(c, 0), 1) for c in range(NUM_CLIENTS)]}")

    # ── Final evaluation with private heads (BATCHED VERSION) ─────────────
    client_test_preds = {}
    for cid in range(NUM_CLIENTS):
        if cid not in client_head_weights:
            continue
        cluster_id = client_to_cluster[str(cid)]
        m, base_f, head_f = build_model(input_shape)
        base_f.set_weights(cluster_base_weights[cluster_id])
        head_f.set_weights(client_head_weights[cid])
        client_test_preds[cid] = m.predict(X_test, verbose=0).flatten()
        tf.keras.backend.clear_session()

    all_preds = route_predictions(
        client_test_preds, test_engine_clusters,
        client_to_cluster, NUM_CLIENTS, y_test
    )
    final_mae = float(np.mean(np.abs(all_preds - y_test)))
    final_nasa = nasa_rul_score(y_test, all_preds)

    print(f"\n  [Final] MAE: {final_mae:.4f} | NASA: {final_nasa:.2f}")

    return {
        "method": "acpfl",
        "dataset": result_label or dataset_key,
        "seed": seed,
        "test_mae": round(final_mae, 4),
        "nasa_score": round(final_nasa, 2),
        "comm_kb": round(total_comm_bytes / 1024, 2),
    }

# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["fedavg", "fedper", "fedprox", "ditto", "cfl", "acpfl"])
    parser.add_argument("--dataset", required=True, choices=["FD001","FD002","FD003","FD004"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results.csv")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    if args.method == "acpfl":
        result = run_acpfl(args.dataset, args.seed, num_clusters=args.k, alpha=args.alpha)
    elif args.method == "cfl":
        result = run_cfl(args.dataset, args.seed)
    else:
        result = run_simulation(args.method, args.dataset, args.seed)
    print(result)

    exists = os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=result.keys())
        if not exists:
            w.writeheader()
        w.writerow(result)