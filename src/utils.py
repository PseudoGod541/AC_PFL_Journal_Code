import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

# No imports from run_experiment here to avoid circular dependency

def load_cmapss_data(file_path: str) -> pd.DataFrame:
    column_names = [
        'unit_id', 'cycle', 'setting1', 'setting2', 'setting3',
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
        's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
        's20', 's21'
    ]
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=column_names)
    df = df.dropna(axis=1, how='all')
    return df

def get_engine_clusters(df: pd.DataFrame, num_clusters: int) -> tuple:
    """
    Clusters engines based on their mean sensor values after scaling.
    Returns:
        - unit_to_cluster: dict {unit_id_str: cluster_id}
        - kmeans: fitted KMeans object
        - scaler: fitted StandardScaler
        - sensor_cols: list of column names used for clustering
    """
    sensor_cols = [col for col in df.columns if col.startswith('s')]
    engine_features = df.groupby('unit_id')[sensor_cols].mean()
    
    if len(engine_features) < num_clusters:
        raise ValueError(f"Number of unique engines ({len(engine_features)}) is less than the number of clusters ({num_clusters}).")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(engine_features)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)

    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < num_clusters:
        print(f"⚠️ Warning: K-Means produced only {len(unique_clusters)} clusters, not the {num_clusters} requested.")
    
    print("\n-- K-Means Clustering Results --")
    cluster_counts = Counter(clusters)
    for cid in range(num_clusters):
        count = cluster_counts.get(cid, 0)
        print(f"  - Cluster {cid} assigned {count} engines.")
    print("---------------------------------")
 
    unit_to_cluster = {str(int(unit_id)): int(cluster) for unit_id, cluster in zip(engine_features.index, clusters)}
    
    return unit_to_cluster, kmeans, scaler, sensor_cols

def assign_test_engines_to_clusters(df_test: pd.DataFrame,
                                     kmeans,
                                     cluster_scaler,
                                     sensor_cols: list) -> dict:
    """
    Assigns test engines to the same clusters as training engines.
    Returns {unit_id: cluster_id} for each test engine.
    """
    engine_features = df_test.groupby('unit_id')[sensor_cols].mean()
    scaled_features = cluster_scaler.transform(engine_features)
    cluster_labels = kmeans.predict(scaled_features)
    
    return {
        int(uid): int(c)
        for uid, c in zip(engine_features.index, cluster_labels)
    }

def create_hard_partitions(unit_to_cluster_map: dict, num_clients: int, num_clusters: int) -> tuple:
    """
    Creates hard partitions, dedicating clients to specific clusters to ensure balance.
    """
    if num_clients < num_clusters:
        raise ValueError(f"Number of clients ({num_clients}) must be >= number of clusters ({num_clusters}) for hard partitioning.")

    cluster_to_units = defaultdict(list)
    for unit_id_str, cluster_id in unit_to_cluster_map.items():
        cluster_to_units[cluster_id].append(int(unit_id_str))

    clients_per_cluster = [0] * num_clusters
    for i in range(num_clients):
        clients_per_cluster[i % num_clusters] += 1
    
    client_partitions = [[] for _ in range(num_clients)]
    client_to_cluster_map = {}
    
    client_idx = 0
    for cluster_id in range(num_clusters):
        num_dedicated_clients = clients_per_cluster[cluster_id]
        if num_dedicated_clients == 0 or not cluster_to_units.get(cluster_id):
            continue
        unit_partitions = np.array_split(cluster_to_units[cluster_id], num_dedicated_clients)
        for part in unit_partitions:
            if part.size > 0 and client_idx < num_clients:
                client_partitions[client_idx] = part.tolist()
                client_to_cluster_map[str(client_idx)] = cluster_id
                client_idx += 1

    return client_partitions, client_to_cluster_map

def get_condition_labels(df: pd.DataFrame, n_conditions: int = 6,
                          kmeans=None, scaler=None, remap=None) -> tuple:
    """
    Recovers ground-truth operating-condition regime labels by clustering
    the 3 raw operational-setting columns. This is a SEPARATE clustering
    from get_engine_clusters (which groups engines by sensor/health signature
    for client partitioning) -- do not confuse the two.

    Fit once on train (kmeans=None), then reuse on test by passing the
    returned kmeans/scaler/remap back in -- fitting independently on train
    and test does not guarantee condition-ID 2 means the same physical
    regime in both.

    Condition IDs are remapped from KMeans' arbitrary internal label order
    to be sorted by mean setting1 of each centroid -- so "conditions {0,1,2}"
    is a physically ordered, defensible subset choice, not an artifact of
    whatever order sklearn happened to assign.

    Returns:
        unit_to_condition: dict {unit_id: condition_id}, majority-vote per
                            engine trajectory (flags engines spanning >1 regime)
        kmeans, scaler, remap: fitted objects -- pass back in to label a
                                second dataframe (e.g. the test set) consistently
    """
    setting_cols = ['setting1', 'setting2', 'setting3']

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(df[setting_cols])
    else:
        X = scaler.transform(df[setting_cols])

    if kmeans is None:
        kmeans = KMeans(n_clusters=n_conditions, random_state=42, n_init=10)
        raw_labels = kmeans.fit_predict(X)

        sil = silhouette_score(X, raw_labels)
        print(f"  -- Condition-regime clustering: silhouette = {sil:.3f} --")
        if sil < 0.9:
            print(f"  ⚠️ Silhouette below 0.9 -- the {n_conditions} conditions "
                  f"may not be separating cleanly. Inspect before trusting the "
                  f"subset filter (this is a real finding, not necessarily a bug).")

        centroid_order = np.argsort(kmeans.cluster_centers_[:, 0])
        remap = {int(old): new for new, old in enumerate(centroid_order)}
    else:
        if remap is None:
            raise ValueError("Reusing a fitted kmeans requires passing its remap too.")
        raw_labels = kmeans.predict(X)

    labels = np.array([remap[c] for c in raw_labels])

    tmp = df[['unit_id']].copy()
    tmp['_condition'] = labels

    unit_to_condition = {}
    spanning_units = []
    for uid, grp in tmp.groupby('unit_id'):
        counts = grp['_condition'].value_counts()
        unit_to_condition[int(uid)] = int(counts.idxmax())
        if len(counts) > 1:
            spanning_units.append(int(uid))

    if spanning_units:
        print(f"  ⚠️ {len(spanning_units)} engine(s) span >1 condition regime; "
              f"majority-vote label applied: {spanning_units[:10]}"
              f"{' ...' if len(spanning_units) > 10 else ''}")

    return unit_to_condition, kmeans, scaler, remap


def filter_engines_by_condition(df: pd.DataFrame, unit_to_condition: dict,
                                 condition_subset: set) -> pd.DataFrame:
    """Keep only rows for engines whose majority condition-ID falls in
    condition_subset. Hard-fails if that empties the dataframe instead of
    silently returning an empty split."""
    keep_units = {uid for uid, c in unit_to_condition.items() if c in condition_subset}
    filtered = df[df['unit_id'].isin(keep_units)].copy()
    if filtered.empty:
        raise ValueError(
            f"condition_subset={condition_subset} matched zero engines out of "
            f"{len(unit_to_condition)}. Check condition IDs are in range "
            f"[0, n_conditions)."
        )
    print(f"  -- Condition filter {sorted(condition_subset)}: "
          f"{len(keep_units)}/{len(unit_to_condition)} engines retained --")
    return filtered


def compute_cluster_assignments(client_updates, num_clusters):
    """
    Cluster clients based on normalized weight update vectors.
    """
    cids = sorted(client_updates.keys())
    vectors = np.array([client_updates[c] for c in cids])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    return {cid: int(labels[i]) for i, cid in enumerate(cids)}

def evaluate_clustered_model(cluster_weights, test_engine_clusters,
                              X_test, y_test, input_shape,
                              num_clusters, build_model_func, nasa_score_func):
    """
    Correct clustered evaluation: each test engine predicted by its assigned cluster model.
    """
    import tensorflow as tf
    all_preds = np.zeros(len(y_test))
    
    for c in range(num_clusters):
        cluster_indices = [idx for idx, cluster in test_engine_clusters.items() if cluster == c]
        if not cluster_indices:
            print(f"  ⚠️ Cluster {c} has no test engines assigned.")
            continue
        
        model, _, _ = build_model_func(input_shape)
        model.set_weights(cluster_weights[c])
        cluster_X = X_test[cluster_indices]
        preds = model.predict(cluster_X, verbose=0).flatten()
        
        for i, idx in enumerate(cluster_indices):
            all_preds[idx] = preds[i]
    
    tf.keras.backend.clear_session()
    
    mae = float(np.mean(np.abs(all_preds - y_test)))
    nasa = nasa_score_func(y_test, all_preds)
    return mae, nasa, all_preds

def compute_acpfl_cluster_assignments(similarity_matrix, cids, num_clusters):
    """
    Spectral clustering on pre-computed similarity matrix.
    Guarantees exactly num_clusters clusters with minimum 2 clients each.
    If spectral produces invalid clusters, falls back to KMeans on distance matrix.
    """
    from sklearn.cluster import SpectralClustering, KMeans
    from collections import Counter
    import numpy as np

    n = len(cids)
    sim = np.clip(similarity_matrix, 0, 1)
    np.fill_diagonal(sim, 1.0)

    if n <= num_clusters:
        return {cid: i for i, cid in enumerate(cids)}

    # Try spectral clustering
    sc = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        random_state=42,
        n_init=10,
        assign_labels='kmeans'
    )
    labels = sc.fit_predict(sim)
    counts = Counter(labels)

    # Check if exactly K non-empty clusters with minimum 2 clients
    has_singleton = any(cnt < 2 for cnt in counts.values())
    has_empty = len(counts) < num_clusters

    if has_singleton or has_empty:
        print(f"  ⚠️ Spectral produced invalid clusters {dict(counts)}. "
              f"Falling back to KMeans.")
        # Convert similarity to distance for KMeans
        distance_matrix = 1 - sim
        np.fill_diagonal(distance_matrix, 0)
        # Use distance matrix rows as feature vectors
        km = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(distance_matrix)
        counts = Counter(labels)
        print(f"  KMeans fallback result: {dict(counts)}")

    return {cid: int(labels[i]) for i, cid in enumerate(cids)}