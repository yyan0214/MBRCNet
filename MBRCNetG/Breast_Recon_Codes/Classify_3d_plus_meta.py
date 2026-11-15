import os
import re
import csv
import numpy as np
import binvox_rw
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import umap
import hdbscan
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from skimage import measure
from scipy.stats import f_oneway, chi2_contingency
import warnings
warnings.filterwarnings("ignore")

# === Config ===
left_dir = "outputs/recon3d_full/binvoxFiles/left_breast_recon"
right_dir = "outputs/recon3d_full/binvoxFiles/right_breast_recon"
metadata_file  = "metadata.xlsx"
KMEANS_K_RANGE = range(2, 5)
VIEW_ELEV = 30
VIEW_AZIM = 120

# ─────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────

def apply_standard_3d_view(ax):
    # Match the mesh visual style used elsewhere in this repo (3D_pipeline.plot_mesh_pair):
    # axis hidden and auto scaling based on mesh bounds.
    ax.set_axis_off()


def _marching_cubes_like_pipeline(vox):
    """Mirror 3D_pipeline.save_mesh_comparison/_mc level selection."""
    v = (vox > 0.5).astype(np.float32)
    if np.max(v) <= 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    vmin, vmax = float(v.min()), float(v.max())
    lv = 0.5 if (vmin < 0.5 < vmax) else (vmin + vmax) / 2.0
    verts, faces, _, _ = measure.marching_cubes(v, level=lv)
    return verts.astype(np.float32), faces.astype(np.int64)


def render_voxel_mesh(ax, vox):
    verts, faces = _marching_cubes_like_pipeline(vox)
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        return
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=0.9)
    mesh.set_edgecolor("k")
    ax.add_collection3d(mesh)
    ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])


# ─────────────────────────────────────────────
# Feature extraction  (16 hand-crafted features, same as Classify_3d_only improved)
# ─────────────────────────────────────────────

def compute_surface_area(vox):
    padded = np.pad(vox, 1, mode='constant')
    surface = 0
    for axis in range(3):
        shifted = np.roll(padded, 1, axis=axis)
        surface += np.sum(np.logical_xor(padded, shifted))
    return surface / vox.size


def compute_bounding_box_ratio(vox):
    x = np.any(vox, axis=(1, 2))
    y = np.any(vox, axis=(0, 2))
    z = np.any(vox, axis=(0, 1))
    size = np.sum(x) * np.sum(y) * np.sum(z)
    volume = np.sum(vox)
    return volume / size if size > 0 else 0


def compute_sphericity(vox):
    V = np.sum(vox)
    A = compute_surface_area(vox) * vox.size
    if A == 0:
        return 0.0
    return (np.pi ** (1/3) * (6 * V) ** (2/3)) / A


def compute_moments_of_inertia(vox):
    coords = np.argwhere(vox > 0).astype(np.float32)
    if len(coords) < 4:
        return np.zeros(3)
    center = coords.mean(axis=0)
    coords -= center
    Ixx = np.sum(coords[:, 1]**2 + coords[:, 2]**2)
    Iyy = np.sum(coords[:, 0]**2 + coords[:, 2]**2)
    Izz = np.sum(coords[:, 0]**2 + coords[:, 1]**2)
    I = np.sort([Ixx, Iyy, Izz])
    return I / (I[-1] + 1e-8)


def compute_projection_features(vox):
    norm = vox.size ** (2/3) + 1e-8
    return np.array([
        np.sum(np.any(vox, axis=2)) / norm,
        np.sum(np.any(vox, axis=1)) / norm,
        np.sum(np.any(vox, axis=0)) / norm,
    ])


def compute_symmetry_all_axes(vox):
    return np.array([
        np.mean(np.abs(vox.astype(np.float32) - np.flip(vox, axis=a).astype(np.float32)))
        for a in range(3)
    ])


def compute_aspect_ratios(vox):
    x = np.any(vox, axis=(1, 2))
    y = np.any(vox, axis=(0, 2))
    z = np.any(vox, axis=(0, 1))
    dx, dy, dz = np.sum(x) + 1e-8, np.sum(y) + 1e-8, np.sum(z) + 1e-8
    return np.array([dx/dy, dy/dz, dx/dz])


def extract_all_features(vox):
    """16 hand-crafted morphological features."""
    volume      = np.sum(vox) / vox.size
    surface     = compute_surface_area(vox)
    bbox_ratio  = compute_bounding_box_ratio(vox)
    sphericity  = compute_sphericity(vox)
    moments     = compute_moments_of_inertia(vox)
    projections = compute_projection_features(vox)
    symmetries  = compute_symmetry_all_axes(vox)
    aspects     = compute_aspect_ratios(vox)
    return np.concatenate([
        [volume, surface, bbox_ratio, sphericity],
        moments, projections, symmetries, aspects
    ])


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_binvox_files(dir_path, suffix):
    hand_feats, flat_voxels, ids, voxel_list = [], [], [], []
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".binvox"):
            sid  = fname.split('.')[0] + suffix
            path = os.path.join(dir_path, fname)
            with open(path, 'rb') as f:
                vox = binvox_rw.read_as_3d_array(f).data.astype(np.float32)
            voxel_list.append(vox)
            hand_feats.append(extract_all_features(vox))
            flat_voxels.append(vox.flatten())
            ids.append(sid)
    return np.array(hand_feats), np.array(flat_voxels), ids, voxel_list


# ─────────────────────────────────────────────
# Clustering helpers
# ─────────────────────────────────────────────

def auto_tune_hdbscan(X_umap, min_size_range=range(2, 9)):
    best_score, best_labels, best_size = -1, None, None
    for min_size in tqdm(min_size_range, desc="Tuning min_cluster_size"):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=1)
        labels    = clusterer.fit_predict(X_umap)
        mask      = labels != -1
        if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
            score = silhouette_score(X_umap[mask], labels[mask])
            print(f"  min_cluster_size={min_size} → Silhouette: {score:.4f}")
            if score > best_score:
                best_score, best_labels, best_size = score, labels, min_size
        else:
            print(f"  min_cluster_size={min_size} → not enough valid clusters.")
    print(f"\n Best min_cluster_size: {best_size}  Silhouette: {best_score:.4f}")
    return best_labels, best_size, best_score


def auto_tune_kmeans(X_embed, k_range=KMEANS_K_RANGE):
    n_samples = X_embed.shape[0]
    best_score, best_labels, best_k = -1, None, None
    for k in k_range:
        if n_samples <= k:
            continue
        km     = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_embed)
        if len(np.unique(labels)) < 2:
            continue
        if np.min(np.bincount(labels)) < 2:
            continue
        score = silhouette_score(X_embed, labels)
        print(f"  k={k} → KMeans silhouette: {score:.4f}")
        if score > best_score:
            best_score, best_labels, best_k = score, labels, k
    if best_k is None:
        print("[WARN] KMeans fallback could not find a valid k.")
    else:
        print(f"[INFO] Best KMeans k={best_k} silhouette={best_score:.4f}")
    return best_labels, best_k, best_score


def assign_noise_to_nearest(X_umap, labels):
    labels = np.asarray(labels).copy()
    mask   = labels != -1
    if np.sum(mask) == 0:
        return np.zeros_like(labels, dtype=int)
    unique_clusters = sorted(set(labels[mask]))
    centroids = np.array([X_umap[labels == lbl].mean(axis=0) for lbl in unique_clusters])
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx) > 0:
        assigned, _ = pairwise_distances_argmin_min(X_umap[noise_idx], centroids)
        for i, idx in enumerate(noise_idx):
            labels[idx] = unique_clusters[assigned[i]]
    return labels


def cluster_with_fallback(X_embed, IDs):
    n_samples = X_embed.shape[0]

    h_labels, best_size, best_score = auto_tune_hdbscan(X_embed)
    if h_labels is not None:
        # Keep HDBSCAN's original -1 noise labels for reporting, while using
        # nearest-cluster assignments only in downstream visualizations.
        h_labels_raw = ensure_valid_labels(h_labels, n_samples)
        h_labels_viz = assign_noise_to_nearest(X_embed, h_labels_raw)
        if len(set(h_labels_viz)) >= 2:
            detail = f"min_cluster_size={best_size}, silhouette={best_score:.4f}"
            print(f"[INFO] Using HDBSCAN. {detail}")
            return np.asarray(h_labels_raw), np.asarray(h_labels_viz), "HDBSCAN", detail

    print("[WARN] HDBSCAN did not produce meaningful output. Switching to KMeans.")
    k_labels, best_k, k_score = auto_tune_kmeans(X_embed)
    if k_labels is not None and len(set(k_labels)) >= 2:
        detail = f"k={best_k}, silhouette={k_score:.4f}"
        print(f"[INFO] Using KMeans fallback. {detail}")
        return np.asarray(k_labels), np.asarray(k_labels), "KMEANS", detail

    print("[WARN] Both methods failed. Falling back to single cluster.")
    labels = np.zeros(n_samples, dtype=int)
    return labels, labels.copy(), "SINGLE", "fallback_single_cluster"


def ensure_valid_labels(labels, n_samples):
    if labels is None:
        print("[WARN] No valid clustering found. Falling back to single cluster.")
        return np.zeros(n_samples, dtype=int)
    labels = np.asarray(labels)
    if labels.shape[0] != n_samples:
        print("[WARN] Label length mismatch. Falling back.")
        return np.zeros(n_samples, dtype=int)
    if np.all(labels == -1):
        print("[WARN] All samples marked noise. Falling back.")
        return np.zeros(n_samples, dtype=int)
    return labels


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def visualize_voxel_by_cluster(vox_map, labels, IDs, save_dir="voxel_clusters"):
    os.makedirs(save_dir, exist_ok=True)
    for label in sorted(set(labels)):
        cluster_indices = [i for i, l in enumerate(labels) if l == label]
        fig = plt.figure(figsize=(8, 4 * len(cluster_indices)))
        for idx, i in enumerate(cluster_indices):
            ax = fig.add_subplot(len(cluster_indices), 1, idx + 1, projection='3d')
            render_voxel_mesh(ax, vox_map[i])
            apply_standard_3d_view(ax)
            ax.set_title(f"ID: {IDs[i]}")
            ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cluster_{label+1}.png")
        plt.close()


def visualize_mean_voxels(vox_map, labels, IDs, save_dir="voxel_means"):
    os.makedirs(save_dir, exist_ok=True)
    for label in sorted(set(labels)):
        indices  = [i for i, l in enumerate(labels) if l == label]
        if not indices:
            continue
        mean_vox = np.mean([vox_map[i] for i in indices], axis=0) > 0.5
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        render_voxel_mesh(ax, mean_vox)
        apply_standard_3d_view(ax)
        ax.set_title(f"Cluster {label+1} Mean Voxel")
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mean_cluster_{label+1}.png")
        plt.close()


def print_cluster_feature_stats(X_hand, labels):
    cols = ["volume", "surface", "bbox_ratio", "sphericity",
            "I0", "I1", "I2",
            "proj_xy", "proj_xz", "proj_yz",
            "sym_x", "sym_y", "sym_z",
            "ar_xy", "ar_yz", "ar_xz"]
    df = pd.DataFrame(X_hand, columns=cols)
    df["label"] = labels
    print("\n Cluster Feature Summary (hand-crafted):")
    print(df.groupby("label")[["volume","surface","bbox_ratio","sphericity",
                                "sym_x","sym_y","sym_z"]].agg(['mean','std']).round(4))


def save_key_geometric_feature_stats(X_hand, labels, output_file="cluster_key_geometric_summary.csv"):
    cols = ["volume", "surface", "bbox_ratio", "sphericity",
            "I0", "I1", "I2",
            "proj_xy", "proj_xz", "proj_yz",
            "sym_x", "sym_y", "sym_z",
            "ar_xy", "ar_yz", "ar_xz"]
    df = pd.DataFrame(X_hand, columns=cols)
    df["cluster"] = labels
    key_cols = ["volume", "sphericity", "sym_x", "sym_y", "sym_z"]
    summary = df.groupby("cluster")[key_cols].agg(["mean", "std"]).round(6)
    summary.to_csv(output_file)
    print(f" Key geometric summary saved → {output_file}")


def print_representative_samples(X_umap, labels, IDs, output_file="cluster_medoids.csv"):
    rows = []
    print("\n Cluster Representative Samples (medoid):")
    for label in sorted(set(labels)):
        indices = [i for i, l in enumerate(labels) if l == label]
        if not indices:
            continue
        cluster_X = X_umap[indices]
        dists     = pairwise_distances(cluster_X)
        medoid    = np.argmin(np.sum(dists, axis=0))
        medoid_id = IDs[indices[medoid]]
        print(f"  Cluster {label+1}: {medoid_id}")
        rows.append({
            "cluster": int(label),
            "cluster_display": int(label) + 1,
            "medoid_id": str(medoid_id),
            "cluster_size": int(len(indices)),
        })
    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f" Medoid list saved → {output_file}")


# ─────────────────────────────────────────────
# Statistics export  (unchanged from original)
# ─────────────────────────────────────────────

def safe_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", str(name)).replace(" ", "_")


def get_stat_value(df, row_label, feature, stat_name):
    key_tuple = (feature, stat_name)
    if isinstance(df.columns, pd.MultiIndex) and key_tuple in df.columns:
        return df.at[row_label, key_tuple]
    if key_tuple in df.columns:
        return df.at[row_label, key_tuple]
    key_flat = f"{feature}_{stat_name}"
    if key_flat in df.columns:
        return df.at[row_label, key_flat]
    try:
        return df.loc[row_label, key_tuple]
    except Exception:
        return np.nan


def save_cluster_feature_statistics(X_hand, labels, meta_features,
                                    output_file="cluster_stats_summary.csv"):
    try:
        import seaborn as sns
    except Exception:
        sns = None

    # Use the same 4 headline features for the CSV summary for backward compatibility
    features   = ["volume", "surface", "bbox_ratio", "sphericity"]
    df_struct  = pd.DataFrame(X_hand[:, :4], columns=features)
    df_struct["cluster"] = labels

    meta_df          = meta_features.copy()
    meta_df['cluster'] = labels

    struct_stats = df_struct.groupby("cluster")[features].agg(["mean", "std"]).fillna(0.0)
    global_mean  = df_struct[features].mean()
    global_std   = df_struct[features].std()

    relative_stats = pd.DataFrame(index=struct_stats.index)
    for feature in features:
        gs = float(global_std[feature]) if not np.isnan(global_std[feature]) else 0.0
        if abs(gs) < 1e-12:
            rel_mean = np.zeros(len(struct_stats), dtype=np.float32)
            rel_std  = np.zeros(len(struct_stats), dtype=np.float32)
        else:
            rel_mean = ((struct_stats[(feature, "mean")] - float(global_mean[feature])) / gs).values
            rel_std  = (struct_stats[(feature, "std")] / gs).values
        relative_stats[(feature, "mean")] = rel_mean
        relative_stats[(feature, "std")]  = rel_std

    if not isinstance(relative_stats.columns, pd.MultiIndex):
        if all(isinstance(c, tuple) and len(c) == 2 for c in relative_stats.columns):
            relative_stats.columns = pd.MultiIndex.from_tuples(relative_stats.columns)

    meta_grouped = meta_df.groupby("cluster").mean(numeric_only=True).fillna(0.0)

    sig_results = []
    if meta_df["cluster"].nunique() >= 2:
        for col in meta_df.columns:
            if col == "cluster":
                continue
            vals     = meta_df[col].dropna()
            if vals.empty:
                continue
            is_binary = np.isin(vals.unique(), [0, 1]).all()
            try:
                if is_binary:
                    table = pd.crosstab(meta_df["cluster"], meta_df[col])
                    if table.shape[0] >= 2 and table.shape[1] >= 2:
                        _, p, _, _ = chi2_contingency(table)
                        sig_results.append((col, float(p)))
                else:
                    groups = [g[col].dropna().values for _, g in meta_df.groupby("cluster")]
                    groups = [g for g in groups if len(g) > 1]
                    if len(groups) >= 2:
                        _, p = f_oneway(*groups)
                        sig_results.append((col, float(p)))
            except Exception:
                continue
    sig_results.sort(key=lambda x: x[1])
    sig_results = sig_results[:10]

    os.makedirs("meta_feature_plots", exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["Section 1: Absolute Cluster Statistics (Mean ± Std)"])
        writer.writerow(["Feature", "Cluster", "Mean", "Std"])
        for feature in features:
            for cluster in struct_stats.index:
                writer.writerow([
                    feature, cluster,
                    get_stat_value(struct_stats, cluster, feature, "mean"),
                    get_stat_value(struct_stats, cluster, feature, "std"),
                ])

        writer.writerow([""])
        writer.writerow(["Section 2: Relative Cluster Statistics (to Global Mean/Std)"])
        writer.writerow(["Feature", "Cluster", "Relative Mean", "Relative Std"])
        for feature in features:
            for cluster in relative_stats.index:
                writer.writerow([
                    feature, cluster,
                    get_stat_value(relative_stats, cluster, feature, "mean"),
                    get_stat_value(relative_stats, cluster, feature, "std"),
                ])

        writer.writerow([""])
        writer.writerow(["Section 3: Metadata Feature Means per Cluster"])
        writer.writerow(["Metadata Feature"] + [f"Cluster {i}" for i in meta_grouped.index])
        for col in meta_grouped.columns:
            writer.writerow([col] + [meta_grouped.loc[i, col] for i in meta_grouped.index])

        writer.writerow([""])
        writer.writerow(["Section 4: Metadata Features Ranked by p-value (Top 10)"])
        writer.writerow(["Feature", "p-value", "Significant(p<0.05)"])
        if not sig_results:
            writer.writerow(["No valid statistical test result", "", ""])
        else:
            for feat, p in sig_results:
                writer.writerow([feat, p, "YES" if p < 0.05 else "NO"])

    # Per-feature plots
    for col in meta_df.columns:
        if col == 'cluster':
            continue
        values    = meta_df[col]
        is_binary = np.isin(values.dropna().unique(), [0, 1]).all()
        try:
            plt.figure(figsize=(6, 4))
            if is_binary:
                counts = meta_df.groupby("cluster")[col].sum()
                title  = f"Feature: {col} | " + " | ".join([f"C{cl}: {int(counts[cl])}" for cl in counts.index])
                plt.bar(counts.index, counts.values, tick_label=[f"C{cl}" for cl in counts.index])
                plt.ylabel("Count (1s)")
            else:
                counts = meta_df.groupby("cluster")[col].count()
                title  = f"Feature: {col} | " + " | ".join([f"C{cl}: {int(counts[cl])}" for cl in counts.index])
                if sns is not None:
                    sns.boxplot(data=meta_df, x="cluster", y=col)
                else:
                    groups = [meta_df[meta_df["cluster"] == cl][col].dropna().values
                              for cl in sorted(meta_df["cluster"].unique())]
                    groups = [g for g in groups if len(g) > 0]
                    if groups:
                        plt.boxplot(groups)
                        plt.xticks(
                            ticks=np.arange(1, len(groups) + 1),
                            labels=[f"C{cl}" for cl in sorted(meta_df["cluster"].unique())[:len(groups)]],
                        )
                plt.ylabel("Value")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(f"meta_feature_plots/{safe_filename(col)}_plot.png")
            plt.close()
        except Exception as e:
            print(f"  Skip plot for {col}: {e}")

    print(f"\n Statistics saved → {output_file}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_umap_hdbscan(X_combined: np.ndarray, IDs, *, umap_seed: int):
    """
    Run UMAP embedding followed by clustering (HDBSCAN with KMeans fallback).
    Returns:
        X_umap: (N, d) embedding
        labels_raw: (N,) int (HDBSCAN labels, with -1 noise retained)
        labels_viz: (N,) int (noise assigned to nearest cluster for visualization)
        method_name: str
        method_detail: str
        umap_dims: int
        n_neighbors: int
    """
    N = int(X_combined.shape[0])
    umap_dims = min(N - 2, 8)
    n_neighbors = max(2, N // 4)
    reducer = umap.UMAP(
        n_components=umap_dims,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=int(umap_seed),
    )
    X_umap = reducer.fit_transform(X_combined)
    labels_raw, labels_viz, method_name, method_detail = cluster_with_fallback(X_umap, IDs)
    labels_raw = ensure_valid_labels(labels_raw, N)
    labels_viz = ensure_valid_labels(labels_viz, N)
    return X_umap, np.asarray(labels_raw), np.asarray(labels_viz), str(method_name), str(method_detail), int(umap_dims), int(n_neighbors)


def compute_pairwise_seed_stability(label_runs):
    import itertools
    pairs = list(itertools.combinations(range(len(label_runs)), 2))
    if not pairs:
        return {"ari_mean": 1.0, "ari_std": 0.0, "nmi_mean": 1.0, "nmi_std": 0.0, "n_pairs": 0}
    aris, nmis = [], []
    for i, j in pairs:
        y1 = np.asarray(label_runs[i])
        y2 = np.asarray(label_runs[j])
        aris.append(adjusted_rand_score(y1, y2))
        nmis.append(normalized_mutual_info_score(y1, y2))
    aris = np.asarray(aris, dtype=float)
    nmis = np.asarray(nmis, dtype=float)
    return {
        "ari_mean": float(aris.mean()),
        "ari_std": float(aris.std(ddof=0)),
        "nmi_mean": float(nmis.mean()),
        "nmi_std": float(nmis.std(ddof=0)),
        "n_pairs": int(len(pairs)),
    }


def compute_seed_consistency_table(label_runs, seed_list):
    """
    For each seed, compute mean ARI/NMI to all other seeds.
    """
    rows = []
    for i, seed in enumerate(seed_list):
        aris_i, nmis_i = [], []
        y_i = np.asarray(label_runs[i])
        for j in range(len(label_runs)):
            if i == j:
                continue
            y_j = np.asarray(label_runs[j])
            aris_i.append(adjusted_rand_score(y_i, y_j))
            nmis_i.append(normalized_mutual_info_score(y_i, y_j))
        rows.append({
            "seed": int(seed),
            "mean_ari_to_others": float(np.mean(aris_i)) if aris_i else 1.0,
            "mean_nmi_to_others": float(np.mean(nmis_i)) if nmis_i else 1.0,
        })
    return rows


def compute_silhouette_safe(X_embed, labels):
    labels = np.asarray(labels)
    valid_mask = labels >= 0
    if np.sum(valid_mask) < 3:
        return float("nan")
    unique_labels = np.unique(labels[valid_mask])
    if unique_labels.size < 2:
        return float("nan")
    return float(silhouette_score(X_embed[valid_mask], labels[valid_mask]))


def compute_subsample_stability(X_combined: np.ndarray, IDs, *, umap_seed: int, n_runs: int = 30, frac: float = 0.8):
    rng = np.random.default_rng(int(umap_seed))
    N = int(X_combined.shape[0])
    k = max(2, int(round(frac * N)))

    label_maps = []
    for _ in range(int(n_runs)):
        idx = np.sort(rng.choice(N, size=k, replace=False))
        X_sub = X_combined[idx]
        IDs_sub = [IDs[i] for i in idx]
        _, y_sub_raw, _, _, _, _, _ = run_umap_hdbscan(X_sub, IDs_sub, umap_seed=int(umap_seed))
        label_maps.append({str(IDs_sub[i]): int(y_sub_raw[i]) for i in range(len(IDs_sub))})

    aris, nmis = [], []
    for i in range(len(label_maps)):
        for j in range(i + 1, len(label_maps)):
            keys = sorted(set(label_maps[i].keys()) & set(label_maps[j].keys()))
            if len(keys) < 3:
                continue
            y1 = [label_maps[i][k] for k in keys]
            y2 = [label_maps[j][k] for k in keys]
            aris.append(adjusted_rand_score(y1, y2))
            nmis.append(normalized_mutual_info_score(y1, y2))

    if not aris:
        return {"ari_mean": float("nan"), "ari_std": float("nan"), "nmi_mean": float("nan"), "nmi_std": float("nan"), "n_pairs": 0}
    aris = np.asarray(aris, dtype=float)
    nmis = np.asarray(nmis, dtype=float)
    return {
        "ari_mean": float(aris.mean()),
        "ari_std": float(aris.std(ddof=0)),
        "nmi_mean": float(nmis.mean()),
        "nmi_std": float(nmis.std(ddof=0)),
        "n_pairs": int(len(aris)),
    }


if __name__ == "__main__":

    # 1. Load voxel data
    print(" Loading voxel files …")
    left_hand,  left_flat,  left_ids,  left_voxels  = load_binvox_files(left_dir,  'l')
    right_hand, right_flat, right_ids, right_voxels = load_binvox_files(right_dir, '')

    X_hand  = np.concatenate([left_hand,  right_hand],  axis=0)   # (N, 16)
    X_flat  = np.concatenate([left_flat,  right_flat],  axis=0)   # (N, D)
    IDs     = left_ids + right_ids
    vox_map = left_voxels + right_voxels
    N = len(IDs)
    print(f"   {N} samples loaded.")

    # 2. PCA on raw voxels → keep 95% variance (cap at 20 components)
    print(" Running PCA on raw voxel data …")
    scaler_flat   = StandardScaler()
    X_flat_scaled = scaler_flat.fit_transform(X_flat)
    n_pca  = min(N - 1, 20)
    pca    = PCA(n_components=n_pca, random_state=42)
    X_pca  = pca.fit_transform(X_flat_scaled)
    explained = np.cumsum(pca.explained_variance_ratio_)
    keep   = np.searchsorted(explained, 0.95) + 1
    X_pca  = X_pca[:, :keep]
    print(f"   PCA: keeping {keep} components ({explained[keep-1]*100:.1f}% variance)")

    # 3. Standardize hand-crafted features and combine with PCA
    scaler_hand   = StandardScaler()
    X_hand_scaled = scaler_hand.fit_transform(X_hand)
    X_3d_combined = np.concatenate([X_hand_scaled * 1.5, X_pca], axis=1)
    print(f"   3D feature dim: {X_3d_combined.shape[1]}")

    # 4. Load and encode metadata
    print(" Loading metadata …")
    metadata = pd.read_excel(metadata_file)
    # Drop identifier and admin columns, and free-text fields that are not suitable for one-hot encoding.
    metadata = metadata.drop(
        columns=[
            "Date subject signed consent",
            "Complete?",
            "Comments",
            "What medications are you taking?",
        ],
        errors="ignore",
    )
    metadata["StudyID"] = metadata["Study ID"].astype(str)
    metadata = metadata.drop(columns=["Study ID"])
    meta_dict = metadata.set_index("StudyID").to_dict("index")

    ids_no_suffix = [sid.rstrip('l') for sid in IDs]
    meta_rows, meta_dim = [], 0
    for sid in ids_no_suffix:
        if sid in meta_dict:
            feats    = list(meta_dict[sid].values())
            meta_dim = len(feats)
        else:
            feats = [0] * meta_dim if meta_dim > 0 else []
        meta_rows.append(feats)

    meta_features = pd.DataFrame(meta_rows).fillna(0)
    meta_features = pd.get_dummies(meta_features)   # one-hot encode categoricals

    scaler_meta   = StandardScaler()
    X_meta_scaled = scaler_meta.fit_transform(meta_features.values.astype(float))
    print(f"   Metadata feature dim: {X_meta_scaled.shape[1]}")

    # 5. Fuse 3D + metadata features — each block contributes exactly 50%
    def _norm_block(X):
        fnorm = np.linalg.norm(X, 'fro')
        return X / (fnorm + 1e-12)

    X_3d_normed   = _norm_block(X_3d_combined)
    X_meta_normed = _norm_block(X_meta_scaled)
    X_combined    = np.concatenate([X_3d_normed, X_meta_normed], axis=1)
    print(f"   Combined feature dim: {X_combined.shape[1]}"
        f"  (3D: {X_3d_normed.shape[1]} dims | meta: {X_meta_normed.shape[1]} dims, 50/50 energy split)")

    # 5b. Output directory for this setting
    output_dir = os.path.join("outputs", "classification_plus_meta")
    os.makedirs(output_dir, exist_ok=True)
    print(f" Saving outputs to: {output_dir}")
    os.chdir(output_dir)

    # 6. Stability across UMAP random seeds
    SEED_LIST = list(range(10))
    print(f"\n Stability across UMAP seeds: {SEED_LIST}")
    label_runs = []
    run_cache = {}
    rows = []
    for s in SEED_LIST:
        X_umap_s, raw_y, y, mname, mdetail, ud, nn = run_umap_hdbscan(X_combined, IDs, umap_seed=int(s))
        sc = compute_silhouette_safe(X_umap_s, y)
        label_runs.append(y)
        run_cache[int(s)] = {
            "X_umap": X_umap_s,
            "labels_raw": raw_y,
            "labels": y,
            "method_name": str(mname),
            "method_detail": str(mdetail),
            "umap_dims": int(ud),
            "n_neighbors": int(nn),
            "silhouette": float(sc) if not np.isnan(sc) else float("nan"),
        }
        rows.append({
            "seed": int(s),
            "umap_dims": int(ud),
            "n_neighbors": int(nn),
            "method": str(mname),
            "method_detail": str(mdetail),
            "silhouette": float(sc) if not np.isnan(sc) else np.nan,
        })
    seed_stab = compute_pairwise_seed_stability(label_runs)
    seed_consistency = pd.DataFrame(compute_seed_consistency_table(label_runs, SEED_LIST))
    seed_rows = pd.DataFrame(rows).merge(seed_consistency, on="seed", how="left")
    best_row = seed_rows.sort_values(
        by=["mean_nmi_to_others", "mean_ari_to_others", "silhouette", "seed"],
        ascending=[False, False, False, True],
    ).iloc[0]
    PRIMARY_SEED = int(best_row["seed"])
    print(
        f"[REP] Representative seed selected by max mean NMI: {PRIMARY_SEED}  "
        f"(mean_NMI={best_row['mean_nmi_to_others']:.4f}, "
        f"mean_ARI={best_row['mean_ari_to_others']:.4f}, "
        f"silhouette={best_row['silhouette']:.4f})"
    )

    rep = run_cache[PRIMARY_SEED]
    X_umap = rep["X_umap"]
    labels_raw = rep["labels_raw"]
    labels = rep["labels"]
    method_name = rep["method_name"]
    method_detail = rep["method_detail"]
    n_neighbors = rep["n_neighbors"]
    print(f"[INFO] Final method: {method_name} ({method_detail})")

    print(
        f"[STAB] Seed stability ARI mean={seed_stab['ari_mean']:.4f} std={seed_stab['ari_std']:.4f}  "
        f"NMI mean={seed_stab['nmi_mean']:.4f} std={seed_stab['nmi_std']:.4f}  pairs={seed_stab['n_pairs']}"
    )
    seed_rows.to_csv("stability_seeds_summary.csv", index=False)
    pd.DataFrame([seed_stab]).to_csv("stability_seed_metrics.csv", index=False)
    pd.DataFrame([{
        "selected_seed": int(PRIMARY_SEED),
        "selection_rule": "max_mean_nmi_to_others",
        "mean_nmi_to_others": float(best_row["mean_nmi_to_others"]),
        "mean_ari_to_others": float(best_row["mean_ari_to_others"]),
        "silhouette": float(best_row["silhouette"]),
        "method": str(best_row["method"]),
        "method_detail": str(best_row["method_detail"]),
        "umap_dims": int(best_row["umap_dims"]),
        "n_neighbors": int(best_row["n_neighbors"]),
    }]).to_csv("representative_seed_summary.csv", index=False)

    # 6b. Stability under subsampling (bootstrap-style)
    subs = compute_subsample_stability(X_combined, IDs, umap_seed=PRIMARY_SEED, n_runs=30, frac=0.8)
    print(
        f"[STAB] Subsample stability ARI mean={subs['ari_mean']:.4f} std={subs['ari_std']:.4f}  "
        f"NMI mean={subs['nmi_mean']:.4f} std={subs['nmi_std']:.4f}  pairs={subs['n_pairs']}"
    )
    pd.DataFrame([subs]).to_csv("stability_subsample_summary.csv", index=False)

    # 8. Print membership and HDBSCAN noise samples from the unmodified labels.
    if method_name == "HDBSCAN":
        n_noise = int(np.sum(labels_raw == -1))
        noise_ids = [IDs[j] for j in range(N) if labels_raw[j] == -1]
        print(f"[NOISE] {n_noise}/{N} samples labeled as noise by HDBSCAN ({n_noise/N*100:.1f}%): {noise_ids}")
    for i in sorted(set(labels)):
        group = [IDs[j] for j in range(N) if labels[j] == i]
        print(f"Cluster {i+1} ({len(group)} samples): {group}")

    # 9. Save cluster assignments
    df_result = pd.DataFrame({
        "StudyID":          IDs,
        "Cluster_raw":      labels_raw,
        "Cluster_viz":      labels,
        "Is_HDBSCAN_Noise": (labels_raw == -1) if method_name == "HDBSCAN" else np.zeros(N, dtype=bool),
        "Method":           method_name,
        "MethodDetail": method_detail,
        "SelectedSeed": int(PRIMARY_SEED),
    })
    df_result.to_excel("cluster_results.xlsx", index=False)

    # 10. 2-D UMAP visualisation
    X_vis = umap.UMAP(n_components=2, random_state=PRIMARY_SEED,
                      n_neighbors=n_neighbors).fit_transform(X_combined)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap='tab10', s=60)
    for i, sid in enumerate(IDs):
        plt.text(X_vis[i, 0] + 0.05, X_vis[i, 1], sid, fontsize=8)
    plt.title(f"3D + Metadata Clustering  |  {method_name}  |  {method_detail}")
    plt.xlabel("UMAP 1");  plt.ylabel("UMAP 2")
    plt.grid(True);  plt.tight_layout()
    plt.savefig("cluster_umap_best.png")
    if os.environ.get("CLS_SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close()

    # 11. Voxel renders + mean voxels
    visualize_voxel_by_cluster(vox_map, labels, IDs)
    visualize_mean_voxels(vox_map, labels, IDs)

    # 12. Feature stats
    print_cluster_feature_stats(X_hand, labels)

    # 13. Full statistics export
    save_cluster_feature_statistics(X_hand, labels, meta_features,
                                    output_file="cluster_stats_summary.csv")
    save_key_geometric_feature_stats(X_hand, labels, output_file="cluster_key_geometric_summary.csv")
    print_representative_samples(X_umap, labels, IDs, output_file="cluster_medoids.csv")

    print("\n[CLS] Finished. Outputs saved.")
