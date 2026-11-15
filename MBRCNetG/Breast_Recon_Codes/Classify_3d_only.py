import os
import numpy as np
import binvox_rw
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import umap
import hdbscan
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from skimage import measure
from scipy.ndimage import label as scipy_label
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings("ignore")

# === Config ===
left_dir = "outputs/recon3d_full/binvoxFiles/left_breast_recon"
right_dir = "outputs/recon3d_full/binvoxFiles/right_breast_recon"
VIEW_ELEV = 30
VIEW_AZIM = 120

# ─────────────────────────────────────────────
# Rendering helpers (unchanged from original)
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
# Feature extraction  ← KEY IMPROVEMENT
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
    """Sphericity = (pi^(1/3) * (6V)^(2/3)) / A"""
    V = np.sum(vox)
    A = compute_surface_area(vox) * vox.size  # unnormalized surface count
    if A == 0:
        return 0.0
    return (np.pi ** (1/3) * (6 * V) ** (2/3)) / A


def compute_moments_of_inertia(vox):
    """Normalized principal moments of inertia from the voxel mass distribution."""
    coords = np.argwhere(vox > 0).astype(np.float32)
    if len(coords) < 4:
        return np.zeros(3)
    center = coords.mean(axis=0)
    coords -= center
    # Inertia tensor
    Ixx = np.sum(coords[:, 1]**2 + coords[:, 2]**2)
    Iyy = np.sum(coords[:, 0]**2 + coords[:, 2]**2)
    Izz = np.sum(coords[:, 0]**2 + coords[:, 1]**2)
    I = np.sort([Ixx, Iyy, Izz])
    I_norm = I / (I[-1] + 1e-8)   # normalize by largest moment
    return I_norm


def compute_projection_features(vox):
    """
    Area ratios of 2-D silhouette projections onto the three anatomical planes.
    Returns 3 values: proj_xy, proj_xz, proj_yz  (each normalized by vox.size^(2/3))
    """
    norm = vox.size ** (2/3) + 1e-8
    proj_xy = np.sum(np.any(vox, axis=2)) / norm
    proj_xz = np.sum(np.any(vox, axis=1)) / norm
    proj_yz = np.sum(np.any(vox, axis=0)) / norm
    return np.array([proj_xy, proj_xz, proj_yz])


def compute_symmetry_all_axes(vox):
    """Symmetry score along all three axes (L2 difference after flip)."""
    feats = []
    for axis in range(3):
        diff = np.mean(np.abs(vox.astype(np.float32) - np.flip(vox, axis=axis).astype(np.float32)))
        feats.append(diff)
    return np.array(feats)


def compute_aspect_ratios(vox):
    """Bounding-box dimensions and their pairwise ratios."""
    x = np.any(vox, axis=(1, 2))
    y = np.any(vox, axis=(0, 2))
    z = np.any(vox, axis=(0, 1))
    dx, dy, dz = np.sum(x) + 1e-8, np.sum(y) + 1e-8, np.sum(z) + 1e-8
    return np.array([dx/dy, dy/dz, dx/dz])


def compute_pca_shape_features(vox, n_components=5):
    """
    Flatten the voxel grid → PCA → keep first n_components principal scores.
    Captures global shape variance not encoded in hand-crafted scalars.
    NOTE: all voxels must share the same resolution; if not, resize first.
    """
    flat = vox.flatten().astype(np.float32)
    # Return raw flattened; PCA across the dataset is applied later
    return flat


def extract_all_features(vox):
    """
    Combine all hand-crafted features into a single vector.
    PCA-based flattened features are handled separately at dataset level.
    """
    volume      = np.sum(vox) / vox.size
    surface     = compute_surface_area(vox)
    bbox_ratio  = compute_bounding_box_ratio(vox)
    sphericity  = compute_sphericity(vox)
    moments     = compute_moments_of_inertia(vox)          # 3 values
    projections = compute_projection_features(vox)         # 3 values
    symmetries  = compute_symmetry_all_axes(vox)           # 3 values
    aspects     = compute_aspect_ratios(vox)               # 3 values

    hand_crafted = np.concatenate([
        [volume, surface, bbox_ratio, sphericity],
        moments, projections, symmetries, aspects
    ])  # 16 features total
    return hand_crafted


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
# HDBSCAN auto-tuning  (unchanged logic, extended range)
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


def ensure_valid_labels(labels, n_samples):
    if labels is None or np.all(labels == -1):
        print("[WARN] Falling back to single cluster.")
        return np.zeros(n_samples, dtype=int)
    if labels.shape[0] != n_samples:
        print("[WARN] Label length mismatch. Falling back.")
        return np.zeros(n_samples, dtype=int)
    return labels


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


# ─────────────────────────────────────────────
# Visualization helpers (unchanged)
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


def print_cluster_feature_stats(X_hand, labels):
    cols = ["volume", "surface", "bbox_ratio", "sphericity",
            "I0", "I1", "I2",
            "proj_xy", "proj_xz", "proj_yz",
            "sym_x", "sym_y", "sym_z",
            "ar_xy", "ar_yz", "ar_xz"]
    df = pd.DataFrame(X_hand, columns=cols)
    df["label"] = labels
    print("\n📊 Cluster Feature Summary (hand-crafted):")
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


def visualize_mean_voxels(vox_map, labels, IDs, save_dir="voxel_means"):
    os.makedirs(save_dir, exist_ok=True)
    for label in sorted(set(labels)):
        indices  = [i for i, l in enumerate(labels) if l == label]
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


def print_representative_samples(X_umap, labels, IDs, output_file="cluster_medoids.csv"):
    rows = []
    print("\n Cluster Representative Samples (medoid):")
    for label in sorted(set(labels)):
        indices = [i for i, l in enumerate(labels) if l == label]
        cluster_X = X_umap[indices]
        dists = pairwise_distances(cluster_X)
        medoid = np.argmin(np.sum(dists, axis=0))
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
# Main
# ─────────────────────────────────────────────

def run_umap_hdbscan(X_combined: np.ndarray, *, umap_seed: int):
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
    raw_labels, best_size, best_score = auto_tune_hdbscan(X_umap)
    raw_labels = ensure_valid_labels(raw_labels, N)
    labels_viz = assign_noise_to_nearest(X_umap, raw_labels)   
    return X_umap, np.asarray(raw_labels), np.asarray(labels_viz), int(best_size), float(best_score), int(umap_dims), int(n_neighbors)

def compute_pairwise_seed_stability(label_runs):
    """
    Compute pairwise ARI and NMI across multiple labelings on the same sample set.
    """
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


def compute_subsample_stability(X_combined: np.ndarray, IDs, *, umap_seed: int, n_runs: int = 30, frac: float = 0.8):
    """
    Subsample without replacement, cluster each subsample, and compare labelings on the overlap.
    """
    rng = np.random.default_rng(int(umap_seed))
    N = int(X_combined.shape[0])
    k = max(2, int(round(frac * N)))

    label_maps = []
    for _ in range(int(n_runs)):
        idx = np.sort(rng.choice(N, size=k, replace=False))
        X_sub = X_combined[idx]
        IDs_sub = [IDs[i] for i in idx]
        _, y_sub_raw, y_sub_viz, _, _, _, _ = run_umap_hdbscan(X_sub, umap_seed=int(umap_seed))
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

    # 1. Load data
    print(" Loading voxel files …")
    left_hand,  left_flat,  left_ids,  left_voxels  = load_binvox_files(left_dir,  'l')
    right_hand, right_flat, right_ids, right_voxels = load_binvox_files(right_dir, '')

    X_hand  = np.concatenate([left_hand,  right_hand],  axis=0)   # (N, 16)
    X_flat  = np.concatenate([left_flat,  right_flat],  axis=0)   # (N, D)
    IDs     = left_ids + right_ids
    vox_map = left_voxels + right_voxels
    N = len(IDs)
    print(f"   {N} samples loaded.")

    # 2. PCA on flattened voxels → keep 95% variance (or at most 20 components)
    print(" Running PCA on raw voxel data …")
    scaler_flat  = StandardScaler()
    X_flat_scaled = scaler_flat.fit_transform(X_flat)
    n_pca = min(N - 1, 20)
    pca   = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X_flat_scaled)
    explained = np.cumsum(pca.explained_variance_ratio_)
    keep = np.searchsorted(explained, 0.95) + 1
    X_pca = X_pca[:, :keep]
    print(f"   PCA: keeping {keep} components ({explained[keep-1]*100:.1f}% variance)")

    # 3. Standardize hand-crafted features and concatenate with PCA components
    scaler_hand  = StandardScaler()
    X_hand_scaled = scaler_hand.fit_transform(X_hand)

    # Weight hand-crafted features slightly higher (x1.5) to balance with PCA dims
    X_combined = np.concatenate([X_hand_scaled * 1.5, X_pca], axis=1)
    print(f"  Combined feature dim: {X_combined.shape[1]}")

    # 3b. Output directory for this setting
    output_dir = os.path.join("outputs", "classification_only")
    os.makedirs(output_dir, exist_ok=True)
    print(f" Saving outputs to: {output_dir}")
    os.chdir(output_dir)

    # 4. Stability across UMAP random seeds
    SEED_LIST = list(range(10))
    print(f"\n Stability across UMAP seeds: {SEED_LIST}")
    label_runs = []
    run_cache = {}
    rows = []
    for s in SEED_LIST:
        X_umap_s, raw_y, y, bs, sc, ud, nn = run_umap_hdbscan(X_combined, umap_seed=int(s))
        label_runs.append(y)
        run_cache[int(s)] = {
            "X_umap": X_umap_s,
            "labels_raw": raw_y,     
            "labels": y,             
            "best_size": int(bs),
            "silhouette": float(sc),
            "umap_dims": int(ud),
            "n_neighbors": int(nn),
        }
        rows.append({
            "seed": int(s),
            "umap_dims": int(ud),
            "n_neighbors": int(nn),
            "min_cluster_size": int(bs),
            "silhouette": float(sc),
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
    best_size = rep["best_size"]
    best_score = rep["silhouette"]
    n_neighbors = rep["n_neighbors"]

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
        "min_cluster_size": int(best_row["min_cluster_size"]),
        "umap_dims": int(best_row["umap_dims"]),
        "n_neighbors": int(best_row["n_neighbors"]),
    }]).to_csv("representative_seed_summary.csv", index=False)

    # 4b. Stability under subsampling (bootstrap-style)
    subs = compute_subsample_stability(X_combined, IDs, umap_seed=PRIMARY_SEED, n_runs=30, frac=0.8)
    print(
        f"[STAB] Subsample stability ARI mean={subs['ari_mean']:.4f} std={subs['ari_std']:.4f}  "
        f"NMI mean={subs['nmi_mean']:.4f} std={subs['nmi_std']:.4f}  pairs={subs['n_pairs']}"
    )
    pd.DataFrame([subs]).to_csv("stability_subsample_summary.csv", index=False)

    # 5. Print and save cluster membership
    n_noise = int(np.sum(labels_raw == -1))
    print(f"[NOISE] {n_noise}/{N} samples labeled as noise by HDBSCAN ({n_noise/N*100:.1f}%)")

    for i in sorted(set(labels_raw)):
        group = [IDs[j] for j in range(N) if labels_raw[j] == i]
        name = "Noise" if i == -1 else f"Cluster {i+1}"
        print(f"{name} ({len(group)} samples): {group}")

    cluster_rows = []
    for idx in range(N):
        cluster_rows.append({
            "StudyID": str(IDs[idx]),
            "Cluster_raw": int(labels_raw[idx]),   
            "Cluster_viz": int(labels[idx]),       
            "SelectedSeed": int(PRIMARY_SEED),
        })
    pd.DataFrame(cluster_rows).to_csv("cluster_results.csv", index=False)

    # 6. 2-D UMAP visualisation
    X_vis = umap.UMAP(n_components=2, random_state=PRIMARY_SEED,
                      n_neighbors=max(2, N // 4)).fit_transform(X_combined)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap='tab10', s=60)
    for i, sid in enumerate(IDs):
        plt.text(X_vis[i, 0] + 0.05, X_vis[i, 1], sid, fontsize=8)
    plt.title(
        f"Breast Clustering  |  Silhouette={best_score:.4f}  |  min_cluster_size={best_size}  |  seed={PRIMARY_SEED}"
    )
    plt.xlabel("UMAP 1");  plt.ylabel("UMAP 2")
    plt.grid(True);  plt.tight_layout()
    plt.savefig("cluster_umap_best.png")
    if os.environ.get("CLS_SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close()

    # 7. Per-cluster voxel renders + mean voxels
    visualize_voxel_by_cluster(vox_map, labels, IDs)
    visualize_mean_voxels(vox_map, labels, IDs)

    # 8. Feature statistics
    print_cluster_feature_stats(X_hand, labels)
    save_key_geometric_feature_stats(X_hand, labels, output_file="cluster_key_geometric_summary.csv")

    # 9. Medoid representatives
    print_representative_samples(X_umap, labels, IDs, output_file="cluster_medoids.csv")

    print("\n[CLS] Finished. Outputs saved.")
