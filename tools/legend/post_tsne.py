# post_tsne.py (별도 스크립트)
import os, glob, torch
import numpy as np
import matplotlib.pyplot as plt

def load_all(snap_dir):
    feats, pids, steps = [], [], []
    for f in sorted(glob.glob(os.path.join(snap_dir, "step_*.pt"))):
        d = torch.load(f, map_location="cpu")
        feats.append(d["features_fp16"].float())      # [N, D]
        pids.append(d["particle_ids"].long().numpy()) # [N]
        steps.append(np.full_like(pids[-1], int(d["step"])))
    X = torch.cat(feats, dim=0)        # [T*N, D]
    P = np.concatenate(pids)           # [T*N]
    S = np.concatenate(steps)          # [T*N]
    return X, P, S

def run_gpu_tsne_and_plot(snap_dir, out_png):
    X, P, S = load_all(snap_dir)           # X: torch [T*N, D]
    try:
        from cuml.manifold import TSNE     # GPU t-SNE
        import cupy as cp
        X_gpu = cp.asarray(X.numpy(), dtype=cp.float32)
        Z = TSNE(n_components=2, perplexity=min(30, X_gpu.shape[0]-1), random_state=42).fit_transform(X_gpu)
        Z = cp.asnumpy(Z)
    except Exception:
        # cuML 없으면 CPU UMAP/PCA로 타협하거나, CPU t-SNE를 아주 드물게만 수행
        from umap import UMAP              # 설치 시 GPU 가속(rapids-umap) 가능, 없으면 CPU
        Z = UMAP(n_components=2, random_state=42).fit_transform(X.numpy())

    # 궤적 플롯
    plt.figure(figsize=(9,7))
    cmap = plt.get_cmap("tab10")
    for i, pid in enumerate(np.unique(P)):
        idx = np.where(P == pid)[0]
        order = np.argsort(S[idx])
        zi = Z[idx][order]
        plt.plot(zi[:,0], zi[:,1], '-', linewidth=1.5, alpha=0.85, color=cmap(i%10), label=f"P{pid}")
        plt.scatter(zi[:,0], zi[:,1], s=10, alpha=0.9, color=cmap(i%10))
        plt.scatter(zi[0,0], zi[0,1], s=60, marker='^', edgecolors='k', linewidths=0.7, color=cmap(i%10))
        plt.scatter(zi[-1,0],zi[-1,1], s=60, marker='s', edgecolors='k', linewidths=0.7, color=cmap(i%10))
    plt.title("t-SNE trajectories (post-training)")
    plt.axis('off'); plt.legend(fontsize=9, ncol=2)
    plt.tight_layout(); plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)

if __name__ == "__main__":
    run_gpu_tsne_and_plot("your_outdir/tsne_snapshots", "your_outdir/tsne_trajectories.png")
