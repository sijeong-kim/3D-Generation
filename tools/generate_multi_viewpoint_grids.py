import argparse
import os
from typing import List, Tuple, Optional

import imageio.v2 as imageio
import numpy as np


def parse_int_list(csv: str) -> List[int]:
    return [int(x.strip()) for x in csv.split(",") if x.strip() != ""]


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def compute_uniform_views(view_total: int, num_views: int) -> List[int]:
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    num = min(num_views, view_total)
    # Use exclusive endpoint to avoid near-duplicate first/last views
    # Example: view_total=120, num=8 -> [0,15,30,45,60,75,90,105]
    lin = np.linspace(0, view_total, num=num, endpoint=False)
    idx = lin.astype(int)
    # Deduplicate while preserving order (safety for edge cases)
    seen = set()
    uniq: List[int] = []
    for i in idx.tolist():
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def discover_available_views(run_dir: str, step: int, view_total: int) -> List[int]:
    """Discover available view indices by scanning one particle directory.

    Falls back to [0..view_total-1] if nothing is found.
    """
    base = os.path.join(
        run_dir,
        "visualizations",
        "multi_viewpoints",
        f"step_{step}_view_{view_total}_iid_particles",
    )
    if not os.path.isdir(base):
        return list(range(view_total))
    # Prefer particle_0, otherwise pick the first particle_* directory
    particle_dir: Optional[str] = None
    p0 = os.path.join(base, "particle_0")
    if os.path.isdir(p0):
        particle_dir = p0
    else:
        for name in sorted(os.listdir(base)):
            if name.startswith("particle_") and os.path.isdir(os.path.join(base, name)):
                particle_dir = os.path.join(base, name)
                break
    if particle_dir is None:
        return list(range(view_total))

    views: List[int] = []
    for fname in os.listdir(particle_dir):
        if not fname.endswith(".png"):
            continue
        if not fname.startswith("view_"):
            continue
        stem = fname.split(".")[0]
        try:
            idx = int(stem.split("_")[-1])
            views.append(idx)
        except Exception:
            continue
    if not views:
        return list(range(view_total))
    return sorted(set(views))


def build_expected_paths(
    run_dir: str,
    step: int,
    view_total: int,
    particles: List[int],
    views: List[int],
) -> List[Tuple[int, int, str]]:
    expected = []
    base = os.path.join(
        run_dir,
        "visualizations",
        "multi_viewpoints",
        f"step_{step}_view_{view_total}_iid_particles",
    )
    for view_idx in views:
        for particle_id in particles:
            img_path = os.path.join(
                base, f"particle_{particle_id}", f"view_{view_idx:03d}.png"
            )
            expected.append((view_idx, particle_id, img_path))
    return expected


def read_and_normalize_image(path: str, target_shape: Tuple[int, int, int] = None) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if target_shape is not None:
        target_h, target_w = target_shape[0], target_shape[1]
        if img.shape[0] != target_h or img.shape[1] != target_w:
            try:
                import cv2  # Lazy import to avoid hard dependency when no resize is needed
            except Exception as e:
                raise RuntimeError(
                    "Resizing required but OpenCV (cv2) is not available. Install with `pip install opencv-python`."
                ) from e
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return img


def compose_grid(
    run_dir: str,
    out_path: str,
    step: int,
    view_total: int,
    particles: List[int],
    views: List[int],
) -> bool:
    # If views was computed globally and may not match this run's availability,
    # recompute uniformly from this run's discovered view list to keep spacing consistent.
    discovered = discover_available_views(run_dir, step, view_total)
    if len(discovered) > 0 and set(views).issubset(set(discovered)) is False:
        total = len(discovered)
        lin = np.linspace(0, total, num=min(len(views), total), endpoint=False)
        picks = lin.astype(int).tolist()
        views = [discovered[i] for i in picks]

    expected = build_expected_paths(run_dir, step, view_total, particles, views)
    missing = [(v, p, pth) for (v, p, pth) in expected if not os.path.isfile(pth)]
    if missing:
        print(f"[WARN] Skipping {os.path.basename(run_dir)}; missing {len(missing)} images.")
        # Optionally, print the first few missing paths for debugging
        for (_, _, mp) in missing[:4]:
            print(f"       missing: {mp}")
        return False

    # Determine tile size from the first image
    first_img_path = expected[0][2]
    first_img = imageio.imread(first_img_path)
    if first_img.ndim == 2:
        first_img = np.stack([first_img, first_img, first_img], axis=-1)
    if first_img.shape[-1] == 4:
        first_img = first_img[..., :3]
    tile_h, tile_w = first_img.shape[0], first_img.shape[1]

    # Build grid: rows are views, columns are particle ids
    rows: List[np.ndarray] = []
    for view_idx in views:
        row_tiles: List[np.ndarray] = []
        for particle_id in particles:
            img_path = os.path.join(
                run_dir,
                "visualizations",
                "multi_viewpoints",
                f"step_{step}_view_{view_total}_iid_particles",
                f"particle_{particle_id}",
                f"view_{view_idx:03d}.png",
            )
            tile = read_and_normalize_image(img_path, (tile_h, tile_w, 3))
            row_tiles.append(tile)
        row_img = np.concatenate(row_tiles, axis=1)
        rows.append(row_img)

    grid = np.concatenate(rows, axis=0)
    ensure_dir(os.path.dirname(out_path))
    imageio.imwrite(out_path, grid)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate N x M grids of multi-viewpoints across particles for runs."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=os.path.join("exp", "exp6_ours_best"),
        help="Directory containing run folders (e.g., BULL__S42).",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="If provided, use runs from exp/<exp-name> (e.g., exp6_ours_best)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save output grids. If omitted, uses results/multi_viewpoints/<exp-name>_app.",
    )
    parser.add_argument(
        "--views",
        type=str,
        default=None,
        help="Comma-separated list of view indices (0-119) to use as rows. If omitted, uses --num-views to pick uniform indices.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=8,
        help="Number of uniformly-divided views to use as rows when --views is not provided.",
    )
    parser.add_argument(
        "--particles",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated list of particle ids to use as columns.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1000,
        help="Optimization step used in folder naming (e.g., step_1000).",
    )
    parser.add_argument(
        "--view-total",
        type=int,
        default=120,
        help="Total number of views used in folder naming (e.g., view_120).",
    )
    parser.add_argument(
        "--only-run",
        type=str,
        default=None,
        help="If provided, only process this specific run name (folder basename).",
    )

    args = parser.parse_args()

    runs_dir = args.runs_dir
    if args.exp_name is not None and args.exp_name.strip() != "":
        runs_dir = os.path.join("exp", args.exp_name)

    # Determine default out_dir if not provided: results/multi_viewpoints/<exp-name>_app_n<num_views>
    if args.out_dir is None or args.out_dir.strip() == "":
        exp_name = args.exp_name
        if exp_name is None or exp_name.strip() == "":
            exp_name = os.path.basename(os.path.normpath(runs_dir))
        # Determine effective number of views to annotate
        if args.views is not None and args.views.strip() != "":
            try:
                effective_num_views = len(parse_int_list(args.views))
            except Exception:
                effective_num_views = args.num_views
        else:
            effective_num_views = args.num_views
        out_dir = os.path.join(
            "results",
            "multi_viewpoints",
            f"{exp_name}_app_v{int(effective_num_views)}",
        )
    else:
        out_dir = args.out_dir
    if args.views is not None and args.views.strip() != "":
        views = parse_int_list(args.views)
    else:
        # Discover available view indices from files, then sample uniformly
        # to match user's expectation: e.g., 6 images and num_views=2 -> [0, 3]
        # This uses indices (0..N-1) in the discovered list, not absolute 0..view_total-1
        # to ensure correct spacing given actual availability.
        discovered = None
        # If only processing a single run, we can discover directly from it; otherwise
        # we will re-compute per-run inside compose_grid.
        if args.only_run:
            only_path = os.path.join(runs_dir, args.only_run)
            discovered = discover_available_views(only_path, args.step, args.view_total)
        if discovered is None or len(discovered) == 0:
            views = compute_uniform_views(args.view_total, args.num_views)
        else:
            total = len(discovered)
            # Evenly select indices over the discovered list, endpoint-exclusive
            lin = np.linspace(0, total, num=min(args.num_views, total), endpoint=False)
            picks = lin.astype(int).tolist()
            views = [discovered[i] for i in picks]
    particles = parse_int_list(args.particles)

    ensure_dir(out_dir)

    if args.only_run:
        candidate_runs = [os.path.join(runs_dir, args.only_run)]
    else:
        candidate_runs = [
            os.path.join(runs_dir, d)
            for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d))
        ]

    processed = 0
    for run_path in sorted(candidate_runs):
        run_name = os.path.basename(run_path)
        out_path = os.path.join(out_dir, f"{run_name}.png")
        ok = compose_grid(
            run_path,
            out_path,
            step=args.step,
            view_total=args.view_total,
            particles=particles,
            views=views,
        )
        if ok:
            processed += 1
            print(f"[OK] Saved {out_path}")

    print(f"Done. Processed {processed} / {len(candidate_runs)} runs.")


if __name__ == "__main__":
    main()


