#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Optional, Tuple


def find_runs(exp_dir: Path) -> List[Path]:
    runs = []
    if not exp_dir.exists():
        return runs
    for child in sorted(exp_dir.iterdir()):
        if child.is_dir():
            runs.append(child)
    return runs


def _parse_step_from_prefix(name: str) -> int:
    if not name.startswith("step_"):
        return -1
    digits = []
    for ch in name[len("step_"):]:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return -1
    try:
        return int("".join(digits))
    except Exception:
        return -1


# ---------------- Multi-viewpoints helpers ----------------
def find_multi_view_sequence(run_dir: Path) -> Optional[Path]:
    mv_dir = run_dir / "visualizations" / "multi_viewpoints"
    if not mv_dir.exists():
        return None

    candidates: List[Tuple[int, Path]] = []
    for child in mv_dir.iterdir():
        if child.is_dir():
            step_value = _parse_step_from_prefix(child.name)
            candidates.append((step_value, child))
    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def collect_mv_frames(sequence_dir: Path) -> List[Path]:
    # Expected files: view_000.png, view_001.png, ...
    frames: List[Path] = []
    if not sequence_dir.exists():
        return frames
    frames = sorted(sequence_dir.glob("view_*.png"))
    if frames:
        return frames
    # Some exports might put frames directly under the step dir
    frames = sorted(sequence_dir.glob("*.png"))
    if frames:
        return frames
    # Or one level deeper
    for child in sequence_dir.iterdir():
        if child.is_dir():
            deep = sorted(child.glob("view_*.png"))
            if deep:
                return deep
    return []


# ---------------- Fixed-viewpoint helpers ----------------
def find_fixed_viewpoint_frames(run_dir: Path) -> List[Path]:
    fixed_dir = run_dir / "visualizations" / "fixed_viewpoint"
    if not fixed_dir.exists():
        return []
    frames = list(fixed_dir.glob("step_*_all_particles.png"))
    frames.sort(key=lambda p: _parse_step_from_prefix(p.name))
    return frames


def save_gif(frames: List[Path], out_path: Path, fps: int) -> None:
    import imageio.v2 as imageio
    ims = [imageio.imread(str(p)) for p in frames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), ims, duration=1.0 / max(1, fps))


def build_gifs_for_experiment(
    workspace_root: Path,
    experiment_name: str,
    fps_mv: int = 30,
    fps_fv: int = 8,
    overwrite: bool = False,
) -> Tuple[int, int, int]:
    """
    Returns (num_mv_done, num_fv_done, total_runs_seen)
    """
    exp_dir = workspace_root / "exp" / experiment_name
    mv_results_dir = workspace_root / "results" / "multi_viewpoints" / experiment_name
    fv_results_dir = workspace_root / "results" / "fixed_viewpoint" / experiment_name

    mv_done = 0
    fv_done = 0
    runs = find_runs(exp_dir)
    for run_dir in runs:
        run_name = run_dir.name

        # Multi-viewpoints
        try:
            seq_dir = find_multi_view_sequence(run_dir)
            if seq_dir is not None:
                mv_frames = collect_mv_frames(seq_dir)
                if mv_frames:
                    mv_out = mv_results_dir / f"{run_name}.gif"
                    if not mv_out.exists() or overwrite:
                        save_gif(mv_frames, mv_out, fps=fps_mv)
                        print(f"[OK][MV] {mv_out} ({len(mv_frames)} frames)")
                        mv_done += 1
                    else:
                        print(f"[KEEP][MV] {mv_out}")
                else:
                    print(f"[SKIP][MV] No frames for run {run_name} in {seq_dir}")
            else:
                print(f"[MISS][MV] No multi_viewpoints for run {run_name}")
        except Exception as e:
            print(f"[FAIL][MV] {run_name}: {e}")

        # Fixed-viewpoint
        try:
            fv_frames = find_fixed_viewpoint_frames(run_dir)
            if fv_frames:
                fv_out = fv_results_dir / f"{run_name}.gif"
                if not fv_out.exists() or overwrite:
                    save_gif(fv_frames, fv_out, fps=fps_fv)
                    print(f"[OK][FV] {fv_out} ({len(fv_frames)} frames)")
                    fv_done += 1
                else:
                    print(f"[KEEP][FV] {fv_out}")
            else:
                print(f"[MISS][FV] No fixed_viewpoint frames for run {run_name}")
        except Exception as e:
            print(f"[FAIL][FV] {run_name}: {e}")

    return mv_done, fv_done, len(runs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create GIFs for a given experiment: multi_viewpoints and fixed_viewpoint.")
    ap.add_argument("experiment", type=str, help="Experiment name under exp/, e.g., exp_gaussian_reproduce")
    ap.add_argument("--fps-mv", type=int, default=30, help="FPS for multi_viewpoints GIFs")
    ap.add_argument("--fps-fv", type=int, default=8, help="FPS for fixed_viewpoint GIFs")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing GIFs")
    ap.add_argument("--workspace", type=str, default=str(Path(__file__).resolve().parents[1]), help="Workspace root path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace).resolve()
    mv_done, fv_done, total = build_gifs_for_experiment(
        workspace_root,
        args.experiment,
        fps_mv=args["fps_mv"] if isinstance(args, dict) else args.fps_mv,
        fps_fv=args["fps_fv"] if isinstance(args, dict) else args.fps_fv,
        overwrite=args.overwrite,
    )
    mv_dir = workspace_root / "results" / "multi_viewpoints" / args.experiment
    fv_dir = workspace_root / "results" / "fixed_viewpoint" / args.experiment
    print(f"[SUMMARY] MV {mv_done}/{total} -> {mv_dir}")
    print(f"[SUMMARY] FV {fv_done}/{total} -> {fv_dir}")


if __name__ == "__main__":
    main()


"""
python tools/make_gif.py exp_gaussian_reproduce --fps-mv 30 --fps-fv 8 --overwrite

 python3 tools/make_multi_viewpoints_gif.py exp0_baseline --fps 30
"""
