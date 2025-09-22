#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gather fixed-viewpoint visualization frames from runs under exp/<exp_name> and
copy them into results/fixed_viewpoint/<exp_name>/ with concise file names.

Example:
  exp/exp6_ours_best_feature/WO__ICE__S42/visualization/fixed_viewpoint/step_1000_all_particles.png
  exp/exp6_ours_best_feature/RLSD__RBF__ICE__S42/visualization/fixed_viewpoint/step_001000_all_particles.png

→ results/fixed_viewpoint/exp6_ours_best_feature/
     ICE_S42.png
     ICE_S42.png  (same last-2 token base name; later run may need overwrite)

By default, output file names are derived from the last two tokens of the run
directory name split by "__" (e.g., WO__ICE__S42 → ICE_S42). If the run name
has fewer than two tokens, the whole run directory name is used.

This script does not perform image conversion. The destination file preserves
the source extension (.png/.jpg/.jpeg). Use --overwrite to replace files.


Usage:
python tools/gather_fixed_viewpoint.py --exp-name exp1_repulsion_kernel
  - For exp1 (repulsion-kernel ablation), filenames become
    <method>_<kernel>_<PROMPT>_<SSEED>.png, e.g., rlsd_cos_BULL_S123.png
  - Otherwise, use --name-mode and --name-tokens to control naming.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import shutil


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Gather fixed_viewpoint frames into results/fixed_viewpoint/<exp_name>")
    ap.add_argument("--exp-name", required=True, type=str,
                    help="Experiment directory under exp/ (e.g., exp6_ours_best_feature)")
    ap.add_argument("--base-dir", type=str, default="exp",
                    help="Base experiments directory (default: exp)")
    ap.add_argument("--out-root", type=str, default="results/fixed_viewpoint",
                    help="Root of output directory (default: results/fixed_viewpoint)")
    ap.add_argument("--step", type=int, default=1000,
                    help="Target step number to gather (default: 1000)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions without copying files")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing files at destination")
    ap.add_argument("--verbose", action="store_true",
                    help="Verbose logging")
    ap.add_argument("--name-mode", type=str, default="last2", choices=["last2", "run", "tokens"],
                    help="How to derive destination basename: last2 tokens, specific number of tokens, or full run name")
    ap.add_argument("--name-tokens", type=int, default=0,
                    help="If --name-mode=tokens, join the last N tokens (auto if 0)")
    ap.add_argument("--suffix", type=str, default=None,
                    help="Optional suffix to append to basename before extension (e.g., _fv)")
    return ap.parse_args()


def find_fixed_viewpoint_dir(run_dir: Path) -> Optional[Path]:
    """Return the path to the run's fixed_viewpoint directory if present.

    Searches common locations under the run directory recursively to be robust.
    """
    # Typical: <run>/visualization/fixed_viewpoint
    direct = run_dir / "visualization" / "fixed_viewpoint"
    if direct.exists():
        return direct

    # Fallback: search recursively for any directory named fixed_viewpoint
    candidates = list(run_dir.rglob("fixed_viewpoint"))
    for c in candidates:
        if c.is_dir():
            return c
    return None


def build_candidate_filenames(step: int) -> List[str]:
    """Return plausible file stems for the step image without extension."""
    return [
        f"step_{step}_all_particles",
        f"step_{step:06d}_all_particles",
    ]


def pick_existing_image(dir_path: Path, stems: List[str]) -> Optional[Path]:
    """Pick an existing image file for any of the stems with preferred extensions.

    Preference order: .png, .jpg, .jpeg
    """
    preferred_exts = [".png", ".jpg", ".jpeg"]
    for stem in stems:
        for ext in preferred_exts:
            fp = dir_path / f"{stem}{ext}"
            if fp.exists():
                return fp
    # As a fallback, allow any extension for the stem
    for stem in stems:
        for p in dir_path.glob(f"{stem}.*"):
            if p.is_file():
                return p
    return None


def derive_basename(run_dir_name: str, mode: str = "last2", tokens: int = 2) -> str:
    """Derive destination basename from run directory name.

    - last2: join the last two tokens split by "__" with underscore
    - run: use the full run directory name as-is
    """
    if mode == "run":
        return run_dir_name
    parts = run_dir_name.split("__")
    if mode == "tokens":
        n = tokens if tokens and tokens > 0 else 2
        n = min(n, len(parts))
        return "_".join(parts[-n:]) if n > 0 else run_dir_name
    # default last2
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return run_dir_name


def extract_exp1_ablation_label(run_dir_name: str) -> Optional[str]:
    """Return lowercase '<method>_<kernel>' for exp1-style names like
    'RLSD__COS__BULL__S123' or 'SVGD__RBF__ICE__S42'.
    """
    parts = run_dir_name.split("__")
    if len(parts) < 2:
        return None
    method, kernel = parts[0].strip().lower(), parts[1].strip().lower()
    # restrict to known sets to avoid accidental misuse
    if method in {"rlsd", "svgd"} and kernel in {"cos", "rbf"}:
        return f"{method}_{kernel}"
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_image(src: Path, dst: Path, overwrite: bool, dry_run: bool, verbose: bool) -> Tuple[bool, str]:
    """Copy src → dst respecting overwrite and dry-run. Returns (copied, message)."""
    if dst.exists() and not overwrite:
        return False, f"[SKIP] exists: {dst} (use --overwrite to replace)"
    if dry_run:
        return True, f"[DRY] copy {src} -> {dst}"
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True, f"[OK] copied {src} -> {dst}"


def main() -> int:
    args = parse_args()

    base_dir = Path(args.base_dir)
    exp_dir = base_dir / args.exp_name
    if not exp_dir.exists() or not exp_dir.is_dir():
        print(f"[ERR] exp directory not found: {exp_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_root) / args.exp_name
    ensure_dir(out_dir)

    # Exclude hidden and 'logs' directories
    run_dirs = [p for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith('.') and p.name != 'logs']
    if args.verbose:
        print(f"[INFO] scanning runs under {exp_dir} ({len(run_dirs)} dirs)")

    stems = build_candidate_filenames(args.step)
    copied = 0
    skipped = 0
    missing: List[str] = []
    handled_dests = set()  # avoid repeated messages for same destination

    # Auto naming policy: for repulsion-kernel experiments, use last 4 tokens to avoid collisions
    auto_is_exp1 = ("repulsion_kernel" in args.exp_name) or ("exp1" in args.exp_name)
    auto_mode = "tokens" if auto_is_exp1 else args.name_mode
    auto_tokens = args.name_tokens if args.name_tokens > 0 else (4 if auto_is_exp1 else 2)

    for run_dir in sorted(run_dirs, key=lambda p: p.name):
        fv_dir = find_fixed_viewpoint_dir(run_dir)
        if fv_dir is None:
            missing.append(f"{run_dir.name}: fixed_viewpoint dir not found")
            if args.verbose:
                print(f"[WARN] fixed_viewpoint not found for {run_dir}")
            continue

        src = pick_existing_image(fv_dir, stems)
        if src is None:
            missing.append(f"{run_dir.name}: step image not found in {fv_dir}")
            if args.verbose:
                print(f"[WARN] no step image in {fv_dir} (looked for {stems})")
            continue

        if auto_is_exp1:
            parts = run_dir.name.split("__")
            label = extract_exp1_ablation_label(run_dir.name)
            # Fallback to generic derivation if pattern not recognized
            if label is None:
                base_name = derive_basename(run_dir.name, mode=auto_mode, tokens=auto_tokens)
            else:
                # Append prompt and seed when available to avoid collisions
                if len(parts) >= 4:
                    prompt, seed = parts[2], parts[3]
                    base_name = f"{label}_{prompt}_{seed}"
                else:
                    base_name = label
        else:
            base_name = derive_basename(run_dir.name, mode=auto_mode, tokens=auto_tokens)
        ext = src.suffix.lower()
        if args.suffix:
            dst_name = f"{base_name}{args.suffix}{ext}"
        else:
            dst_name = f"{base_name}{ext}"
        dst = out_dir / dst_name

        # If this destination was already handled, avoid duplicate prints/ops
        if str(dst) in handled_dests:
            continue
        handled_dests.add(str(dst))

        ok, msg = copy_image(src, dst, overwrite=args.overwrite, dry_run=args.dry_run, verbose=args.verbose)
        print(msg)
        if ok:
            copied += 1
        else:
            skipped += 1

    # Summary
    print(f"[DONE] copied={copied} skipped={skipped} out_dir={out_dir}")
    if missing:
        print("[MISS] the following runs had missing images or dirs:")
        # Deduplicate messages to keep output concise
        for m in sorted(set(missing)):
            print(f"  - {m}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


