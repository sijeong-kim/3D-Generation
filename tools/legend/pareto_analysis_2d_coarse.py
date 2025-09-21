#!/usr/bin/env python3
"""
2D Pareto Analysis (row=prompt, col=(kernel, repulsion))

- % changes vs baseline at same step
- Baseline의 plateau 시점 이후 점은 회색으로 표현(남겨둠)
- ε-제약(Feasible): Fidelity ≥ -ε, Diversity ≥ δ (feasible만 진하게, infeasible은 연하게)
- λ 그룹별 step 오름차순 점선 연결 (plateau 이후는 회색 점선)
- Pareto 비지배점(*) + 우하향 체인(--), 동일가중(X) [모두 feasible subset에서 계산]
- Subplot: 행=prompt, 열=(kernel_type, repulsion_type)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.patches import Patch, Rectangle

def format_repulsion_type(repulsion_type):
    """Format repulsion type for display."""
    if repulsion_type.lower() == 'rlsd':
        return 'RLSD-F'
    elif repulsion_type.lower() == 'svgd':
        return 'SVGD'
    elif repulsion_type.lower() == 'wo':
        return 'Baseline'
    else:
        return repulsion_type.upper()

def get_prompt_description(prompt_key):
    """Get full description for prompt abbreviations."""
    prompt_descriptions = {
        'hamburger': 'a photo of a hamburger',
        'icecream': 'a photo of an ice cream', 
        'cactus': 'a small saguaro cactus planted in a clay pot',
        'tulip': 'a photo of a tulip'
    }
    return prompt_descriptions.get(prompt_key.lower(), prompt_key.title())


# ------------------------- global style -------------------------
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
})

# ------------------------- plateau config -------------------------
USE_PLATEAU = True
PLATEAU_METRIC = "fidelity_mean"  # baseline에서 플래토 판단할 지표
PLATEAU_WINDOW = 10
PLATEAU_TOL = 1e-3
PLATEAU_PATIENCE = 3
PLATEAU_BUFFER_STEPS = 0  # 검출지점 이후 여유 버퍼

# ------------------------- ε-constraint thresholds -------------------------
FIDELITY_MIN_PCT = -5.0   # ΔF ≥ -5%
DIVERSITY_MIN_PCT = 30.0  # ΔD ≥ +30%

# ------------------------- parsing & loading -------------------------

def parse_config_name(config_name: str):
    # New pattern: REPULSION__KERNEL__λVALUE__PROMPT__S{SEED}
    # e.g., RLSD__COS__λ100__CACT__S42
    pattern = r'(\w+)__(\w+)__λ([\dK.]+)__(\w+)__S(\d+)'
    m = re.match(pattern, config_name)
    if not m:
        return None
    
    # Convert lambda value back to numeric format
    lambda_str = m.group(3)
    if lambda_str.endswith('K'):
        lambda_value = str(int(float(lambda_str[:-1]) * 1000))
    else:
        lambda_value = lambda_str
    
    return {
        "repulsion_type": m.group(1).lower(),  # RLSD -> rlsd
        "kernel_type": m.group(2).lower(),     # COS -> cosine, RBF -> rbf
        "lambda_repulsion": lambda_value,
        "prompt": m.group(4).lower(),          # CACT -> cactus
        "seed": m.group(5),
    }

def load_baseline_data(baseline_dir: Path) -> dict:
    out = {}
    for item in baseline_dir.iterdir():
        if not item.is_dir() or item.name.startswith('.') or item.name == 'logs':
            continue
        csv_path = item / "metrics" / "quantitative_metrics.csv"
        if csv_path.exists():
            # New baseline naming: PROMPT__S{SEED} -> prompt=PROMPT_seed=SEED
            # e.g., CACT__S42 -> prompt=cactus_seed=42
            if '__S' in item.name:
                parts = item.name.split('__S')
                if len(parts) == 2:
                    prompt_part = parts[0].lower()  # CACT -> cactus
                    seed_part = parts[1]
                    key = f"prompt={prompt_part}_seed={seed_part}"
                else:
                    key = item.name
            else:
                key = item.name
            out[key] = pd.read_csv(csv_path)
            print(f"Loaded baseline: {item.name} -> {key}")
    return out

# ------------------------- step alignment & diffs -------------------------

def get_step_value(df: pd.DataFrame, idx: int, step_col: str = "step"):
    if step_col in df.columns and len(df) > 0:
        return idx, int(df[step_col].iloc[idx])
    return idx, idx

def calculate_metric_differences_at_step(exp_df, baseline_df, metrics, idx, step_col="step"):
    if step_col in exp_df.columns and step_col in baseline_df.columns:
        step_val = exp_df[step_col].iloc[idx]
        b_mask = baseline_df[step_col] == step_val
        if b_mask.any():
            b_row = baseline_df[b_mask].iloc[-1]
        else:
            b_row = baseline_df.iloc[min(idx, len(baseline_df) - 1)]
    else:
        b_row = baseline_df.iloc[min(idx, len(baseline_df) - 1)]
    e_row = exp_df.iloc[idx]
    diffs = {}
    for raw_col, pretty in metrics.items():
        if raw_col in e_row and raw_col in b_row:
            e, b = float(e_row[raw_col]), float(b_row[raw_col])
            if np.isfinite(e) and np.isfinite(b) and b != 0:
                diffs[pretty] = (e - b) / b * 100.0
            else:
                diffs[pretty] = np.nan
        else:
            diffs[pretty] = np.nan
    return diffs

# ------------------------- Pareto logic -------------------------

def _clean_xy(df, x_col, y_col):
    out = df[[x_col, y_col]].copy()
    mask = np.isfinite(out[x_col]) & np.isfinite(out[y_col])
    return df.loc[mask]

def pareto_efficient_mask(points: np.ndarray, maximize=(True, True), eps=1e-12) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    eff = np.ones(n, dtype=bool)
    for i in range(n):
        if not eff[i]:
            continue
        ge_all = np.ones(n, dtype=bool)
        strict_any = np.zeros(n, dtype=bool)
        for d, is_max in enumerate(maximize):
            if is_max:
                ge_d = pts[:, d] >= pts[i, d] - eps
                gt_d = pts[:, d] >  pts[i, d] + eps
            else:
                ge_d = pts[:, d] <= pts[i, d] + eps
                gt_d = pts[:, d] <  pts[i, d] - eps
            ge_all &= ge_d
            strict_any |= gt_d
        dominates = ge_all & strict_any
        if np.any(dominates):
            eff[i] = False
    return eff

def find_pareto_2d(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    df2 = _clean_xy(df, x_col, y_col)
    if df2.empty:
        return df2
    pts = df2[[x_col, y_col]].to_numpy()
    mask = pareto_efficient_mask(pts, maximize=(True, True))
    pareto = df2.loc[mask].drop_duplicates(subset=[x_col, y_col])
    return pareto

def _robust_minmax(s: pd.Series):
    q1, q3 = np.nanpercentile(s, [25, 75])
    iqr = max(q3 - q1, 1e-12)
    lo = max(np.nanmin(s), q1 - 1.5*iqr)
    hi = min(np.nanmax(s), q3 + 1.5*iqr)
    s_clipped = s.clip(lo, hi)
    return (s_clipped - lo) / max(hi - lo, 1e-12)

def pick_equal_weight_point(df: pd.DataFrame, x_col: str, y_col: str):
    cand = find_pareto_2d(df, x_col, y_col)
    if cand.empty:
        return None
    xz = _robust_minmax(cand[x_col]); yz = _robust_minmax(cand[y_col])
    dist = np.sqrt((1 - xz)**2 + (1 - yz)**2)
    idx = dist.idxmin()
    return cand.loc[idx]

# ------------------------- plotting -------------------------

def create_2d_pareto_subplot(
    ax, data_full, x_col: str, y_col: str, draw_cbar=False,
    lam_col="Lambda", step_col="Step"
):
    """
    data_full: 반드시 IsPlateau(bool), IsFeasible(bool) 컬럼을 포함해야 함.
    Pareto/체인/eq-weight는 IsFeasible==True subset에서 계산/표시.
    나머지 점들은 조건별 스타일링으로 모두 표시.
    """
    ax.grid(True, alpha=0.3)
    ax.axhline(0, linestyle=':', linewidth=1.0)
    ax.axvline(0, linestyle=':', linewidth=1.0)

    if data_full is None or data_full.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
        return pd.DataFrame(), None, None

    # λ 색상 매핑 (feasible 표현에만 사용)
    lam_vals = np.sort(data_full[lam_col].unique())
    vmin = float(np.min(lam_vals)); vmax = float(np.max(lam_vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return pd.DataFrame(), None, None
    if abs(vmax - vmin) < 1e-12: vmax = vmin + 1e-12
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    # 그룹별 스타일: pre/post × feasible/infeasible
    for lam in lam_vals:
        g = data_full[data_full[lam_col] == lam].copy()
        order_key = step_col if step_col in g.columns else 'StepIdx'
        g = g.sort_values(order_key)

        # 선: pre-plateau는 λ색 점선, post-plateau는 회색 점선
        g_pre = g[~g["IsPlateau"]]
        if len(g_pre) >= 2:
            ax.plot(g_pre[x_col], g_pre[y_col], linestyle='--', alpha=0.5, linewidth=0.9,
                    color=sm.to_rgba(lam), zorder=1)
        g_post = g[g["IsPlateau"]]
        if len(g_post) >= 2:
            ax.plot(g_post[x_col], g_post[y_col], linestyle='--', alpha=0.35, linewidth=0.9,
                    color='0.65', zorder=1)

        # 점: 조건별 스타일
        # pre-plateau & feasible (진하게, λ색)
        sub = g_pre[g_pre["IsFeasible"]]
        ax.scatter(sub[x_col], sub[y_col], s=24, alpha=0.9, color=sm.to_rgba(lam), zorder=2)

        # pre-plateau & infeasible (희미하게, λ색)
        sub = g_pre[~g_pre["IsFeasible"]]
        ax.scatter(sub[x_col], sub[y_col], s=22, alpha=0.3, color=sm.to_rgba(lam), zorder=1)

        # post-plateau & feasible (회색 진하게)
        sub = g_post[g_post["IsFeasible"]]
        ax.scatter(sub[x_col], sub[y_col], s=28, alpha=0.5, color='0.35', edgecolor='0.25', linewidths=0.6, zorder=2)

        # post-plateau & infeasible (회색 연하게)
        sub = g_post[~g_post["IsFeasible"]]
        ax.scatter(sub[x_col], sub[y_col], s=22, alpha=0.1, color='0.6', zorder=1)
        
    # Pareto/체인/eq-weight: feasible subset만
    feas = data_full[data_full["IsFeasible"]].copy()
    pareto_raw_df = pd.DataFrame()
    pareto_chain_df = pd.DataFrame()
    eq_point = None
    if not feas.empty:
        pareto_raw = find_pareto_2d(feas, x_col, y_col)
        if not pareto_raw.empty:
            pareto_raw_df = pareto_raw.copy()
            ax.scatter(pareto_raw[x_col], pareto_raw[y_col], c='red', s=70,
                       alpha=0.95, marker='*', label='Pareto optimal', zorder=4)
            chain = pareto_raw.sort_values(x_col, ascending=False).copy()
            chain['_y_cummax_descx'] = chain[y_col].cummax()
            chain = chain[chain[y_col] >= chain['_y_cummax_descx'] - 1e-12]
            pareto_chain_df = chain.drop(columns=['_y_cummax_descx']).sort_values(x_col, ascending=True)
            ax.plot(pareto_chain_df[x_col], pareto_chain_df[y_col],
                    linestyle='--', linewidth=1.8, alpha=0.95, zorder=5)
        eq = pick_equal_weight_point(feas, x_col, y_col)
        if eq is not None:
            ax.scatter([eq[x_col]], [eq[y_col]], s=120,
                       marker='X', edgecolor='k', linewidths=1.2, c='gold',
                       label='Eq-weight best', zorder=6)
            eq_point = eq

    if draw_cbar:
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('λ (repulsion)')
        return pareto_raw_df, pareto_chain_df, eq_point, cbar
    return pareto_raw_df, pareto_chain_df, eq_point, None

def draw_feasible_region(ax, xlim, ylim):
    """
    Feasible(연한 회색) + Infeasible(빗금) 동시 표시.
    경계선은 점선. zorder=0으로 배경에 깐다.
    """
    # --- Feasible 영역: ΔF ≥ FIDELITY_MIN_PCT, ΔD ≥ DIVERSITY_MIN_PCT ---
    feas_x0 = max(FIDELITY_MIN_PCT, xlim[0])
    feas_y0 = max(DIVERSITY_MIN_PCT, ylim[0])
    feas_w  = max(0.0, xlim[1] - feas_x0)
    feas_h  = max(0.0, ylim[1] - feas_y0)
    if feas_w > 0 and feas_h > 0:
        feas_rect = Rectangle((feas_x0, feas_y0), feas_w, feas_h,
                              facecolor='0.9', edgecolor='none', alpha=0.15, zorder=0)
        ax.add_patch(feas_rect)

    # --- Infeasible 영역: 좌측(ΔF < ε) ---
    left_w = max(0.0, FIDELITY_MIN_PCT - xlim[0])
    if left_w > 0:
        left = Rectangle((xlim[0], ylim[0]), left_w, ylim[1]-ylim[0],
                         facecolor='none', edgecolor='0.7', hatch='////',
                         linewidth=0.0, alpha=0.3, zorder=0)
        ax.add_patch(left)

    # --- Infeasible 영역: 하단(ΔD < δ) ---
    bot_h = max(0.0, DIVERSITY_MIN_PCT - ylim[0])
    if bot_h > 0:
        bottom = Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], bot_h,
                           facecolor='none', edgecolor='0.7', hatch="////",
                           linewidth=0.0, alpha=0.3, zorder=0)
        ax.add_patch(bottom)

    # --- 경계선(ε-제약) ---
    ax.axvline(FIDELITY_MIN_PCT, linestyle='--', linewidth=1.0, color='0.5', alpha=0.85)
    ax.axhline(DIVERSITY_MIN_PCT, linestyle='--', linewidth=1.0, color='0.5', alpha=0.85)

# ------------------------- plateau detection -------------------------

def detect_plateau_step(series, steps, window=10, tol=1e-3, patience=3):
    """
    롤링 평균 -> 1차 차분 |Δ| < tol 상태가 'patience'회 연속 확인되면 해당 step을 plateau로 간주
    """
    s = pd.Series(series).rolling(window, min_periods=max(2, window//2)).mean()
    d = s.diff().abs()
    if d.isna().all():
        return int(steps.iloc[-1])
    run = 0
    for i, ok in enumerate(d < tol):
        run = run + 1 if bool(ok) else 0
        if run >= patience:
            return int(steps.iloc[i])
    return int(steps.iloc[-1])

# ------------------------- main runner -------------------------

def main():
    baseline_exp = "exp0_baseline"
    # New experiment structure: multiple experiments by repulsion type
    # Exp1 (coarse) experiments only
    experiment_exps = [
        "exp1_lambda_coarse_rlsd", 
        "exp1_lambda_coarse_svgd"
    ]
    base_dir = Path("exp")
    output_dir = Path("analysis/pareto_analysis_exp1")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "inter_particle_diversity_mean": "Diversity Improvement (%)",
        "fidelity_mean": "Fidelity Change (%)",
        "cross_view_consistency_mean": "Cross-View Consistency Change (%)",
    }

    # 로드
    baseline_dir = base_dir / baseline_exp
    baseline_data = load_baseline_data(baseline_dir)

    # Collect all experiment configs from all experiment directories
    exp_configs = []
    for experiment_exp in experiment_exps:
        experiment_dir = base_dir / experiment_exp
        if experiment_dir.exists():
            configs = [p.name for p in experiment_dir.iterdir()
                      if p.is_dir() and not p.name.startswith('.') and p.name != "logs"]
            exp_configs.extend(configs)
            print(f"Found {len(configs)} experiment configs in {experiment_exp}")
    print(f"Total {len(exp_configs)} experiment configs")

    # 수집
    config_data = {}
    plateau_by_prompt = {}  # prompt별 baseline plateau step 기록
    for cfg_name in exp_configs:
        params = parse_config_name(cfg_name)
        if not params or params["seed"] != "42":
            continue
        key = (params["kernel_type"].upper(),
               get_prompt_description(params["prompt"]),
               format_repulsion_type(params["repulsion_type"]))
        config_data.setdefault(key, [])
        
        # Find the correct experiment directory for this config
        experiment_dir = None
        for exp_name in experiment_exps:
            exp_dir = base_dir / exp_name
            if exp_dir.exists() and (exp_dir / cfg_name).exists():
                experiment_dir = exp_dir
                break
        
        if experiment_dir is None:
            continue
            
        metrics_path = experiment_dir / cfg_name / "metrics" / "quantitative_metrics.csv"
        if not metrics_path.exists():
            continue
        exp_df = pd.read_csv(metrics_path)
        base_key = f"prompt={params['prompt']}_seed=42"
        if base_key not in baseline_data:
            continue

        base_df = baseline_data[base_key].copy()

        # ------ baseline plateau step 계산 (자르지 않고 flag만 달기) ------
        if USE_PLATEAU:
            if "step" in base_df.columns and PLATEAU_METRIC in base_df.columns:
                p_step = detect_plateau_step(
                    base_df[PLATEAU_METRIC], base_df["step"],
                    window=PLATEAU_WINDOW, tol=PLATEAU_TOL, patience=PLATEAU_PATIENCE
                ) + PLATEAU_BUFFER_STEPS
            else:
                p_step = None
        else:
            p_step = None
        plateau_by_prompt[get_prompt_description(params["prompt"])] = p_step

        # Δ 누적 (전체 step에 대해)
        for i in range(len(exp_df)):
            idx_used, step_label = get_step_value(exp_df, i, step_col="step")
            diffs = calculate_metric_differences_at_step(exp_df, base_df, metrics, idx_used, step_col="step")
            if all(np.isnan(v) for v in diffs.values()):
                continue
            is_plateau = (p_step is not None) and (int(step_label) >= int(p_step))
            config_data[key].append({
                "Kernel": params["kernel_type"].upper(),
                "Prompt": get_prompt_description(params["prompt"]),
                "Repulsion": format_repulsion_type(params["repulsion_type"]),
                "Lambda": float(params["lambda_repulsion"]),
                "StepIdx": idx_used,
                "Step": step_label,
                "IsPlateau": is_plateau,
                **diffs
            })

    all_rows = [r for rows in config_data.values() for r in rows]
    if not all_rows:
        print("No data assembled. Check paths/config names.")
        return
    all_df = pd.DataFrame(all_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(output_dir / "pareto_analysis_all_rows.csv", index=False)

    x_col, y_col = "Fidelity Change (%)", "Diversity Improvement (%)"

    # --- Grid: rows=prompt, cols=(kernel, repulsion) ---
    prompts = sorted({k[1] for k in config_data.keys()})
    col_pairs = sorted({(k[0], k[2]) for k in config_data.keys()})

    n_rows, n_cols = len(prompts), len(col_pairs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5*n_cols, 5.5*n_rows), squeeze=False)

    # 열 헤더
    for j, (kernel, rep) in enumerate(col_pairs):
        axes[0, j].set_title(f"{kernel} + {rep}".replace("_", " "), fontsize=13, pad=12, fontweight='bold')

    # 공통 축 범위
    xmin = np.nanpercentile(all_df[x_col], 1); xmax = np.nanpercentile(all_df[x_col], 99)
    ymin = np.nanpercentile(all_df[y_col], 1); ymax = np.nanpercentile(all_df[y_col], 99)
    padx = 0.05*(xmax - xmin); pady = 0.05*(ymax - ymin)
    xlim = (xmin - padx, xmax + padx); ylim = (ymin - pady, ymax + pady)
    # 제약선 보이도록 확장
    xlim = (min(xlim[0], FIDELITY_MIN_PCT - 1.0), max(xlim[1], FIDELITY_MIN_PCT + 1.0))
    ylim = (min(ylim[0], DIVERSITY_MIN_PCT - 1.0), max(ylim[1], DIVERSITY_MIN_PCT + 1.0))

    # pareto_all, eq_best_all = [], []
    feasible_summary = []
    colorbar_drawn = False
    last_valid_axes = None
    
    pareto_raw_all, pareto_chain_all, eq_best_all = [], [], []

    for i, prompt in enumerate(prompts):
        axes[i, 0].text(-0.22, 0.5, prompt, transform=axes[i, 0].transAxes,
                        rotation=90, va='center', ha='right', fontsize=12, weight='bold')
        p_step = plateau_by_prompt.get(prompt, None)

        for j, (kernel, rep) in enumerate(col_pairs):
            ax = axes[i, j]
            key = (kernel, prompt, rep)
            df_panel = pd.DataFrame(config_data.get(key, []))
            if df_panel.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)
                continue

            # ε-제약 마스크 (표시/통계용)
            df_panel["IsFeasible"] = (df_panel[x_col] >= FIDELITY_MIN_PCT) & (df_panel[y_col] >= DIVERSITY_MIN_PCT)

            # 통계 요약
            feasible_summary.append({
                "Prompt": prompt, "Kernel": kernel, "Repulsion": rep,
                "FeasibleCount": int(df_panel["IsFeasible"].sum()),
                "Total": int(len(df_panel)),
                "FeasibleRatio": float(df_panel["IsFeasible"].mean()),
            })

            # 컬러바는 우하단에서 한 번만
            draw_cbar = (i == n_rows-1 and j == n_cols-1 and not colorbar_drawn)
            # pareto_df, eq_point, cbar = create_2d_pareto_subplot(
            #     ax, df_panel, x_col, y_col, draw_cbar=draw_cbar
            # )
            
            pareto_raw_df, pareto_chain_df, eq_point, cbar = create_2d_pareto_subplot(
                ax, df_panel, x_col, y_col, draw_cbar=draw_cbar
            )
            
            if cbar is not None:
                colorbar_drawn = True
            last_valid_axes = ax

            # 축 포맷/범위
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            
            # Enhanced styling
            ax.tick_params(labelsize=9, direction='in', length=4, width=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)

            # 가용영역 음영/경계선
            draw_feasible_region(ax, xlim, ylim)

            # plateau step 텍스트(옵션)
            if USE_PLATEAU and p_step is not None:
                ax.text(0.98, 0.02, f"Plateau ≥ {p_step}", transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=8, color='0.45')

            if i < n_rows - 1: ax.set_xlabel("")
            if j > 0:          ax.set_ylabel("")

            # 데이터 축적 (feasible Pareto/eq)
            if pareto_raw_df is not None and not pareto_raw_df.empty:
                pareto_tmp = pareto_raw_df.copy()
                pareto_tmp["Panel"] = f"{prompt} | {kernel}+{rep}"
                pareto_raw_all.append(pareto_tmp)
            if pareto_chain_df is not None and not pareto_chain_df.empty:
                pareto_tmp = pareto_chain_df.copy()
                pareto_tmp["Panel"] = f"{prompt} | {kernel}+{rep}"
                pareto_chain_all.append(pareto_tmp)
            if eq_point is not None and not pd.isna(eq_point.get(x_col, np.nan)):
                eq_best_all.append(eq_point)

    # 우하단 패널이 비어 컬러바를 못 그렸다면, 마지막 유효 패널에서 컬러바 표시
    if not colorbar_drawn and last_valid_axes is not None:
        lam_vals = np.sort(all_df["Lambda"].unique())
        vmin = float(np.min(lam_vals)); vmax = float(np.max(lam_vals))
        if abs(vmax - vmin) < 1e-12: vmax = vmin + 1e-12
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=plt.cm.viridis)
        cbar = plt.colorbar(sm, ax=last_valid_axes, fraction=0.046, pad=0.02)
        cbar.set_label('λ (repulsion)')

    # 공통 라벨/레이아웃
    fig.suptitle(
        f"Pareto Frontier Analysis (All Steps; Plateau Points in Gray; Feasible: ΔF ≥ {FIDELITY_MIN_PCT:.0f}% & ΔD ≥ {DIVERSITY_MIN_PCT:.0f}%)\n"
        "Diversity Improvement (%) vs Fidelity Change (%) (↑ is Better; Chain is Down-Right on Feasible Set)",
        fontsize=16, y=0.98, fontweight='bold'
    )
    fig.supxlabel("Fidelity Change (%)", fontsize=12, fontweight='bold')
    fig.supylabel("Diversity Improvement (%)", fontsize=12, fontweight='bold')

    # 범례
    star = mpl.lines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=9, label='Pareto optimal')
    equ  = mpl.lines.Line2D([], [], color='gold', marker='X', markeredgecolor='k',
                            linestyle='None', markersize=9, label='Eq-weight best')
    # fe_patch = Patch(alpha=0.1, label=f'Feasible region (ΔF≥{FIDELITY_MIN_PCT:.0f}%, ΔD≥{DIVERSITY_MIN_PCT:.0f}%)')
    # inf_region = Patch(facecolor='none', edgecolor='0.5', hatch='////', label='Infeasible region (ε-constraint)')
    
        # 새 패치 (설명용)
    feas_region = Patch(facecolor='0.9', edgecolor='none', alpha=0.15,
                        label='Feasible region (ΔF≥ε, ΔD≥δ)')
    inf_region  = Patch(facecolor='none', edgecolor='0.5', hatch='////',
                        label='Infeasible region')
        

    pre_hard = mpl.lines.Line2D([], [], color='black', marker='o', linestyle='None', alpha=0.9, markersize=6, label='Pre-plateau feasible')
    pre_fade = mpl.lines.Line2D([], [], color='black', marker='o', linestyle='None', alpha=0.25, markersize=6, label='Pre-plateau infeasible')
    post_hard= mpl.lines.Line2D([], [], color='0.35', marker='o', linestyle='None', alpha=0.9, markersize=6, label='Post-plateau feasible')
    post_fade= mpl.lines.Line2D([], [], color='0.6', marker='o', linestyle='None', alpha=0.25, markersize=6, label='Post-plateau infeasible')
    # fig.legend([star, equ, fe_patch, pre_hard, pre_fade, post_hard, post_fade],
    #            ['Pareto optimal', 'Eq-weight best', fe_patch.get_label(),
    #             'Pre feasible', 'Pre infeasible', 'Post feasible', 'Post infeasible'],
    #            loc="lower center", bbox_to_anchor=(0.5, 0.02),
    #            ncol=4, frameon=False, fontsize=9)
    
    # fig.legend([star, equ, inf_region, pre_hard, pre_fade, post_hard, post_fade],
    #        ['Pareto optimal', 'Eq-weight best', 'Infeasible region (ε-constraint)',
    #         'Pre feasible', 'Pre infeasible', 'Post feasible', 'Post infeasible'],
    #        loc="lower center", bbox_to_anchor=(0.5, 0.02),
    #        ncol=4, frameon=False, fontsize=9)
    
    # 기존 star, equ, pre_hard, pre_fade, post_hard, post_fade와 함께 사용
    fig.legend(
        [star, equ, feas_region, inf_region, pre_hard, pre_fade, post_hard, post_fade],
        ['Pareto optimal', 'Eq-weight best',
        'Feasible region', 'Infeasible region',
        'Pre feasible', 'Pre infeasible', 'Post feasible', 'Post infeasible'],
        loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=4, frameon=False, fontsize=9
    )

    plt.tight_layout(rect=[0.03, 0.11, 0.99, 0.95])
    
    # Save in multiple high-quality formats for dissertation
    plt.savefig(output_dir / "pareto_2d_analysis.png", dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.savefig(output_dir / "pareto_2d_analysis.pdf", bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.savefig(output_dir / "pareto_2d_analysis.svg", bbox_inches="tight", facecolor='white', edgecolor='none')


    # 결과 저장
    if pareto_raw_all:
        pd.concat(pareto_raw_all, ignore_index=True)\
        .to_csv(output_dir / "pareto_optimal_points.csv", index=False)

    if pareto_chain_all:
        pd.concat(pareto_chain_all, ignore_index=True)\
        .to_csv(output_dir / "pareto_chain_points.csv", index=False)

    if eq_best_all:
        pd.DataFrame(eq_best_all).to_csv(output_dir / "equal_weight_best_points.csv", index=False)

    if feasible_summary:
        pd.DataFrame(feasible_summary).to_csv(output_dir / "feasible_summary_by_panel.csv", index=False)

if __name__ == "__main__":
    main()
