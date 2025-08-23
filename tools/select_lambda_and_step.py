# save as tools/select_lambda_and_step.py
import os, glob, pandas as pd
import numpy as np
from pathlib import Path

OUT = "./tools/analysis/exp1_lambda_coarse_svgd"
os.makedirs(OUT, exist_ok=True)

def read_last(df):
    return df.iloc[-1].to_dict() if len(df) else {}

def moving_slope(x, y, window=5):
    if len(x) < window: return 0.0
    xw = np.array(x[-window:], float)
    yw = np.array(y[-window:], float)
    A = np.vstack([xw, np.ones_like(xw)]).T
    m, _ = np.linalg.lstsq(A, yw, rcond=None)[0]
    return m

def select_best_step(qdf, kdf, ldf,
                     min_consistency=0.40,
                     fid_tau=0.99, div_tau=0.95,
                     neff_eps=1e-3, neff_win=5,
                     ratio_low=5.0, ratio_high=25.0):
    if qdf.empty: return None
    # align by step
    for name, df in [("k",kdf), ("l",ldf)]:
        if df is None or df.empty:
            # create empty with step
            pass
    steps = sorted(set(qdf['step']))
    q = qdf.set_index('step')
    k = kdf.set_index('step') if kdf is not None and not kdf.empty else pd.DataFrame(index=steps)
    l = ldf.set_index('step') if ldf is not None and not ldf.empty else pd.DataFrame(index=steps)

    fid_max = q['fidelity_mean'].max()
    div_max = q['inter_particle_diversity_mean'].max()

    # build candidate table
    rows=[]
    for s in steps:
        try:
            fid = q.at[s,'fidelity_mean']
            div = q.at[s,'inter_particle_diversity_mean']
            con = q.at[s,'cross_view_consistency_mean']
        except KeyError:
            continue
        if np.isnan(fid) or np.isnan(div) or np.isnan(con): continue

        # plateaus (neff)
        neff = k.at[s,'neff_mean'] if 'neff_mean' in k.columns else np.nan
        # slope over last neff_win points
        s_idx = [t for t in steps if t<=s][-neff_win:]
        neff_slope = np.nan
        if 'neff_mean' in k.columns and len(s_idx)>=2:
            neff_slope = moving_slope(s_idx, [k.at[t,'neff_mean'] for t in s_idx], neff_win)
        ratio = l.at[s,'scaled_repulsion_loss_ratio'] if 'scaled_repulsion_loss_ratio' in l.columns else np.nan

        ok = True
        if fid < fid_tau*fid_max: ok=False
        if div < div_tau*div_max: ok=False
        if con < min_consistency: ok=False
        if not np.isnan(neff_slope) and abs(neff_slope) > neff_eps: ok=False
        if not np.isnan(ratio) and not (ratio_low <= ratio <= ratio_high): ok=False

        # a simple utility for tie-break
        utility = 1.0*(fid/fid_max) + 0.8*(div/div_max) + 0.4*(con)
        rows.append((s, ok, utility, fid, div, con, neff, neff_slope, ratio))

    if not rows:
        return None
    # pick first satisfying; else best utility
    rows_sorted = sorted(rows, key=lambda r: r[0])
    for r in rows_sorted:
        if r[1]: return r
    return sorted(rows, key=lambda r: r[2], reverse=True)[0]

def scan(outdir="runs"):
    recs=[]
    for qpath in glob.glob(f"{outdir}/**/metrics/quantitative_metrics.csv", recursive=True):
        base = Path(qpath).parents[1]
        run = str(base)
        try:
            qdf = pd.read_csv(qpath)
        except: 
            continue
        kpath = base / "metrics" / "kernel_stats.csv"
        lpath = base / "metrics" / "losses.csv"
        kdf = pd.read_csv(kpath) if kpath.exists() else pd.DataFrame()
        ldf = pd.read_csv(lpath) if lpath.exists() else pd.DataFrame()

        sel = select_best_step(qdf, kdf, ldf)
        if sel is None:
            status = "no_pick"
            step = np.nan; util=np.nan; fid=div=con=neff=neff_slope=ratio=np.nan
        else:
            step, ok, util, fid, div, con, neff, neff_slope, ratio = sel
            status = "picked_ok" if ok else "picked_relaxed"
        # parse λ, repulsion, kernel from path name if 포함돼 있다면
        parts = base.name.split("_")
        meta = {kv.split("=")[0]: kv.split("=")[1] for kv in parts if "=" in kv}
        recs.append({
            "run": run, "status": status, "step": step, "utility": util,
            "fidelity": fid, "diversity": div, "consistency": con,
            "neff_mean": neff, "neff_slope": neff_slope, "ratio%": ratio,
            **meta
        })
    df = pd.DataFrame(recs)
    df.to_csv(f"{OUT}/lambda_step_selection.csv", index=False)
    print(f"saved -> {OUT}/lambda_step_selection.csv")
    return df

if __name__=="__main__":
    df=scan(outdir="exp/exp1_lambda_coarse_svgd")  # or base outdir
    # λ별 요약
    if "lambda_repulsion" in df.columns:
        g = df.groupby(["repulsion_type","kernel_type","prompt","lambda_repulsion"])
        summ = g.agg({"utility":"max","fidelity":"max","diversity":"max","consistency":"max","step":"min"}).reset_index()
        summ = summ.sort_values(["repulsion_type","kernel_type","prompt","utility"], ascending=[True,True,True,False])
        summ.to_csv(f"{OUT}/lambda_summary.csv", index=False)
        print(f"saved -> {OUT}/lambda_summary.csv")

# lambda_summary.csv에서 utility 상위 + consistency ≥ 0.40 + ratio% 5~25에 들어오는 λ/step 조합만 추출 → exp2 fine 범위로 세분화.