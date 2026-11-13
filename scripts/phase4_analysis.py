"""Phase 4 analysis on the development period: sensitivity, walk-forward, deflated Sharpe, capacity,
cost stress, SPY comparison, and the pre-stated vol-target rule. Writes research/phase4/."""
from __future__ import annotations

import json
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from intraday import data, metrics, research  # noqa: E402
from intraday.config import DEV_END, DEV_START, ROOT  # noqa: E402
from intraday.portfolio import PortfolioConfig, run_portfolio  # noqa: E402
from intraday.universe import FEATURES_FILE  # noqa: E402
from scripts import phase4_grid as G  # noqa: E402

OUT = ROOT / "research" / "phase4"
P2 = ROOT / "research" / "phase2"
DEFAULT = {"strategy": "noise_area", "band_mult": 1.0, "lookback": 14, "step": 30}

# diverging: red (negative) <- neutral gray -> blue (positive)  [reference palette ramps]
CMAP = LinearSegmentedColormap.from_list("div", ["#8f2424", "#e34948", "#f0efec", "#3987e5", "#104281"])
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def heatmap(df: pd.DataFrame, title: str, xlabel: str, ylabel: str, path, mark=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    v = df.to_numpy(dtype=float)
    lim = max(0.25, np.nanmax(np.abs(v)))
    im = ax.imshow(v, cmap=CMAP, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(df.shape[1]), [str(c) for c in df.columns], color=INK2)
    ax.set_yticks(range(df.shape[0]), [str(i) for i in df.index], color=INK2)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                dark = abs(v[i, j]) > 0.6 * lim
                ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if dark else INK)
    if mark is not None:
        i, j = mark
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=2))
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("net Sharpe (dev, measured costs)", color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def grid_table(name: str) -> pd.DataFrame:
    rets = G.load(name)
    rows = []
    for p in G.grid(name):
        r = rets.get(G.key(p))
        if r is not None:
            rows.append({**p, **metrics.summary(r)})
    return pd.DataFrame(rows)


def sensitivity() -> dict:
    out = {}
    t = grid_table("noise")
    if not t.empty:
        a = t[t.step == 30].pivot(index="band_mult", columns="lookback", values="sharpe")
        b = t[t.lookback == 14].pivot(index="band_mult", columns="step", values="sharpe")
        heatmap(a, "Noise area: net Sharpe by band multiplier × lookback (30-min decisions)",
                "lookback (days)", "band multiplier", OUT / "heatmap_noise_band_lookback.png",
                mark=(list(a.index).index(1.0), list(a.columns).index(14)))
        heatmap(b, "Noise area: net Sharpe by band multiplier × decision interval (14-day lookback)",
                "decision interval (min)", "band multiplier", OUT / "heatmap_noise_band_step.png",
                mark=(list(b.index).index(1.0), list(b.columns).index(30)))
        out["noise"] = {"band_x_lookback": a.round(3).to_dict(), "band_x_step": b.round(3).to_dict(),
                        "max_dd_band_x_lookback": t[t.step == 30].pivot(index="band_mult", columns="lookback",
                                                                       values="max_dd").round(3).to_dict(),
                        "points": len(t), "share_sharpe_gt_0": float((t.sharpe > 0).mean()),
                        "share_sharpe_gt_0.5": float((t.sharpe > 0.5).mean()),
                        "median_sharpe": float(t.sharpe.median()), "default_sharpe": float(
                            t[(t.band_mult == 1.0) & (t.lookback == 14) & (t.step == 30)].sharpe.iloc[0])}
    for name, x, y, xl, yl in (("orb", "top_n", "stop_atr_frac", "top N", "stop (fraction of ATR)"),
                               ("xs", "xs_n", "frac", "universe size", "decile fraction")):
        t = grid_table(name)
        if t.empty:
            continue
        m = t.pivot(index=y, columns=x, values="sharpe")
        heatmap(m, f"{name.upper()}: net Sharpe ({'IEX selection, optimistic fills' if name == 'orb' else 'rev1'})",
                xl, yl, OUT / f"heatmap_{name}.png")
        out[name] = {"sharpe": m.round(3).to_dict(), "share_sharpe_gt_0": float((t.sharpe > 0).mean()),
                     "best_sharpe": float(t.sharpe.max())}
    return out


def walk_forward() -> dict:
    rets = G.load("noise")
    grid = [p for p in G.grid("noise") if G.key(p) in rets]

    def run(p, start, end):
        r = rets[G.key(p)]
        idx = pd.Index(r.index)
        return r[(idx >= start) & (idx <= end)]

    oos, table = research.walk_forward(run, grid)
    default = run(DEFAULT, oos.index.min(), oos.index.max())
    table["chosen"] = table["chosen"].map(lambda p: {k: v for k, v in p.items() if k != "strategy"})
    return {"oos_walk_forward": metrics.summary(oos), "paper_default_same_windows": metrics.summary(default),
            "folds": table.astype({"test_start": str, "test_end": str}).to_dict("records")}


def effective_trials(base: pd.DataFrame) -> tuple[float, dict]:
    """N_eff per strategy family = rho + (1 - rho) * n, rho = average pairwise correlation of the
    family's daily returns (correlated variants are not independent trials); summed over families."""
    rets = {}
    for _, row in base.iterrows():
        p = research.RETURNS_DIR / f"{row['id']}.parquet"
        if p.exists():
            rets[row["id"]] = pd.read_parquet(p)["ret"]
    fam = base.set_index("id")["strategy"]
    per, total = {}, 0.0
    for f, ids in fam.groupby(fam).groups.items():
        ids = [i for i in ids if i in rets]
        n = len(ids)
        if n > 1:
            c = pd.DataFrame({i: rets[i] for i in ids}).corr().to_numpy()
            rho = float((c.sum() - n) / (n * (n - 1)))
            ne = rho + (1 - rho) * n
        else:
            rho, ne = 1.0, 1.0
        per[f] = {"configs": n, "avg_corr": round(rho, 3), "n_eff": round(ne, 2)}
        total += ne
    return total, per


def deflated() -> dict:
    """Deflated Sharpe of the candidate. Null variance of a Sharpe estimate = 1/(T-1) (under no skill,
    dispersion across trials is estimation noise; the noise-area grid's dispersion matches it). The
    cross-trial variance is also reported but is meaningless here: cost-killed strategies (Sharpe -26)
    are not 'unskilled draws' and blow it up."""
    t = research.load_trials()
    base = t[t["costs"].isin(["x1", "measured_x1"])]
    cand = pd.read_parquet(P2 / "noise_area__paper__measured_x1__daily.parquet")["ret"]
    port = phase3_portfolio(0.05, "x1")
    v_null = 1.0 / (cand.notna().sum() - 1)
    n_eff, per = effective_trials(base)
    rows = {}
    for label, n in (("single_preregistered", 1), ("effective_trials", int(round(n_eff))),
                     ("distinct_configurations", int(len(base))), ("all_recorded_rows", int(len(t)))):
        rows[label] = {"n": n, "sleeve": metrics.deflated_sharpe(cand, n, v_null),
                       "portfolio_5pct": metrics.deflated_sharpe(port, n, v_null)}
    cross = float((base["sharpe"] / np.sqrt(metrics.ANN)).var())
    return {"v_null_daily": v_null, "n_eff_by_family": per, "n_eff_total": n_eff, "by_trial_count": rows,
            "psr_vs_zero_sleeve": metrics.probabilistic_sharpe(cand),
            "reference_cross_trial_variance": {"var": cross, "sleeve": metrics.deflated_sharpe(cand, int(len(base)), cross)}}


def phase3_portfolio(target_vol: float, cost: str) -> pd.Series:
    from scripts.phase3_run import CFG
    r = pd.read_parquet(P2 / f"noise_area__paper__measured_{cost}__daily.parquet")["ret"]
    R = pd.DataFrame({"noise_area": r})
    res = run_portfolio(R, replace(CFG, target_vol=target_vol, kill_drawdown=float("inf")))
    p = res.returns
    return p[p.index >= res.scale[res.scale > 0].index.min()]


def vol_target_rule() -> dict:
    """Pre-stated: highest of 5/7.5/10% with dev max DD at 2x measured costs <= 12%."""
    rows = {}
    for tv in (0.05, 0.075, 0.10):
        rows[tv] = {c: {k: round(v, 4) for k, v in metrics.summary(phase3_portfolio(tv, c)).items()}
                    for c in ("x1", "x2", "x3")}
    ok = [tv for tv in rows if rows[tv]["x2"]["max_dd"] >= -0.12]
    return {"table": rows, "chosen": max(ok) if ok else None, "rule": "highest target with x2-cost max DD >= -12%"}


def capacity() -> dict:
    """Square-root impact on top of the measured spread, for the noise-area sleeve."""
    fills = pd.read_parquet(P2 / "noise_area__paper__measured_x1__fills.parquet")
    fills = fills[fills["purpose"].isin(["entry", "exit", "flatten"])]
    f = pd.read_parquet(FEATURES_FILE, filters=[("symbol", "==", "SPY")]).set_index("date")
    raw = data.load_daily("raw", DEV_START, DEV_END, ["SPY"]).set_index("date")
    adv_usd = (raw["close"] * raw["volume"]).rolling(20).median().shift(1)
    vol_d = f["ret1"].rolling(20).std().shift(1)
    rows = []
    for x in fills.itertuples():
        d = x.ts.tz_convert("America/New_York").date()
        bars = data.load_intraday(data.SPY_1MIN, d, ["SPY"]).set_index("ts")
        mv = float(bars["volume"].get(x.ts, np.nan) * bars["vwap"].get(x.ts, np.nan))
        rows.append({"date": d, "lev_notional": abs(x.qty) * x.price / 100_000.0, "minute_usd": mv,
                     "adv_usd": adv_usd.get(d, np.nan), "sigma": vol_d.get(d, np.nan)})
    df = pd.DataFrame(rows).dropna()
    # Sharpe vs one-way cost (bps): from the measured x1/x2/x3 sleeve runs
    js = {c: json.loads((P2 / f"noise_area__paper__measured_{c}__summary.json").read_text()) for c in ("x1", "x2", "x3")}
    m = json.loads((ROOT / "research/phase3/measured_costs.json").read_text())["spy"]["half_spread_bps_mean"]
    xs_, ys_ = [m, 2 * m, 3 * m], [js[c]["sharpe"] for c in ("x1", "x2", "x3")]
    slope, icpt = np.polyfit(xs_, ys_, 1)
    out = {"sharpe_per_bp_one_way": float(slope), "sharpe_at_zero_cost": float(icpt), "rows": []}
    for aum in (1e5, 1e6, 1e7, 1e8, 5e8, 1e9):
        q = df["lev_notional"] * aum
        part = q / df["minute_usd"]
        impact_bps = 1.0 * df["sigma"] * np.sqrt(q / df["adv_usd"]) * 1e4      # Y = 1 (conservative)
        extra = float(impact_bps.mean())
        out["rows"].append({"aum_usd": aum, "median_participation_of_fill_minute": float(part.median()),
                            "p90_participation": float(part.quantile(0.9)), "mean_impact_bps": extra,
                            "est_sharpe": float(js["x1"]["sharpe"] + slope * extra)})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"sensitivity": sensitivity(), "walk_forward": walk_forward(), "deflated_sharpe": deflated(),
           "vol_target_rule": vol_target_rule(), "capacity": capacity()}
    spy = pd.read_parquet(FEATURES_FILE, filters=[("symbol", "==", "SPY")]).set_index("date")["ret1"]
    res["spy_buy_hold_dev"] = metrics.summary(spy[(spy.index >= DEV_START) & (spy.index <= DEV_END)])
    (OUT / "phase4_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "sensitivity"}, indent=1, default=str)[:6000])
    print(json.dumps(res["sensitivity"], indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
