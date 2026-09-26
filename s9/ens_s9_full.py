"""S9 全套回測：xG 同時入 Elo／Dixon-Coles／LGB 三條預測線，同「冇 xG」基線用同一批比賽比較。
- 資料：football-data.co.uk 五大聯賽（2000 起暖身）＋ Understat xG（2014/15 起，賽後 xG 只用嚟更新賽後狀態同滾動特徵，shift(1)）。
- walk-forward：逐季重訓，只用測試季之前資料；集成權重＋向量標度只用前兩季季外預測擬合。
- 賠率權重 0：賠率只作價值注對手盤。applied_to_freeze=false。
用法：python ens_s9_full.py <results_dir> <understat_dir> <out.json>
"""
import sys, glob, json, math, warnings, datetime
from collections import defaultdict, deque
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
RES, UND, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DIVS = ("E0", "SP1", "I1", "D1", "F1")
W = 0.3  # 目標值＝W×入球＋(1−W)×xG（上次 DC 軌最佳）

def pdate(s):
    for f in ("%d/%m/%Y", "%d/%m/%y"):
        try: return datetime.datetime.strptime(s, f).date()
        except Exception: pass
    return None

rows = []
for f in sorted(glob.glob(f"{RES}/*.csv")):
    div = f.rsplit("_", 1)[-1][:-4]
    if div not in DIVS: continue
    d = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip", low_memory=False)
    for r in d.to_dict("records"):
        dt = pdate(str(r.get("Date", "")))
        try: gh, ga = int(r["FTHG"]), int(r["FTAG"])
        except Exception: continue
        if not dt or not isinstance(r.get("HomeTeam"), str): continue
        g = lambda *k: [float(r.get(x)) if pd.notna(r.get(x)) else np.nan for x in k]
        mx = g("MaxH", "MaxD", "MaxA")
        if np.isnan(mx).any(): mx = g("BbMxH", "BbMxD", "BbMxA")
        rows.append(dict(div=div, d=dt, h=r["HomeTeam"], a=r["AwayTeam"], gh=gh, ga=ga,
                         hs=r.get("HS"), as_=r.get("AS"), hst=r.get("HST"), ast=r.get("AST"),
                         mx=mx, b365=g("B365H", "B365D", "B365A")))
df = pd.DataFrame(rows).drop_duplicates(["div", "d", "h", "a"]).sort_values(["d", "div", "h"]).reset_index(drop=True)
df["season"] = [x.year if x.month >= 7 else x.year - 1 for x in df["d"]]

xg = {}
for f in glob.glob(f"{UND}/*.csv"):
    for r in pd.read_csv(f).to_dict("records"):
        xg[(r["div"], datetime.date.fromisoformat(r["date"]), r["home"], r["away"])] = r
hit = [xg.get((r.div, r.d, r.h, r.a)) for r in df.itertuples()]
df["has_xg"] = [x is not None for x in hit]
print("matches", len(df), "with xG", int(df.has_xg.sum()), flush=True)

MAXG = 9
FACT = np.array([math.factorial(i) for i in range(MAXG)], float)
def matrix(lh, la, rho=-0.05):
    ph = np.exp(-lh) * lh ** np.arange(MAXG) / FACT; pa = np.exp(-la) * la ** np.arange(MAXG) / FACT
    M = np.outer(ph, pa)
    M[0, 0] *= 1 - lh * la * rho; M[0, 1] *= 1 + lh * rho; M[1, 0] *= 1 + la * rho; M[1, 1] *= 1 - rho
    M = np.clip(M, 1e-12, None); return M / M.sum()
def hda(M): return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())
nf = lambda v: float(v) if pd.notna(v) else np.nan

def build(use_xg):
    elo = defaultdict(lambda: 1500.0); ATK = defaultdict(float); DEF = defaultdict(float)
    mu = defaultdict(lambda: 0.30); seen = defaultdict(int); last_s = {}; last_d = {}
    hist = defaultdict(lambda: deque(maxlen=10)); hv = defaultdict(lambda: deque(maxlen=5)); av = defaultdict(lambda: deque(maxlen=5))
    feats = []
    for r, x in zip(df.itertuples(), hit):
        h, a = r.h, r.a
        for t in (h, a):
            if last_s.get(t) not in (None, r.season):
                elo[t] = 1500 + (elo[t] - 1500) * 0.7; ATK[t] *= 0.8; DEF[t] *= 0.8
            last_s[t] = r.season
        eh, ea = elo[h], elo[a]
        lh = min(max(math.exp(mu[r.div] + ATK[h] - DEF[a] + 0.12), .15), 6); la = min(max(math.exp(mu[r.div] + ATK[a] - DEF[h]), .15), 6)
        ph, pd_, pa = hda(matrix(lh, la))
        f = dict(elo_diff=eh - ea, elo_exp=1 / (1 + 10 ** (-(eh + 60 - ea) / 400)), lam_h=lh, lam_a=la,
                 dc_ph=ph, dc_pd=pd_, dc_pa=pa, atk_h=ATK[h], def_h=DEF[h], atk_a=ATK[a], def_a=DEF[a],
                 seen_h=seen[h], seen_a=seen[a],
                 rest_h=(r.d - last_d[h]).days if h in last_d else np.nan, rest_a=(r.d - last_d[a]).days if a in last_d else np.nan,
                 month=r.d.month)
        keys = ["gf", "ga", "pts", "sh", "sot"] + (["xgf", "xga", "npxgf", "ppda", "deep", "xpts"] if use_xg else [])
        for tag, dq in (("h", hist[h]), ("a", hist[a])):
            for k in keys:
                v = [e[k] for e in dq if e[k] == e[k]]; f[f"{k}_l10_{tag}"] = float(np.mean(v)) if v else np.nan
        for tag, dq in (("h", hv[h]), ("a", av[a])):
            for k in ["gf", "ga"] + (["xgf", "xga"] if use_xg else []):
                v = [e[k] for e in dq if e[k] == e[k]]; f[f"{k}_ven_{tag}"] = float(np.mean(v)) if v else np.nan
        if use_xg:
            f["xgd_diff"] = (f["xgf_l10_h"] - f["xga_l10_h"]) - (f["xgf_l10_a"] - f["xga_l10_a"])
        feats.append(f)
        # ---- 賽後更新（本場 xG 只喺呢度用，下一場先見到）----
        gh, ga = float(r.gh), float(r.ga)
        th, ta = gh, ga
        sc = 1.0 if gh > ga else .5 if gh == ga else 0.0
        if use_xg and x is not None:
            xh, xa = float(x["home_xg"]), float(x["away_xg"])
            th, ta = W * gh + (1 - W) * xh, W * ga + (1 - W) * xa
            p = hda(matrix(max(xh, .05), max(xa, .05), 0.0)); sc = W * sc + (1 - W) * (p[0] + .5 * p[1])
        exp_ = 1 / (1 + 10 ** (-(eh + 60 - ea) / 400)); kk = 22 * (1 + min(abs(th - ta), 4) * .25)
        elo[h] = eh + kk * (sc - exp_); elo[a] = ea - kk * (sc - exp_)
        ATK[h] += .01 * (th - lh); DEF[a] -= .01 * (th - lh); ATK[a] += .01 * (ta - la); DEF[h] -= .01 * (ta - la)
        mu[r.div] += .0005 * ((th + ta) - (lh + la)); seen[h] += 1; seen[a] += 1
        def rec(gf, gag, s, st, side):
            e = dict(gf=gf, ga=gag, pts=3. if gf > gag else 1. if gf == gag else 0., sh=nf(s), sot=nf(st))
            if use_xg:
                o = "away" if side == "home" else "home"
                e.update(xgf=float(x[f"{side}_xg"]) if x else np.nan, xga=float(x[f"{o}_xg"]) if x else np.nan,
                         npxgf=float(x[f"{side}_npxg"]) if x else np.nan, ppda=float(x[f"{side}_ppda"]) if x else np.nan,
                         deep=float(x[f"{side}_deep"]) if x else np.nan, xpts=float(x[f"{side}_xpts"]) if x else np.nan)
            return e
        eh_, ea_ = rec(gh, ga, r.hs, r.hst, "home"), rec(ga, gh, r.as_, r.ast, "away")
        hist[h].append(eh_); hist[a].append(ea_); hv[h].append(eh_); av[a].append(ea_)
        last_d[h] = last_d[a] = r.d
    X = pd.DataFrame(feats); X["div"] = df["div"].astype("category"); return X

y = np.where(df.gh > df.ga, 0, np.where(df.gh == df.ga, 1, 2)); season = df.season.values
EPS = 1e-9
def norm(P): P = np.clip(P, EPS, None); return P / P.sum(1, keepdims=True)
def rps(P, yy):
    Y = np.zeros_like(P); Y[np.arange(len(yy)), yy] = 1
    return float(np.mean(np.sum((np.cumsum(P[:, :2], 1) - np.cumsum(Y[:, :2], 1)) ** 2, 1) / 2))
PAR = dict(objective="multiclass", num_class=3, learning_rate=.04, num_leaves=31, min_data_in_leaf=150,
           feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, lambda_l2=2., verbose=-1, num_threads=8)
TEST = list(range(2016, 2026))
GRID = [(l / 20, d / 20, 1 - l / 20 - d / 20) for l in range(6, 21) for d in range(0, 11) if -1e-9 <= 1 - l / 20 - d / 20 <= .5]

def run(use_xg):
    X = build(use_xg); warm = (X.seen_h.values >= 30) & (X.seen_a.values >= 30)
    Pe = np.full((len(X), 3), np.nan); Pl = np.full((len(X), 3), np.nan); Pd = norm(X[["dc_ph", "dc_pd", "dc_pa"]].values)
    for s in range(2012, 2026):
        tr, te = (season < s) & warm, (season == s) & warm
        lr = LogisticRegression(max_iter=1000).fit(X.loc[tr, ["elo_diff", "elo_exp"]].values, y[tr])
        Pe[te] = lr.predict_proba(X.loc[te, ["elo_diff", "elo_exp"]].values)
        Pl[te] = lgb.train(PAR, lgb.Dataset(X[tr], y[tr]), 400).predict(X[te])
        print(" ", "xg" if use_xg else "base", s, flush=True)
    def bl(a, m):
        L = a[0] * np.log(Pl[m]) + a[1] * np.log(Pd[m]) + a[2] * np.log(Pe[m]); return norm(np.exp(L - L.max(1, keepdims=True)))
    P = np.full((len(X), 3), np.nan); alphas = {}
    for s in TEST:
        te = (season == s) & warm; cal = np.isin(season, [s - 2, s - 1]) & warm
        a = min(GRID, key=lambda g: rps(bl(g, cal), y[cal])); alphas[s] = a
        vs = LogisticRegression(max_iter=2000, C=10.).fit(np.log(bl(a, cal)), y[cal])
        P[te] = vs.predict_proba(np.log(bl(a, te)))
    m = ~np.isnan(P[:, 0])
    single = {k: round(rps(norm(Q[m]), y[m]), 4) for k, Q in (("lgb", Pl), ("dc", Pd), ("elo", Pe))}
    return P, m, alphas, single, X.shape[1]

def backtest(P, m, which, thr, only=None, pmax=None, kelly=False):
    O = np.array([r for r in df[which].values]); n = st = pl = 0; bank = 1.0
    for i in np.where(m)[0]:
        o = O[i]
        if np.isnan(o).any() or min(o) <= 1.01: continue
        for k in range(3):
            if only is not None and k != only: continue
            if pmax is not None and P[i, k] > pmax: continue
            e = P[i, k] * o[k] - 1
            if e < thr: continue
            stake = min(.02, e / (o[k] - 1)) * bank if kelly else 1.0
            n += 1; st += stake; win = (y[i] == k); pl += stake * (o[k] - 1) if win else -stake
            if kelly: bank += stake * (o[k] - 1) if win else -stake
    return dict(bets=n, yield_pct=round(100 * pl / st, 2) if st else None)

out = {"W_goals": W, "test_seasons": [TEST[0], TEST[-1]], "market_beta": 0, "applied_to_freeze": False}
for tag, ux in (("base_no_xg", False), ("full_xg_3tracks", True)):
    P, m, al, single, nfeat = run(ux)
    both = m & df.has_xg.values
    r = dict(n_features=nfeat, matches=int(m.sum()), rps_all=round(rps(P[m], y[m]), 4),
             rps_xg_covered=round(rps(P[both], y[both]), 4), single_track_rps=single,
             alpha_last=al[TEST[-1]], acc=round(float(np.mean(P[m].argmax(1) == y[m])), 4))
    vb = {}
    for t in (0, .02, .05, .10): vb[f"flat_best_thr{int(t*100):02d}"] = backtest(P, m, "mx", t)
    for t in (0, .05, .10): vb[f"flat_b365_thr{int(t*100):02d}"] = backtest(P, m, "b365", t)
    vb["kelly_best_thr05"] = backtest(P, m, "mx", .05, kelly=True)
    for t in (.05, .10): vb[f"draw_only_best_thr{int(t*100):02d}"] = backtest(P, m, "mx", t, only=1, pmax=.30)
    r["value_bets"] = vb; out[tag] = r; print(tag, json.dumps(r, ensure_ascii=False), flush=True)
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
