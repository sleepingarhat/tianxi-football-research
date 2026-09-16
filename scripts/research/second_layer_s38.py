#!/usr/bin/env python3
"""S38 第二層推薦結算 walk-forward（只讀凍結軌，唔改 λ、唔改矩陣、唔升指紋）。

第一層＝矩陣 M 加總三格 (P_H,P_D,P_A)，一分不動、對 RPS。
第二層只出一句結算：
  1 一面倒 max(P_H,P_A) >= tau  → 強隊 −1 勝
  2 近盤   |P_H − P_A| <= delta → 弱隊 +1 勝
  3 其餘                        → 三格較高嗰邊直勝（唔出和、唔出讓球）
主閘＝推薦堆嘅矩陣隱含贏率對實際贏率差距 ≤ 0.02；副＝三類覆蓋率；另看逐季一致性。
λ 層鎖死＝生產參數 lr=0.04 rho=-0.05 gamma=0.12 regress=0.80 warm=40。
資料：football-data.co.uk 五大聯賽 2000–2026（Matches.csv），評分季 >= 2021。
"""
from __future__ import annotations
import csv, io, json, math, os, urllib.request
from collections import defaultdict

CSV = "/tmp/fb/Matches.csv"; MAXG = 10; EVAL_FROM = 2021
K = {"H": 0, "D": 1, "A": 2}
TAUS = [0.60, 0.65, 0.70, 0.75]; DELTAS = [0.05, 0.07, 0.10]
DIVS = ["E0", "D1", "SP1", "I1", "F1"]


def download():
    rows = []
    for y in range(0, 26):
        s = f"{y:02d}{y+1:02d}"
        for d in DIVS:
            try:
                raw = urllib.request.urlopen(
                    f"https://www.football-data.co.uk/mmz4281/{s}/{d}.csv", timeout=30
                ).read().decode("utf-8", "ignore")
            except Exception:
                continue
            for r in csv.DictReader(io.StringIO(raw)):
                dt = (r.get("Date") or "").strip().split("/")
                if len(dt) != 3 or not r.get("FTHG"):
                    continue
                dd, mm, yy = dt
                yy = int(yy); yy = yy + 2000 if yy < 100 else yy
                try:
                    gh, ga = int(r["FTHG"]), int(r["FTAG"])
                except Exception:
                    continue
                rows.append({"Division": d, "MatchDate": f"{yy:04d}-{int(mm):02d}-{int(dd):02d}",
                             "HomeTeam": r["HomeTeam"].strip(), "AwayTeam": r["AwayTeam"].strip(),
                             "FTHome": gh, "FTAway": ga, "FTResult": r.get("FTR", "").strip()})
    rows.sort(key=lambda r: (r["MatchDate"], r["Division"]))
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def season_of(y, m):
    return y if m >= 7 else y - 1


def build_matrix(lh, la, rho):
    ph = [math.exp(-lh)]; pa = [math.exp(-la)]
    for k in range(1, MAXG + 1):
        ph.append(ph[-1] * lh / k); pa.append(pa[-1] * la / k)
    M = [[ph[i] * pa[j] for j in range(MAXG + 1)] for i in range(MAXG + 1)]
    for (i, j), t in (((0, 0), 1 - lh * la * rho), ((0, 1), 1 + lh * rho),
                      ((1, 0), 1 + la * rho), ((1, 1), 1 - rho)):
        M[i][j] *= max(t, 1e-9)
    s = sum(sum(r) for r in M)
    return [[v / s for v in r] for r in M]


def margins(M):
    out = defaultdict(float)
    for i in range(MAXG + 1):
        for j in range(MAXG + 1):
            out[i - j] += M[i][j]
    return out


def load():
    if not os.path.exists(CSV):
        download()
    rows = []
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            y, m, _ = (int(x) for x in r["MatchDate"].split("-"))
            rows.append({"div": r["Division"], "home": r["HomeTeam"], "away": r["AwayTeam"],
                         "gh": int(r["FTHome"]), "ga": int(r["FTAway"]), "res": r["FTResult"],
                         "sea": season_of(y, m)})
    return rows


def frozen(rows, *, lr=0.04, rho=-0.05, gamma=0.12, regress=0.80, warm=40):
    atk = defaultdict(float); dfn = defaultdict(float); seen = defaultdict(int)
    base = defaultdict(lambda: math.log(1.35)); last = {}; out = []
    for r in rows:
        d, h, a, sea = r["div"], r["home"], r["away"], r["sea"]
        if last.get(d) != sea:
            last[d] = sea
            for t in list(atk):
                atk[t] *= regress; dfn[t] *= regress
        lh = min(max(math.exp(base[d] + gamma + atk[h] - dfn[a]), 0.15), 6.0)
        la = min(max(math.exp(base[d] - gamma + atk[a] - dfn[h]), 0.15), 6.0)
        if seen[h] >= warm and seen[a] >= warm and sea >= EVAL_FROM:
            mg = margins(build_matrix(lh, la, rho))
            ph = sum(v for k, v in mg.items() if k > 0)
            pa = sum(v for k, v in mg.items() if k < 0)
            out.append({"sea": sea, "p": (ph, mg[0], pa), "mg": dict(mg),
                        "k": K[r["res"]], "d": r["gh"] - r["ga"]})
        eh, ea = r["gh"] - lh, r["ga"] - la
        atk[h] += lr * eh; dfn[a] -= lr * eh; atk[a] += lr * ea; dfn[h] -= lr * ea
        for t in (h, a):
            atk[t] = min(max(atk[t], -1.2), 1.2); dfn[t] = min(max(dfn[t], -1.2), 1.2)
        base[d] += 0.002 * ((r["gh"] + r["ga"]) - (lh + la)) / max(lh + la, 0.5)
        seen[h] += 1; seen[a] += 1
    return out


def leg(mg, side_home, line):
    w = p = l = 0.0
    for k, v in mg.items():
        eff = (k if side_home else -k) + line
        if eff > 0: w += v
        elif eff == 0 and line != 0: p += v
        else: l += v
    return w, p, l


def bucket(s, tau, delta):
    ph, _, pa = s["p"]; strong_home = ph >= pa
    if max(ph, pa) >= tau: return 1, strong_home, -1
    if abs(ph - pa) <= delta: return 2, not strong_home, 1
    return 3, strong_home, 0


def outcome(s, side_home, line):
    eff = (s["d"] if side_home else -s["d"]) + line
    return "win" if eff > 0 else ("push" if eff == 0 and line != 0 else "lose")


def evaluate(samples, tau, delta):
    st = {b: {"n": 0, "pw": 0.0, "pp": 0.0, "win": 0, "push": 0} for b in (1, 2, 3)}
    per_sea = defaultdict(lambda: defaultdict(lambda: {"n": 0, "pw": 0.0, "win": 0}))
    for s in samples:
        b, side_home, line = bucket(s, tau, delta)
        w, p, _ = leg(s["mg"], side_home, line)
        o = outcome(s, side_home, line)
        x = st[b]; x["n"] += 1; x["pw"] += w; x["pp"] += p
        x["win"] += int(o == "win"); x["push"] += int(o == "push")
        ps = per_sea[s["sea"]][b]; ps["n"] += 1; ps["pw"] += w; ps["win"] += int(o == "win")
    total = len(samples); out = {"tau": tau, "delta": delta, "n": total, "buckets": {}}
    for b, x in st.items():
        n = x["n"] or 1
        out["buckets"][b] = {"n": x["n"], "coverage": round(x["n"] / total, 4),
                             "pred_win": round(x["pw"] / n, 4), "act_win": round(x["win"] / n, 4),
                             "pred_push": round(x["pp"] / n, 4), "act_push": round(x["push"] / n, 4),
                             "cal_gap_win": round(abs(x["pw"] - x["win"]) / n, 4)}
    out["per_season"] = {str(sea): {b: {"n": v["n"], "pred_win": round(v["pw"] / max(v["n"], 1), 4),
                                        "act_win": round(v["win"] / max(v["n"], 1), 4)}
                                    for b, v in bs.items()} for sea, bs in sorted(per_sea.items())}
    out["max_cal_gap"] = round(max(out["buckets"][b]["cal_gap_win"] for b in (1, 2, 3)), 4)
    return out


def main():
    samples = frozen(load())
    grid = [evaluate(samples, t, d) for t in TAUS for d in DELTAS]
    res = {"n": len(samples), "eval_from": EVAL_FROM, "gate_cal_gap_max": 0.02,
           "chosen": {"tau": 0.60, "delta": 0.10, "minus1_gate_passed": False},
           "grid": sorted(grid, key=lambda r: r["max_cal_gap"])}
    with open("/tmp/bt/s38_second_layer.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for r in sorted(grid, key=lambda r: (r["tau"], r["delta"])):
        b = r["buckets"]
        print(r["tau"], r["delta"], [b[k]["coverage"] for k in (1, 2, 3)],
              [b[k]["cal_gap_win"] for k in (1, 2, 3)])


main()
