#!/usr/bin/env python3
"""S26b 回測：「永遠唔會預測和局」點解決最好？

λ 層完全鎖死＝現行生產軌（全局 atk/dfn、gamma=0.12、lr=0.04、regress=0.80、
warm=40、rho=-0.05、逐聯賽 base）。本回測唔改 λ、唔改矩陣生成式。

分兩層試：
  第一層（只改標籤，機率一分不動 → RPS／logloss／ECE 全部唔會變）
    A argmax（現行）
    B 主客距離閘：|P_H − P_A| < τ → 出和
    C 和局加權：argmax(P_H, c·P_D, P_A)
    D 弱勢閘：max(P_H, P_A) < θ → 出和
  第二層（真正改機率，要過主閘先有資格）
    E 和局對數機率平移 δ：P_D → P_D·e^δ 再歸一（常數，唔逐場、唔跟近況）

尺：命中率、和局召回／精確率、macro-F1、平衡命中率、出和比例；
第二層另量 RPS、1X2 log-loss、ECE 十桶對照基準。
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

CSV = "/tmp/fb/Matches.csv"
MAXG = 10
EVAL_FROM = 2021
K = {"H": 0, "D": 1, "A": 2}


def season_of(y, m):
    return y if m >= 7 else y - 1


def build_matrix(lh, la, rho):
    ph = [math.exp(-lh)]
    pa = [math.exp(-la)]
    for k in range(1, MAXG + 1):
        ph.append(ph[-1] * lh / k)
        pa.append(pa[-1] * la / k)
    M = [[ph[i] * pa[j] for j in range(MAXG + 1)] for i in range(MAXG + 1)]
    for (i, j), t in (((0, 0), 1 - lh * la * rho), ((0, 1), 1 + lh * rho),
                      ((1, 0), 1 + la * rho), ((1, 1), 1 - rho)):
        M[i][j] *= max(t, 1e-9)
    s = sum(sum(r) for r in M)
    return [[v / s for v in r] for r in M]


def derive3(M):
    ph = pd_ = pa = 0.0
    for i in range(MAXG + 1):
        for j in range(MAXG + 1):
            p = M[i][j]
            if i > j:
                ph += p
            elif i == j:
                pd_ += p
            else:
                pa += p
    return ph, pd_, pa


def rps3(p, k):
    y = [0.0, 0.0, 0.0]
    y[k] = 1.0
    c = cy = s = 0.0
    for i in range(2):
        c += p[i]
        cy += y[i]
        s += (c - cy) ** 2
    return s / 2


def load():
    rows = []
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            y, m, _ = (int(x) for x in r["MatchDate"].split("-"))
            rows.append({"div": r["Division"], "home": r["HomeTeam"], "away": r["AwayTeam"],
                         "gh": int(r["FTHome"]), "ga": int(r["FTAway"]), "res": r["FTResult"],
                         "sea": season_of(y, m)})
    return rows


def frozen_probs(rows, *, lr=0.04, rho=-0.05, gamma=0.12, regress=0.80, warm=40):
    """跑一次生產軌，抽出評分季每場嘅凍結三格（之後所有策略共用同一批機率）。"""
    atk = defaultdict(float)
    dfn = defaultdict(float)
    seen = defaultdict(int)
    base = defaultdict(lambda: math.log(1.35))
    last = {}
    out = []
    for r in rows:
        d, h, a, sea = r["div"], r["home"], r["away"], r["sea"]
        if last.get(d) != sea:
            last[d] = sea
            for t in list(atk):
                atk[t] *= regress
                dfn[t] *= regress
        lh = min(max(math.exp(base[d] + gamma + atk[h] - dfn[a]), 0.15), 6.0)
        la = min(max(math.exp(base[d] - gamma + atk[a] - dfn[h]), 0.15), 6.0)
        if seen[h] >= warm and seen[a] >= warm and sea >= EVAL_FROM:
            pH, pD, pA = derive3(build_matrix(lh, la, rho))
            out.append({"div": d, "sea": sea, "p": (pH, pD, pA), "k": K[r["res"]]})
        eh, ea = r["gh"] - lh, r["ga"] - la
        atk[h] += lr * eh
        dfn[a] -= lr * eh
        atk[a] += lr * ea
        dfn[h] -= lr * ea
        for t in (h, a):
            atk[t] = min(max(atk[t], -1.2), 1.2)
            dfn[t] = min(max(dfn[t], -1.2), 1.2)
        base[d] += 0.002 * ((r["gh"] + r["ga"]) - (lh + la)) / max(lh + la, 0.5)
        seen[h] += 1
        seen[a] += 1
    return out


def label_metrics(samples, chooser):
    n = len(samples)
    hit = 0
    pred_cnt = [0, 0, 0]
    act_cnt = [0, 0, 0]
    tp = [0, 0, 0]
    per_sea = defaultdict(lambda: [0, 0])
    for s in samples:
        g = chooser(s["p"])
        k = s["k"]
        pred_cnt[g] += 1
        act_cnt[k] += 1
        if g == k:
            hit += 1
            tp[g] += 1
        per_sea[s["sea"]][0] += 1
        per_sea[s["sea"]][1] += int(g == k)
    rec = [tp[i] / act_cnt[i] if act_cnt[i] else 0.0 for i in range(3)]
    pre = [tp[i] / pred_cnt[i] if pred_cnt[i] else 0.0 for i in range(3)]
    f1 = [2 * pre[i] * rec[i] / (pre[i] + rec[i]) if pre[i] + rec[i] else 0.0 for i in range(3)]
    return {
        "n": n,
        "accuracy": round(hit / n, 4),
        "pred_share": [round(c / n, 4) for c in pred_cnt],
        "draw_recall": round(rec[1], 4),
        "draw_precision": round(pre[1], 4),
        "macro_f1": round(sum(f1) / 3, 4),
        "balanced_acc": round(sum(rec) / 3, 4),
        "per_season_acc": {str(s): round(v[1] / v[0], 4) for s, v in sorted(per_sea.items())},
    }


def prob_metrics(samples, shift):
    n = len(samples)
    s_rps = s_ll = 0.0
    bins = defaultdict(lambda: [0, 0.0, 0])
    per_sea = defaultdict(lambda: [0, 0.0, 0.0])
    draw_argmax = 0
    hit = 0
    mult = math.exp(shift)
    for s in samples:
        pH, pD, pA = s["p"]
        pD2 = pD * mult
        z = pH + pD2 + pA
        p = (pH / z, pD2 / z, pA / z)
        k = s["k"]
        g = max(range(3), key=lambda i: p[i])
        draw_argmax += int(g == 1)
        hit += int(g == k)
        s_rps += rps3(p, k)
        s_ll += -math.log(max(p[k], 1e-12))
        per_sea[s["sea"]][0] += 1
        per_sea[s["sea"]][1] += rps3(p, k)
        per_sea[s["sea"]][2] += -math.log(max(p[k], 1e-12))
        for i in range(3):
            b = min(9, int(p[i] * 10))
            bins[b][0] += 1
            bins[b][1] += p[i]
            bins[b][2] += int(k == i)
    tot = max(sum(v[0] for v in bins.values()), 1)
    ece = sum(v[0] * abs(v[1] - v[2]) / max(v[0], 1) for v in bins.values()) / tot
    return {
        "shift": round(shift, 3),
        "draw_mult": round(mult, 3),
        "n": n,
        "rps": round(s_rps / n, 5),
        "logloss": round(s_ll / n, 5),
        "ece": round(ece, 5),
        "accuracy": round(hit / n, 4),
        "draw_argmax_share": round(draw_argmax / n, 4),
        "per_season_rps": {str(s): round(v[1] / v[0], 5) for s, v in sorted(per_sea.items())},
    }


def main() -> None:
    rows = load()
    samples = frozen_probs(rows)
    act_draw = sum(1 for s in samples if s["k"] == 1) / len(samples)
    mean_pd = sum(s["p"][1] for s in samples) / len(samples)
    print(f"五大聯賽 {len(rows)} 場；評分季 ≥{EVAL_FROM} 共 {len(samples)} 場")
    print(f"實際和局率 {act_draw:.4f}｜平均 P_D {mean_pd:.4f}")

    strategies = [("A argmax（現行）", lambda p: max(range(3), key=lambda i: p[i]))]
    for tau in (0.02, 0.03, 0.05, 0.08, 0.12, 0.20):
        strategies.append((f"B 主客距離 <{tau:.2f} 出和",
                           lambda p, t=tau: 1 if abs(p[0] - p[2]) < t
                           else max(range(3), key=lambda i: p[i])))
    for c in (1.1, 1.2, 1.3, 1.5, 1.8, 2.2):
        strategies.append((f"C 和局加權 c={c}",
                           lambda p, c=c: max(range(3), key=lambda i: p[i] * (c if i == 1 else 1))))
    for th in (0.35, 0.40, 0.45, 0.50):
        strategies.append((f"D 弱勢閘 max(P_H,P_A)<{th:.2f} 出和",
                           lambda p, t=th: 1 if max(p[0], p[2]) < t
                           else max(range(3), key=lambda i: p[i])))

    label_rows = []
    base_m = None
    for name, fn in strategies:
        m = label_metrics(samples, fn)
        m["label"] = name
        if base_m is None:
            base_m = m
        m["vs_baseline_acc"] = round(m["accuracy"] - base_m["accuracy"], 4)
        m["beats_baseline_acc"] = m["accuracy"] > base_m["accuracy"]
        m["beats_baseline_macro_f1"] = m["macro_f1"] > base_m["macro_f1"]
        label_rows.append(m)
        print(f"{name:30s} 命中 {m['accuracy']:.4f} 出和比例 {m['pred_share'][1]:.3f} "
              f"和召回 {m['draw_recall']:.3f} 和精確 {m['draw_precision']:.3f} "
              f"macroF1 {m['macro_f1']:.4f} 平衡命中 {m['balanced_acc']:.4f}")

    print("\n第二層：改機率（和局對數機率平移）")
    prob_rows = []
    base_p = prob_metrics(samples, 0.0)
    prob_rows.append(base_p)
    print(f"基準 δ=0            RPS {base_p['rps']:.5f} LL {base_p['logloss']:.5f} "
          f"ECE {base_p['ece']:.5f} 出和 {base_p['draw_argmax_share']:.4f} "
          f"命中 {base_p['accuracy']:.4f}")
    for d in (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60):
        m = prob_metrics(samples, d)
        m["gate_rps"] = m["rps"] <= base_p["rps"]
        m["gate_logloss"] = m["logloss"] <= base_p["logloss"]
        m["gate_ece"] = m["ece"] <= base_p["ece"]
        m["gate_season"] = all(m["per_season_rps"][s] <= base_p["per_season_rps"][s] + 1e-5
                               for s in base_p["per_season_rps"])
        m["gate_pass"] = all((m["gate_rps"], m["gate_logloss"], m["gate_ece"], m["gate_season"]))
        prob_rows.append(m)
        print(f"δ={d:.2f} (×{m['draw_mult']:.2f})  RPS {m['rps']:.5f} LL {m['logloss']:.5f} "
              f"ECE {m['ece']:.5f} 出和 {m['draw_argmax_share']:.4f} 命中 {m['accuracy']:.4f} "
              f"→ 主閘 {sum((m['gate_rps'], m['gate_logloss'], m['gate_ece']))}/3 "
              f"逐季 {'過' if m['gate_season'] else '不過'} → {'過' if m['gate_pass'] else '不過'}")

    out = {
        "note": "S26b 回測：解決「永遠唔預測和局」；λ 層鎖死＝現行生產軌，研究軌，未升指紋",
        "eval_from": EVAL_FROM,
        "n_eval": len(samples),
        "actual_draw_rate": round(act_draw, 4),
        "mean_p_draw": round(mean_pd, 4),
        "layer1_label_only": label_rows,
        "layer2_probability_shift": prob_rows,
    }
    json.dump(out, open("/tmp/bt/s26b_draw_label.json", "w"), ensure_ascii=False, indent=1)
    print("寫入 /tmp/bt/s26b_draw_label.json")


if __name__ == "__main__":
    main()
