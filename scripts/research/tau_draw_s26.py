#!/usr/bin/env python3
"""S26 研究：近盤場（|P_H − P_A| 細）實際和局率量表。

只量、唔改：讀已鎖凍結帳（prediction_log），逐場只用凍結欄，禁回測、禁 live 重算。
問題：|P_H − P_A| < τ 嘅場，實際和局比例有冇同時
  (a) 高過實際主勝比例，同 (b) 高過該批場嘅平均 P_D。
兩條都成立先有資格考慮把「近盤 → 出和」寫成標籤規則；
未過閘一律唔入產品、唔改指紋、唔改矩陣。

用法：python3 tau_draw_s26.py <2026-09.json ...> --out result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TAUS = [0.03, 0.05, 0.08, 0.12]
BIG5 = {"E0", "D1", "SP1", "I1", "F1"}
IDX = {"home": 0, "draw": 1, "away": 2, "H": 0, "D": 1, "A": 2}


def load(paths: list[str]) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        matches = doc.get("matches", doc) if isinstance(doc, dict) else doc
        rows = matches.values() if isinstance(matches, dict) else matches
        for r in rows:
            res = r.get("result") or {}
            p3 = r.get("p")
            ftr = res.get("ftr")
            if not (isinstance(p3, list) and len(p3) == 3 and ftr in IDX):
                continue  # 未完場或無凍結三格 → 唔入表
            if not r.get("locked_at"):
                continue  # 未鎖唔入表
            out.append(r)
    return out


def rps(p: list[float], actual: int) -> float:
    c = 0.0
    s = 0.0
    for i in range(2):
        c += p[i]
        s += (c - (1.0 if actual <= i else 0.0)) ** 2
    return s / 2.0


def bucket(rows: list[dict], tau: float | None, lo: float | None = None) -> dict:
    sel = []
    for r in rows:
        p = r["p"]
        gap = abs(p[0] - p[2])
        if tau is not None and gap >= tau:
            continue
        if lo is not None and gap < lo:
            continue
        sel.append(r)
    n = len(sel)
    if n == 0:
        return {"n": 0}
    act = [IDX[r["result"]["ftr"]] for r in sel]
    draw_rate = sum(1 for a in act if a == 1) / n
    home_rate = sum(1 for a in act if a == 0) / n
    away_rate = sum(1 for a in act if a == 2) / n
    mean_pd = sum(r["p"][1] for r in sel) / n
    mean_ph = sum(r["p"][0] for r in sel) / n
    hit_argmax = sum(1 for r, a in zip(sel, act) if max(range(3), key=lambda i: r["p"][i]) == a) / n
    hit_forced_draw = draw_rate  # 硬出和嘅命中率就係實際和局率
    mean_rps = sum(rps(r["p"], a) for r, a in zip(sel, act)) / n
    # 硬出和唔改機率，所以 RPS 不變；比較嘅係「報咗一個唔係最高機率嘅結果」
    return {
        "n": n,
        "actual_draw": round(draw_rate, 4),
        "actual_home": round(home_rate, 4),
        "actual_away": round(away_rate, 4),
        "mean_p_draw": round(mean_pd, 4),
        "mean_p_home": round(mean_ph, 4),
        "hit_argmax": round(hit_argmax, 4),
        "hit_forced_draw": round(hit_forced_draw, 4),
        "mean_rps": round(mean_rps, 5),
        "gate_a_draw_gt_home": draw_rate > home_rate,
        "gate_b_draw_gt_mean_pd": draw_rate > mean_pd,
        "gate_c_forced_beats_argmax": hit_forced_draw > hit_argmax,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="tau_draw_result.json")
    args = ap.parse_args()

    rows = load(args.logs)
    big5 = [r for r in rows if r.get("div") in BIG5]

    def table(rs: list[dict]) -> dict:
        t: dict = {"all": bucket(rs, None)}
        for tau in TAUS:
            t[f"gap_lt_{tau}"] = bucket(rs, tau)
        t["gap_ge_0.12"] = bucket(rs, None, lo=0.12)
        return t

    out = {
        "spec": "S26 近盤和局量表（research_only）",
        "source": "已鎖凍結帳，只讀凍結欄；禁回測、禁 live 重算",
        "n_settled_locked": len(rows),
        "gates": {
            "a": "實際和局率 > 實際主勝率",
            "b": "實際和局率 > 該批平均 P_D",
            "c": "硬出和命中率 > argmax 命中率",
        },
        "verdict_rule": "三閘任一唔過 → 唔寫入產品、唔改指紋；樣本 < 200 場只出表唔定規則",
        "all_divs": table(rows),
        "big5_only": table(big5),
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
