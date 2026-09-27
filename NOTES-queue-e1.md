# 聯合隊列結案（2026-09-27）

指紋、α、market_beta=0、第二層 τ=0.6／δ=0.1、LambdaRank、CatBoost：**零改**。
本倉結果唔寫入生產凍結。

## F-R1 動態攻防時間結構 vs S3／固定 lr SGD

已跑：`scripts/research/dynamic_ad_s25.py` → `data/research/s25/dynamic_ad_result.json`（S25-5）。

基準（固定 lr SGD，n=8603）：RPS 0.20524／ECE 0.01099／格 log-loss 2.9456。

24 組動態 τ₀∈{5,10} × q∈{0,0.0001,0.0005,0.001} × obs∈{0.5,1.0}：**gate_pass 全 false**。
最近一組 `τ₀=5 q=0.0001 obs=1.0` RPS 0.20551（仍差於基準）／ECE 0.01288（輸）／季閘亦輸。
低 q 近乎齊唱 1-1；高 q 校準同大細一齊退。

**裁決：唔採用。列入 NOTES-rejected（時間結構刀已否決）。唔重跑、唔升指紋。**

Owen 動態 DC 若只係同一條「λ 隨機遊走／過程噪」家族，視作 F-R1 子集，唔另開刀。

## F-R2 pi-rating vs Elo softmax

狀態：未跑。協議先寫死先准動手：

- 只替換 Elo 軌（生產權重 0.25），LGB 0.65 同 DC 0.10 鎖死
- 同一張波膽矩陣加總 1X2
- walk-forward 2021–26 五大；三主閘 RPS／實際格 log-loss／ECE 全季優於 S5.1 基準
- 任何一季 RPS 退 → 整刀否決
- 禁止賠率入模、禁止同動態攻防疊加

## F-R3 Δλ 授權名單／密度／xG

結構已有；寫入器強制 `delta=0 applied=false`。
未授權 xG／陣容入凍結。缺資料紅燈、Δλ=0。
**產線維持全 0。**

## 共用 E2

足球完場對照句已由 `explainMatch().settled` 填空（1X2 中／唔中 + 第二層贏／走水／輸 + RPS）。
唔另開模型。賽馬對照句在 `explainRace().settled`，等鎖後 join 名次先顯示。
