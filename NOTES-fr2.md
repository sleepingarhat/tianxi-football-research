# F-R2 pi-rating vs Elo softmax（2026-09-27）

生產指紋／S5.1 權重／market_beta：**零改**。

## 口徑

- 資料：football-data.co.uk 五大 2014–26，熱身 40 場
- 基準：生產 elo_s2（HFA 60、K0 22、REG 0.70、DRAW0/1 = 0.30/0.18）
- 試驗：Constantinou pi-rating
- 超參只睇 2018–20；2021–25 封死
- 鎖定：λ=0.02 γ=1.0 scale=0.6
- 無賠率、無 xG、無 LGB／DC 重訓

## 2021–25（n=8090）

| | RPS | log-loss | ECE |
|---|---|---|---|
| Elo softmax | 0.20608 | 1.00554 | 0.02368 |
| pi 鎖定組 | 0.20377 | 0.99846 | 0.00388 |

逐季 RPS 全優於 Elo。

## 裁決

**評分軌 PASS_TRACK。生產唔升。**
要入產必須移植 `ens_s5.py` 三軌集成再過三主閘＋三副閘＋逐季。
