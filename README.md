# Feature Engineering — NBA Player Longevity Prediction

## Dataset
**nba_players.csv** — 1,340 NBA player records, 22 columns. Raw per-game statistics from player debut seasons. Prediction goal: will this player last 5+ years in the NBA?

**Target variable**: `target_5yrs` — binary (1 = lasted 5+ years, 0 = did not)

## Feature Engineering Workflow

### 1. Target Variable Isolation
`target_5yrs` defined as the dependent variable. Class distribution: ~62% lasted 5+ years / ~38% did not — moderate imbalance noted for downstream modeling.

### 2. Non-Predictive Columns Dropped
- `Unnamed: 0`: auto-generated row index, no signal
- `name`: player identifier — causes **data leakage** (model would memorize names, not learn generalizable patterns from stats)

### 3. Null Value Handling
Zero missing values found. No imputation required. Note: `0` values in percentage columns (`3p`, `ft`) are valid encoded values for players who never attempted those shot types, not missing data.

### 4. Correlation Analysis
Found 10+ feature pairs with |r| > 0.8 (high multicollinearity), including:
- `fgm` / `fga` / `pts` (mathematically linked)
- `oreb` / `dreb` / `reb` (components = total)
- `ftm` / `fta` / `pts` (FT contributes to scoring)
- `3p_made` / `3pa` (makes and attempts)

### 5. Engineered Features (5 new composites)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `pts_per_min` | pts / min | Scoring rate without playing-time bias |
| `per_simple` | (pts+reb+ast+stl+blk - missed_FG - missed_FT - tov) / min | All-around efficiency; replaces 10+ correlated stats |
| `ts_pct` | pts / (2×(fga + 0.44×fta)) | True Shooting %: weights 2pt/3pt/FT correctly |
| `ast_to_ratio` | ast / tov | Playmaking quality (rewards decision-making, not just volume) |
| `def_contrib_per_min` | (stl+blk+dreb) / min | Combined defensive impact per minute |

### 6. Redundant Columns Dropped
`fgm`, `fga`, `ftm`, `fta`, `3p_made`, `3pa`, `oreb`, `dreb` — all now represented more efficiently by percentage stats and engineered composites.

## Final Dataset
- **Shape**: 1,340 rows × 14 columns (13 predictors + 1 target)
- **Missing values**: 0
- **Multicollinearity**: significantly reduced vs. raw dataset
- **Output file**: `nba_features_engineered.csv`

## Environment Setup
```
pip install pandas numpy matplotlib seaborn
```

## Files
- `nba_feature_engineering.ipynb` — full notebook, all cells executed
- `nba_players.csv` — raw source dataset
- `nba_features_engineered.csv` — final ML-ready output
- `README.md` — this file
- 
