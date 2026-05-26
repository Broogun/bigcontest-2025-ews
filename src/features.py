"""피처 엔지니어링: 추세 계산 + 리스크 신호 정렬 + 시간감쇠 스냅샷"""
import numpy as np
import pandas as pd
from .config import ALL_FEATS, PEER_GROUP_COL


# ── 1. 추세 피처 (3개월 차분) ──────────────────────────────────────────────────
TREND_MAP = {
    "RC_M1_SAA":          "trend_sales",
    "RC_M1_TO_UE_CT":     "trend_trx",
    "RC_M1_AV_NP_AT":     "trend_spend",
    "RC_M1_SAA":          "trend_sales",
    "M1_SME_RY_SAA_RAT":  "trend_vs_ind",
    "MCT_UE_CLN_REU_RAT": "trend_return",
    "RC_M1_SHC_FLP_UE_CLN_RAT": "trend_float",
}


def add_trend_features(panel: pd.DataFrame) -> pd.DataFrame:
    """점포 내 3개월 차분으로 추세 피처 추가"""
    for src, dst in TREND_MAP.items():
        if src in panel.columns and dst not in panel.columns:
            panel[dst] = panel.groupby("ENCODED_MCT")[src].transform(
                lambda x: x.diff(3)
            )
    return panel


# ── 2. 리스크 정렬 피처 (높을수록 위험) ───────────────────────────────────────
def add_risk_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    f_ 피처: 값이 클수록 위험도 높음.
    양의 상관 지표는 그대로, 음의 상관 지표는 부호 반전.
    """
    panel["f_sales_lvl"]      = panel["RC_M1_SAA"]
    panel["f_trx_lvl"]        = panel["RC_M1_TO_UE_CT"]
    panel["f_spend_lvl"]      = panel["RC_M1_AV_NP_AT"]
    panel["f_sales_trend"]    = panel.get("trend_sales")
    panel["f_trx_trend"]      = panel.get("trend_trx")
    panel["f_return_rate"]    = -panel["MCT_UE_CLN_REU_RAT"]
    panel["f_return_trend"]   = -panel.get("trend_return", pd.Series(dtype=float, index=panel.index))
    panel["f_float_ratio"]    = panel.get("RC_M1_SHC_FLP_UE_CLN_RAT")
    panel["f_float_trend"]    = panel.get("trend_float")
    panel["f_resid_ratio"]    = panel.get("RC_M1_SHC_RSD_UE_CLN_RAT")
    panel["f_rank_ind"]       = panel.get("M1_SME_RY_SAA_RAT")   # 낮을수록 위험 → keep
    panel["f_rank_dist"]      = panel.get("M1_SME_RY_CNT_RAT")
    panel["f_rank_ind_trend"] = panel.get("trend_vs_ind")
    panel["f_rank_dist_trend"]= panel.get("trend_vs_ind")
    panel["f_vs_ind_sales"]   = -panel["M1_SME_RY_SAA_RAT"].clip(-70, 500)
    panel["f_vs_ind_trend"]   = -panel.get("trend_vs_ind", pd.Series(dtype=float, index=panel.index))
    return panel


# ── 3. 외부 신호: 업종/상권 폐업 밀도 ────────────────────────────────────────
def add_peer_closure_features(panel: pd.DataFrame) -> pd.DataFrame:
    """각 시점 기준 업종/상권 내 폐업 점포 비율 계산"""
    for group_col, feat_col in [
        ("HPSN_MCT_ZCD_NM",    "f_peer_close_ind"),
        ("HPSN_MCT_BZN_CD_NM", "f_peer_close_dist"),
    ]:
        if group_col not in panel.columns:
            continue
        closed_cnt = (
            panel.groupby(["TA_YM", group_col])["is_closed_obs"]
            .transform("sum")
        )
        total_cnt = (
            panel.groupby(["TA_YM", group_col])["ENCODED_MCT"]
            .transform("count")
        )
        panel[feat_col] = (closed_cnt / total_cnt.replace(0, np.nan)) * 100
    return panel


# ── 4. 시간감쇠 가중 평균 ─────────────────────────────────────────────────────
def decay_wmean(vals: np.ndarray, lam: float) -> float:
    """지수감쇠 가중 평균. 최신 관측값 가중치=1, 과거로 갈수록 lam 배씩 감소."""
    v = vals[~np.isnan(vals)]
    if len(v) == 0:
        return np.nan
    w = np.array([lam ** (len(v) - 1 - i) for i in range(len(v))])
    return float(np.average(v, weights=w))


# ── 5. 점포별 스냅샷 생성 ────────────────────────────────────────────────────
def build_snapshot(panel: pd.DataFrame,
                   feat_cols: list = None,
                   lam: float = 0.75) -> pd.DataFrame:
    """
    점포별로 feat_cols 각각에 decay_wmean 적용.
    결과: 점포 1행 × n 피처 열.
    """
    if feat_cols is None:
        feat_cols = ALL_FEATS

    meta_cols = [PEER_GROUP_COL, "HPSN_MCT_ZCD_NM", "MCT_NM",
                 "is_closed_obs", "is_closed_all"]
    panel_s = panel.sort_values(["ENCODED_MCT", "TA_YM"])
    rows = []
    for mct, grp in panel_s.groupby("ENCODED_MCT"):
        r = {"ENCODED_MCT": mct, "n_obs_months": len(grp)}
        for c in meta_cols:
            if c in grp.columns:
                r[c] = grp[c].iloc[-1]
        for f in feat_cols:
            r[f] = decay_wmean(grp[f].values, lam) if f in grp.columns else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ── 6. 그룹 내 백분위 순위 ────────────────────────────────────────────────────
def add_peer_ranks(snapshot: pd.DataFrame,
                   feat_cols: list = None,
                   group_col: str = PEER_GROUP_COL) -> pd.DataFrame:
    """feat_cols 각각에 대해 그룹 내 백분위 순위 컬럼(rank_f_*) 추가"""
    if feat_cols is None:
        feat_cols = ALL_FEATS
    for f in feat_cols:
        if f in snapshot.columns:
            snapshot[f"rank_{f}"] = (
                snapshot.groupby(group_col)[f].rank(pct=True) * 100
            )
    return snapshot
