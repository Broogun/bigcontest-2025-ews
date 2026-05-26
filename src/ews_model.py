"""EWS 점수 산출, 등급 분류, 통계 검증"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .config import (
    FEAT_INTERNAL, FEAT_COMPETITIVE, FEAT_EXTERNAL,
    W_INTERNAL, W_COMPETITIVE, W_EXTERNAL,
    PEER_GROUP_COL, RISK_THRESHOLDS,
)


# ── 1. 종합 위험 점수 산출 ────────────────────────────────────────────────────
def compute_ews_score(snapshot: pd.DataFrame,
                      w_int:  float = W_INTERNAL,
                      w_comp: float = W_COMPETITIVE,
                      w_ext:  float = W_EXTERNAL,
                      group_col: str = PEER_GROUP_COL) -> pd.DataFrame:
    """
    rank_f_* 컬럼 → 컴포넌트 평균 → 가중합 → 그룹 내 백분위 순위.

    Parameters
    ----------
    snapshot  : build_snapshot + add_peer_ranks 결과 DataFrame
    w_int/comp/ext : 컴포넌트 가중치 (합 = 1.0)
    group_col : 피어 그룹 기준 컬럼

    Returns
    -------
    snapshot에 s_int / s_comp / s_ext / risk_score / risk_rank 컬럼 추가
    """
    snap = snapshot.copy()

    r_int  = [f"rank_{f}" for f in FEAT_INTERNAL    if f"rank_{f}" in snap.columns]
    r_comp = [f"rank_{f}" for f in FEAT_COMPETITIVE if f"rank_{f}" in snap.columns]
    r_ext  = [f"rank_{f}" for f in FEAT_EXTERNAL    if f"rank_{f}" in snap.columns]

    def _mean(row, cols):
        v = [row[c] for c in cols if pd.notna(row[c])]
        return float(np.mean(v)) if v else np.nan

    snap["s_int"]  = snap.apply(lambda r: _mean(r, r_int),  axis=1)
    snap["s_comp"] = snap.apply(lambda r: _mean(r, r_comp), axis=1)
    snap["s_ext"]  = snap.apply(lambda r: _mean(r, r_ext),  axis=1)

    snap["risk_score"] = (
        snap["s_int"].fillna(snap["s_int"].median())   * w_int  +
        snap["s_comp"].fillna(snap["s_comp"].median()) * w_comp +
        snap["s_ext"].fillna(snap["s_ext"].median())   * w_ext
    )
    snap["risk_rank"] = (
        snap.groupby(group_col)["risk_score"].rank(pct=True) * 100
    )
    return snap


# ── 2. 위험 등급 분류 ─────────────────────────────────────────────────────────
def classify_risk(rank_pct) -> str:
    """백분위 순위 → 위험 / 경고 / 주의 / 정상"""
    if pd.isna(rank_pct):
        return "평가불가"
    for grade, threshold in sorted(RISK_THRESHOLDS.items(), key=lambda x: -x[1]):
        if rank_pct >= threshold:
            return grade
    return "정상"


# ── 3. 통계 검증 ──────────────────────────────────────────────────────────────
def permutation_test(y_true: np.ndarray, y_score: np.ndarray,
                     n_perm: int = 1000, seed: int = 42) -> dict:
    """
    순열 검정.  H0: AUC는 레이블과 무관.
    Returns: {observed_auc, perm_mean, perm_std, p_value, z_score}
    """
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(y_true, y_score)
    perm = np.array([
        roc_auc_score(rng.permutation(y_true), y_score)
        for _ in range(n_perm)
    ])
    p_val = float((perm >= obs).mean())
    return dict(
        observed_auc=obs,
        perm_mean=perm.mean(),
        perm_std=perm.std(),
        p_value=p_val,
        z_score=(obs - perm.mean()) / perm.std(),
    )


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                 n_boot: int = 1000, ci: float = 95,
                 seed: int = 42) -> dict:
    """
    층화 부트스트랩 AUC 신뢰구간.
    Returns: {mean, std, ci_lower, ci_upper}
    """
    rng = np.random.default_rng(seed)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        yb, sb = y_true[idx], y_score[idx]
        if yb.sum() in (0, len(yb)):
            continue
        aucs.append(roc_auc_score(yb, sb))
    aucs = np.array(aucs)
    lo, hi = np.percentile(aucs, [(100 - ci) / 2, 50 + ci / 2])
    return dict(mean=aucs.mean(), std=aucs.std(), ci_lower=lo, ci_upper=hi)


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.05) -> float:
    """상위 k% 고위험군 내 폐업율 / 전체 폐업율"""
    cut  = np.percentile(y_score, (1 - k) * 100)
    mask = y_score >= cut
    base = y_true.mean()
    if base == 0 or mask.sum() == 0:
        return np.nan
    return y_true[mask].mean() / base
