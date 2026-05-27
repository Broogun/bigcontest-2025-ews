"""LightGBM 폐업 예측 모델 — 탐지 트랙 (Detection Track)

두 트랙 아키텍처:
  Track 1 (본 모듈): LightGBM — 탐지 정확도 최우선 (CV AUC 0.797)
  Track 2 (ews_model.py): EWS 튜닝 — 해석 가능성 최우선 (AUC 0.737)
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from .config import LGB_BEST_PARAMS, DW_PREFIX


def get_ml_features(snap: pd.DataFrame) -> list:
    """스냅샷 DataFrame에서 ML 피처 컬럼(dw_f_*) 목록 반환"""
    return [c for c in snap.columns if c.startswith(DW_PREFIX)]


def build_lgb_model() -> LGBMClassifier:
    """최적 하이퍼파라미터로 LightGBM 분류기 생성

    파라미터 출처: 04_ml_baseline.ipynb RandomizedSearchCV (5-Fold, n_iter=60)
    CV AUC: 0.797  |  Holdout AUC: 0.760  |  Lift@5%: 6.0x
    """
    return LGBMClassifier(**LGB_BEST_PARAMS)


def cv_predict(
    snap: pd.DataFrame,
    label_col: str = "is_closed_obs",
    n_splits: int = 5,
) -> pd.Series:
    """5-Fold Stratified CV 예측 확률을 snap 인덱스에 맞춰 반환

    Parameters
    ----------
    snap : 스냅샷 DataFrame (dw_f_* 피처 + label_col 포함)
    label_col : 폐업 레이블 컬럼명
    n_splits : CV fold 수

    Returns
    -------
    pd.Series — 각 점포의 폐업 예측 확률 (0~1)
    """
    feat_cols = get_ml_features(snap)
    X = snap[feat_cols].values
    y = snap[label_col].values

    model = build_lgb_model()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return pd.Series(probs, index=snap.index, name="lgb_prob")


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, k_pct: float = 0.05) -> dict:
    """AUC-ROC 와 Lift@k% 계산

    Parameters
    ----------
    y_true : 실제 레이블 (0/1)
    y_prob : 예측 확률
    k_pct  : 상위 k% 기준 (기본 5%)

    Returns
    -------
    dict with keys: auc, lift_at_k, k_threshold
    """
    auc = roc_auc_score(y_true, y_prob)
    k = max(1, int(len(y_true) * k_pct))
    top_k_idx = np.argsort(y_prob)[-k:]
    base_rate = y_true.mean()
    lift = y_true[top_k_idx].mean() / base_rate if base_rate > 0 else float("nan")
    return {"auc": auc, "lift_at_k": lift, "k_threshold": k}


def add_lgb_score(snap: pd.DataFrame, label_col: str = "is_closed_obs") -> pd.DataFrame:
    """스냅샷에 LightGBM CV 예측 확률 컬럼(lgb_prob)을 추가해 반환

    평가 대상 점포(label_col이 유효한 행)에만 적용합니다.
    """
    snap = snap.copy()
    valid = snap[label_col].notna()
    probs = cv_predict(snap[valid], label_col=label_col)
    snap.loc[valid, "lgb_prob"] = probs.values
    return snap
