"""LightGBM 폐업 예측 모델 — 탐지 트랙 (Detection Track)

두 관점 진단 아키텍처:
  관점 1 (본 모듈): LightGBM — lgb_rank(위험 등급) + SHAP 그룹(예측 근거)  CV AUC 0.798
  관점 2 (ews_model.py): EWS — s_int/comp/ext(업종 내 또래 비교)  AUC 0.737
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

    파라미터 출처: 04_ml_baseline.ipynb RandomizedSearchCV (5-Fold, n_iter=40)
    CV AUC: 0.798  |  Lift@5%: 7.3x  |  Temporal AUC: 0.631
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
    """스냅샷에 LightGBM 결과 컬럼을 추가해 반환

    추가 컬럼:
      lgb_prob : CV 예측 확률 (0~1)
      lgb_rank : 전체 점포 내 위험 백분위 (0~100, 높을수록 위험)

    평가 대상 점포(label_col이 유효한 행)에만 적용한다.
    """
    snap = snap.copy()
    valid = snap[label_col].notna()
    probs = cv_predict(snap[valid], label_col=label_col)
    snap.loc[valid, "lgb_prob"] = probs.values
    snap.loc[valid, "lgb_rank"] = probs.rank(pct=True).values * 100
    return snap


def compute_shap_groups(
    model: LGBMClassifier,
    X: pd.DataFrame,
    feat_groups: dict | None = None,
) -> pd.DataFrame:
    """LightGBM SHAP 값을 내부·경쟁·외부 그룹별로 합산해 percentile 반환

    Parameters
    ----------
    model      : 학습 완료된 LGBMClassifier
    X          : 피처 DataFrame (dw_f_* 18개, FEAT_INTERNAL+COMPETITIVE+EXTERNAL 순)
    feat_groups: {"int": slice(0,10), "comp": slice(10,16), "ext": slice(16,18)}
                 None 이면 기본값(18개 피처 기준) 사용

    Returns
    -------
    DataFrame with columns: shap_int_pct, shap_comp_pct, shap_ext_pct (0~100)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap 패키지가 필요합니다: pip install shap")

    if feat_groups is None:
        feat_groups = {"int": slice(0, 10), "comp": slice(10, 16), "ext": slice(16, 18)}

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    sv1 = sv if (hasattr(sv, "ndim") and sv.ndim == 2) else sv[:, :, 1]

    result = pd.DataFrame(index=X.index)
    for name, sl in feat_groups.items():
        raw = sv1[:, sl].sum(axis=1)
        result[f"shap_{name}_pct"] = pd.Series(raw, index=X.index).rank(pct=True).values * 100
    return result
