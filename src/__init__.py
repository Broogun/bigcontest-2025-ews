"""서울 성동구 요식 가맹점 조기경보 시스템 — src 패키지

두 트랙 아키텍처:
  Track 1 (ml_model): LightGBM — 탐지 정확도 (CV AUC 0.797)
  Track 2 (ews_model): EWS 튜닝 — 해석 가능성 (AUC 0.737)
"""
from .preprocessing import load_master, load_sales, load_customer, build_panel
from .features import build_snapshot
from .ews_model import compute_ews_score, classify_risk
from .ml_model import build_lgb_model, cv_predict, evaluate, add_lgb_score

__all__ = [
    # 전처리
    "load_master",
    "load_sales",
    "load_customer",
    "build_panel",
    # 피처 엔지니어링
    "build_snapshot",
    # EWS (해석 트랙)
    "compute_ews_score",
    "classify_risk",
    # LightGBM (탐지 트랙)
    "build_lgb_model",
    "cv_predict",
    "evaluate",
    "add_lgb_score",
]
