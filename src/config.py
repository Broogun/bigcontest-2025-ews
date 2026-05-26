"""프로젝트 전역 설정 및 상수"""
import os

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# 원본 파일명 (data/ 에 위치, .gitignore 로 제외)
RAW_MASTER = "big_data_set1_f.csv"
RAW_SALES  = "big_data_set2_f.csv"
RAW_CUST   = "big_data_set3_f.csv"

# 중간 결과 파일명
OUT_PANEL    = "p_project_data_v2.csv"
OUT_FEATURES = "p_project_features.csv"
OUT_SNAPSHOT = "p_project_snapshot_tuned.csv"
OUT_REPORTS  = "p_project_reports.csv"

# ── 전처리 파라미터 ───────────────────────────────────────────────────────────
SENTINEL     = -999999.9
SENTINEL_INT = -999999
OBS_START    = "2023-01-01"
OBS_END      = "2024-12-31"

BUCKET_COLS = [
    "RC_M1_SAA", "RC_M1_TO_UE_CT", "RC_M1_UE_CUS_CN",
    "RC_M1_AV_NP_AT", "MCT_OPE_MS_CN", "APV_CE_RAT",
]

# ── EWS 모델 파라미터 ─────────────────────────────────────────────────────────
LAMBDA_OPT    = 0.75   # 최적 시간감쇠 파라미터 (STAGE 2 grid search)
W_INTERNAL    = 0.65   # 내부 신호 가중치    (STAGE 3 최적화)
W_COMPETITIVE = 0.30   # 경쟁 신호 가중치
W_EXTERNAL    = 0.05   # 외부 신호 가중치

# ── 위험 등급 임계값 (백분위) ─────────────────────────────────────────────────
RISK_THRESHOLDS = {"위험": 85, "경고": 65, "주의": 40}

# ── 피처 그룹 ─────────────────────────────────────────────────────────────────
FEAT_INTERNAL = [
    "f_sales_lvl", "f_trx_lvl", "f_spend_lvl",
    "f_sales_trend", "f_trx_trend",
    "f_return_rate", "f_return_trend",
    "f_float_ratio", "f_float_trend", "f_resid_ratio",
]
FEAT_COMPETITIVE = [
    "f_rank_ind", "f_rank_dist",
    "f_rank_ind_trend", "f_rank_dist_trend",
    "f_vs_ind_sales", "f_vs_ind_trend",
]
FEAT_EXTERNAL = ["f_peer_close_ind", "f_peer_close_dist"]
ALL_FEATS = FEAT_INTERNAL + FEAT_COMPETITIVE + FEAT_EXTERNAL

PEER_GROUP_COL = "HPSN_MCT_BZN_CD_NM"   # 상권 기준 피어 그룹
