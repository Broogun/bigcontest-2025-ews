"""원본 데이터 로드 및 전처리 함수"""
import numpy as np
import pandas as pd
from .config import (
    SENTINEL, SENTINEL_INT, OBS_START, OBS_END, BUCKET_COLS,
)


def load_master(path: str) -> pd.DataFrame:
    """dataset1 (점포 마스터) 로드 및 전처리"""
    master = pd.read_csv(path, encoding="cp949")

    for col in master.select_dtypes("float").columns:
        master[col] = master[col].replace(SENTINEL, np.nan)

    master["ARE_D_dt"] = pd.to_datetime(
        master["ARE_D"].astype(str), errors="coerce"
    )
    master["MCT_ME_D_dt"] = master["MCT_ME_D"].apply(
        lambda x: pd.Timestamp(str(int(x))) if pd.notna(x) else pd.NaT
    )

    # 관측 구간 내 폐업 레이블
    master["is_closed_obs"] = (
        master["MCT_ME_D_dt"].notna()
        & master["MCT_ME_D_dt"].between(OBS_START, OBS_END)
    ).astype(int)
    master["is_closed_all"] = master["MCT_ME_D_dt"].notna().astype(int)
    return master


def load_sales(path: str) -> pd.DataFrame:
    """dataset2 (월별 매출) 로드 및 전처리
    버킷 컬럼: '5_75-90%' 형식 → 정수 1-6
    """
    sales = pd.read_csv(path, encoding="cp949")

    for col in BUCKET_COLS:
        if col in sales.columns:
            sales[col] = (
                sales[col].astype(str).str.extract(r"^(\d+)")[0].astype(float)
            )

    for col in sales.select_dtypes("float").columns:
        sales[col] = sales[col].replace(SENTINEL, np.nan)

    return sales


def load_customer(path: str) -> pd.DataFrame:
    """dataset3 (월별 고객) 로드 및 전처리
    sentinel: float -999999.9 / int -999999 모두 처리
    """
    cust = pd.read_csv(path, encoding="cp949")

    for col in cust.columns:
        if cust[col].dtype == float:
            cust[col] = cust[col].replace(SENTINEL, np.nan)
        elif cust[col].dtype in ("int64", "int32"):
            mask = cust[col] == SENTINEL_INT
            if mask.any():
                cust[col] = cust[col].astype(float)
                cust.loc[mask, col] = np.nan

    return cust


def build_panel(master: pd.DataFrame,
                sales: pd.DataFrame,
                cust: pd.DataFrame) -> pd.DataFrame:
    """매출 + 고객 outer join → 점포 마스터 left join"""
    panel = pd.merge(sales, cust, on=["ENCODED_MCT", "TA_YM"], how="outer")

    panel["TA_YM_dt"] = pd.to_datetime(
        panel["TA_YM"].apply(lambda x: str(int(x)) if pd.notna(x) else None),
        format="%Y%m",
        errors="coerce",
    )

    master_cols = [
        "ENCODED_MCT", "MCT_NM", "MCT_BSE_AR",
        "HPSN_MCT_ZCD_NM", "HPSN_MCT_BZN_CD_NM", "HPSN_MCT_ZCD_NM_1",
        "ARE_D_dt", "MCT_ME_D_dt", "is_closed_obs", "is_closed_all",
    ]
    panel = pd.merge(panel, master[master_cols], on="ENCODED_MCT", how="left")

    panel["tenure_months"] = (
        (panel["TA_YM_dt"] - panel["ARE_D_dt"]) / np.timedelta64(1, "M")
    ).round().astype("Int64")

    panel["months_to_close"] = (
        (panel["MCT_ME_D_dt"] - panel["TA_YM_dt"]) / np.timedelta64(1, "M")
    ).round().astype("Int64")

    return panel
