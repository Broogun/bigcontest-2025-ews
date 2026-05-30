import os
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── 0. 설정 ────────────────────────────────────────────────────────────────────
# 두 관점 진단 시스템:
#   관점 1 (탐지 · 근거): lgb_rank + shap_int/comp/ext_pct  ← lgb_predictions.csv
#   관점 2 (또래 비교):   s_int / s_comp / s_ext             ← snapshot_tuned.csv
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "p_project_snapshot_tuned.csv")
LGB_PATH  = os.path.join(os.path.dirname(__file__), "..", "outputs", "lgb_predictions.csv")

# API 키: Streamlit Cloud secrets → 환경변수 순으로 로드
def _get_api_key() -> str:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.environ.get("OPENAI_API_KEY", "")

# ── 1. 위험 등급 분류 ──────────────────────────────────────────────────────────
GRADE_META = {
    "위험":   {"color": "#EF4444", "bg": "#FEE2E2", "emoji": "🔴", "action": "즉시 지원 신청 권장"},
    "경고":   {"color": "#F59E0B", "bg": "#FEF3C7", "emoji": "🟡", "action": "2주 내 사전 검토 권장"},
    "주의":   {"color": "#3B82F6", "bg": "#DBEAFE", "emoji": "🔵", "action": "정기 모니터링 시작"},
    "정상":   {"color": "#10B981", "bg": "#D1FAE5", "emoji": "🟢", "action": "현 수준 유지, 성장 자금 검토"},
    "평가불가": {"color": "#9CA3AF", "bg": "#F1F5F9", "emoji": "⚪", "action": "데이터 부족"},
}

def classify(rank_pct):
    if pd.isna(rank_pct): return "평가불가"
    if rank_pct >= 85:    return "위험"
    if rank_pct >= 65:    return "경고"
    if rank_pct >= 40:    return "주의"
    return "정상"

# ── 2. 맞춤 금융상품 매핑 ──────────────────────────────────────────────────────
PRODUCTS = {
    "위험": [
        dict(name="긴급경영안정자금",   org="소상공인시장진흥공단", rate="연 2.0%",       limit="최대 3,000만원", note="즉시 신청 · 별도 담보 없음"),
        dict(name="위기극복 특별보증",   org="서울신용보증재단",     rate="보증료 0.5%",   limit="최대 5,000만원", note="영업일 3일 내 처리"),
        dict(name="폐업예방 경영컨설팅", org="소상공인지원센터",     rate="무료",          limit="연 12시간",      note="담당 컨설턴트 1:1 배정"),
    ],
    "경고": [
        dict(name="소상공인 정책자금",   org="소상공인시장진흥공단", rate="연 3.5%",       limit="최대 7,000만원", note="EWS 경고 등급 우선 심사"),
        dict(name="경영개선 패키지",     org="중소벤처기업부",       rate="무료",          limit="컨설팅 12시간",  note="매출·비용 구조 분석 포함"),
        dict(name="매출연동 한도대출",   org="IBK기업은행",          rate="연 4.5%",       limit="월평균 매출 3배", note="별도 담보 불필요"),
    ],
    "주의": [
        dict(name="EWS 연계 금리우대대출", org="하나/우리은행",      rate="우대금리 0.5%p 차감", limit="최대 1억원",  note="EWS 점수 제출 시 즉시 적용"),
        dict(name="경기대응 보증",          org="신용보증기금",       rate="보증료 0.3% 감면",    limit="최대 3억원",  note="사전심사 약 1주일"),
        dict(name="선제적 모니터링 서비스", org="본 시스템 (무료)",   rate="무료",                limit="월 1회 리포트", note="경고 등급 진입 前 알림 발송"),
    ],
    "정상": [
        dict(name="소상공인 성장자금",     org="소상공인시장진흥공단", rate="연 2.0% (우대)", limit="최대 1억원",    note="우수 가맹점 우선 배정"),
        dict(name="스마트상점 전환 지원",  org="중소벤처기업부",       rate="무료",           limit="최대 500만원",  note="키오스크·배달앱 연동 지원"),
        dict(name="혁신성장 보증",         org="기술보증기금",         rate="보증료 0.5%",    limit="최대 5억원",    note="성장 가능성 평가 포함"),
    ],
    "평가불가": [],
}

# ── 3. 피처 한글 라벨 ──────────────────────────────────────────────────────────
FEAT_KO = {
    "rank_f_sales_lvl":       "매출 수준",
    "rank_f_trx_lvl":         "거래 건수",
    "rank_f_spend_lvl":       "객단가",
    "rank_f_sales_trend":     "매출 추세 (3개월)",
    "rank_f_trx_trend":       "거래 추세 (3개월)",
    "rank_f_return_rate":     "재방문율",
    "rank_f_return_trend":    "재방문 추세",
    "rank_f_float_ratio":     "유동 고객 비중",
    "rank_f_float_trend":     "유동 고객 추세",
    "rank_f_resid_ratio":     "주거 고객 비중",
    "rank_f_rank_ind":        "업종 내 상대 순위",
    "rank_f_rank_dist":       "상권 내 상대 순위",
    "rank_f_rank_ind_trend":  "업종 순위 추세",
    "rank_f_rank_dist_trend": "상권 순위 추세",
    "rank_f_vs_ind_sales":    "업종 대비 매출",
    "rank_f_vs_ind_trend":    "업종 대비 추세",
    "rank_f_peer_close_ind":  "업종 폐업 밀도",
    "rank_f_peer_close_dist": "상권 폐업 밀도",
}
F_INT  = ["rank_f_sales_lvl","rank_f_trx_lvl","rank_f_spend_lvl",
          "rank_f_sales_trend","rank_f_trx_trend",
          "rank_f_return_rate","rank_f_return_trend",
          "rank_f_float_ratio","rank_f_float_trend","rank_f_resid_ratio"]
F_COMP = ["rank_f_rank_ind","rank_f_rank_dist",
          "rank_f_rank_ind_trend","rank_f_rank_dist_trend",
          "rank_f_vs_ind_sales","rank_f_vs_ind_trend"]
F_EXT  = ["rank_f_peer_close_ind","rank_f_peer_close_dist"]

RADAR_FEATS = [
    "rank_f_sales_trend", "rank_f_trx_trend", "rank_f_return_rate",
    "rank_f_vs_ind_sales", "rank_f_rank_ind", "rank_f_peer_close_ind",
]

def hex_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── 4. 페이지 설정 & CSS ───────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="성동구 EWS Dashboard", page_icon="🔔")

st.markdown("""
<style>
/* ── CSS 변수: 라이트 모드 기본값 ──────────────────────────── */
:root {
    --bg-app:        #F1F5F9;
    --bg-card:       #FFFFFF;
    --bg-subtle:     #F8FAFC;
    --bg-bar:        #F1F5F9;
    --border:        #E2E8F0;
    --text-primary:  #0F172A;
    --text-heading:  #1E293B;
    --text-muted:    #64748B;
    --text-faint:    #94A3B8;
    --text-body:     #334155;
    --text-detail:   #475569;
    --badge-rate-bg: #EFF6FF;
    --badge-rate-fg: #2563EB;
    --badge-lim-bg:  #F0FDF4;
    --badge-lim-fg:  #16A34A;
}
/* ── CSS 변수: 다크 모드 오버라이드 ────────────────────────── */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-app:        #0F172A;
        --bg-card:       #1E293B;
        --bg-subtle:     #243047;
        --bg-bar:        #1E293B;
        --border:        #334155;
        --text-primary:  #F1F5F9;
        --text-heading:  #E2E8F0;
        --text-muted:    #94A3B8;
        --text-faint:    #64748B;
        --text-body:     #CBD5E1;
        --text-detail:   #94A3B8;
        --badge-rate-bg: #1E3A5F;
        --badge-rate-fg: #93C5FD;
        --badge-lim-bg:  #14532D;
        --badge-lim-fg:  #86EFAC;
    }
}
/* ── 앱 전체 ────────────────────────────────────────────────── */
.stApp { background-color: var(--bg-app); }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card); border-radius: 14px;
    border: 1.5px solid var(--border);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    padding: 20px; margin-bottom: 16px;
}
h1 { color: var(--text-primary); font-weight: 800; }
h2 { color: var(--text-heading); font-weight: 700; }
h3 { color: var(--text-heading); font-weight: 700; }
.risk-badge {
    display: inline-block; padding: 5px 18px;
    border-radius: 30px; font-weight: 800; font-size: 1.4rem;
}
.metric-label { color: var(--text-muted); font-size: 0.82rem; font-weight: 700; text-transform: uppercase; }
.metric-value { font-size: 2.2rem; font-weight: 900; line-height: 1.1; }
.stTabs [data-baseweb="tab"] {
    font-size: 1.05rem; font-weight: 700; height: 48px;
    border-radius: 8px; border: 1px solid var(--border); padding: 0 22px;
}
.stTabs [aria-selected="true"] { background-color: #0F172A !important; color: white !important; }
@media (prefers-color-scheme: dark) {
    .stTabs [aria-selected="true"] { background-color: #3B82F6 !important; }
}
div.stButton > button {
    background: #2563EB; color: white; border-radius: 10px;
    font-weight: 700; font-size: 1.05rem; border: none; width: 100%; padding: 13px;
}
div.stButton > button:hover { background: #1D4ED8; }
</style>
""", unsafe_allow_html=True)

# ── 5. 데이터 로드 ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # LightGBM 예측값 + SHAP 그룹 merge (notebook 04 실행 시 생성)
    try:
        lgb_raw  = pd.read_csv(LGB_PATH)
        shap_cols = [c for c in ["shap_int_pct", "shap_comp_pct", "shap_ext_pct"] if c in lgb_raw.columns]
        merge_cols = ["ENCODED_MCT", "lgb_prob", "lgb_rank"] + shap_cols
        df = df.merge(lgb_raw[merge_cols], on="ENCODED_MCT", how="left")
        df["grade_rank"] = df["lgb_rank"].combine_first(df["risk_rank_opt"])
        df["using_lgb"]  = df["lgb_rank"].notna()
    except FileNotFoundError:
        df["grade_rank"] = df["risk_rank_opt"]
        df["using_lgb"]  = False

    df["grade"] = df["grade_rank"].apply(classify)
    df["display_name"] = (
        df["MCT_NM"].fillna("").astype(str) + "  |  "
        + df["HPSN_MCT_ZCD_NM"].fillna("").astype(str) + "  "
        + df["HPSN_MCT_BZN_CD_NM"].fillna("").astype(str)
        + "  #" + df["ENCODED_MCT"].astype(str).str[-4:]
    )
    return df[df["risk_rank_opt"].notna()].copy()   # 평가 가능한 점포만

df = load_data()
rank_cols = [c for c in df.columns if c.startswith("rank_f_")]
has_shap  = all(c in df.columns for c in ["shap_int_pct", "shap_comp_pct", "shap_ext_pct"])

# ── 6. 사이드바 ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔔 성동구 EWS")
    st.caption("영세/중소 요식 가맹점 경영위기 조기경보 시스템")
    st.divider()

    grade_order = ["위험", "경고", "주의", "정상"]
    grade_filter = st.multiselect(
        "위험 등급 필터",
        options=grade_order,
        default=grade_order,
    )
    search = st.text_input("🔍 점포명 / 상권 검색", placeholder="예: 성수, 한식, 커피...")

    df_flt = df[df["grade"].isin(grade_filter)]
    if search:
        df_flt = df_flt[df_flt["display_name"].str.contains(search, case=False, na=False)]

    if df_flt.empty:
        st.warning("검색 결과가 없습니다.")
        st.stop()

    selected = st.selectbox("점포 선택", df_flt["display_name"].tolist())
    st.divider()

    st.markdown("**전체 등급 현황**")
    total = len(df)
    for g in grade_order:
        cnt = (df["grade"] == g).sum()
        meta = GRADE_META[g]
        st.markdown(
            f"{meta['emoji']} **{g}**: {cnt}개 "
            f"<span style='color:#94A3B8;'>({cnt/total*100:.1f}%)</span>",
            unsafe_allow_html=True,
        )

# ── 7. 선택 점포 데이터 준비 ───────────────────────────────────────────────────
row        = df[df["display_name"] == selected].iloc[0]
grade      = row["grade"]
meta       = GRADE_META[grade]
color      = meta["color"]
grade_rank = float(row["grade_rank"])          # 등급 결정 기준 백분위
using_lgb  = bool(row.get("using_lgb", False)) # LightGBM 사용 여부

# 유효한 rank_f_* 컬럼 + Top 위험 신호
avail_rank = [c for c in rank_cols if c in row.index and pd.notna(row[c])]
top_signals = sorted(avail_rank, key=lambda c: row[c], reverse=True)[:5]

# ── 8. 헤더 ───────────────────────────────────────────────────────────────────
st.title("🔔 성동구 소상공인 경영위기 조기경보 시스템")

h1, h2, h3, h4 = st.columns(4)
h1.metric("점포명",  str(row["MCT_NM"]))
h2.metric("업종",    str(row["HPSN_MCT_ZCD_NM"]))
h3.metric("상권",    str(row["HPSN_MCT_BZN_CD_NM"]))
h4.metric("관측기간", f"{int(row['n_obs_months'])}개월")
st.divider()

tab1, tab2, tab3 = st.tabs(["📋 종합 진단", "📊 신호 분석", "💳 맞춤 금융 서비스"])

# ═══════════════════════════════════════════════════════════════
# TAB 1: 종합 진단
# ═══════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1, 2.3], gap="large")

    # ── 왼쪽: 등급 게이지 ─────────────────────────────────────
    with col_l:
        with st.container(border=True):
            top_pct = max(1, round(100 - grade_rank))
            st.markdown(f"""
            <div style="text-align:center; padding:8px 0;">
                <div class="metric-label" style="margin-bottom:8px;">종합 위기 등급</div>
                <div class="risk-badge"
                     style="background:{color}22; color:{color}; border:2px solid {color};">
                    {meta['emoji']} {grade}
                </div>
                <div style="margin-top:14px; font-size:0.85rem; color:var(--text-muted);">
                    성동구 요식업 전체 기준
                </div>
                <div style="font-size:2.8rem; font-weight:900; color:{color}; line-height:1.1;">
                    상위 {top_pct}<span style="font-size:1rem;">%</span>
                </div>
                <div style="font-size:0.8rem; color:var(--text-faint); margin-top:4px;">
                    4,183개 점포 중 위험도 상위 {top_pct}% 해당
                </div>
            </div>
            """, unsafe_allow_html=True)

            steps = [
                {"range": [0,  40],  "color": "#D1FAE5"},
                {"range": [40, 65],  "color": "#DBEAFE"},
                {"range": [65, 85],  "color": "#FEF3C7"},
                {"range": [85, 100], "color": "#FEE2E2"},
            ]
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=grade_rank,
                number={"suffix": "%ile", "font": {"color": color, "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"size": 10}},
                    "bar":  {"color": color, "thickness": 0.25},
                    "steps": steps,
                    "threshold": {
                        "line": {"color": "#0F172A", "width": 3},
                        "value": row["risk_rank_opt"],
                    },
                },
            ))
            fig_g.update_layout(
                height=180, paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=0),
            )
            st.plotly_chart(fig_g, use_container_width=True)

            # 범례
            st.markdown("""
            <div style="font-size:0.78rem; color:var(--text-muted); line-height:1.8;">
            🟢 정상 &lt;40  🔵 주의 40~65  🟡 경고 65~85  🔴 위험 ≥85
            </div>
            """, unsafe_allow_html=True)

    # ── 오른쪽: 두 관점 카드 + Top 신호 ─────────────────────
    with col_r:

        def _pct_label(val):
            """%ile 숫자 → '상위 X%' 직관적 표현 (높을수록 위험)"""
            if np.isnan(val): return "N/A"
            top = max(1, round(100 - val))
            return f"상위 {top}%"

        def _component_cards(cards):
            cols = st.columns(3)
            for i, (label, score_col, desc) in enumerate(cards):
                s_val  = row.get(score_col, float("nan"))
                s_val  = float(s_val) if pd.notna(s_val) else float("nan")
                s_g    = classify(s_val) if not np.isnan(s_val) else "평가불가"
                s_c    = GRADE_META[s_g]["color"]
                s_disp = _pct_label(s_val)
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"""
                        <div style="text-align:center;">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value" style="color:{s_c};">{s_disp}</div>
                            <div style="font-size:0.78rem; color:var(--text-faint); margin:4px 0;">{desc}</div>
                            <div style="font-size:0.9rem; font-weight:700; color:{s_c};">{GRADE_META[s_g]['emoji']} {s_g}</div>
                        </div>
                        """, unsafe_allow_html=True)

        # ── 관점 1: 위험 판정 이유 ────────────────────────────
        if using_lgb and has_shap:
            st.markdown("### 🔍 위험 판정 이유")
            st.caption("이 점포의 어느 부분이 위험 등급에 영향을 미쳤는가 (전체 점포 기준)")
            _component_cards([
                ("내부 요인",  "shap_int_pct",  "매출 · 거래 · 고객"),
                ("경쟁 요인",  "shap_comp_pct", "업종 · 상권 위치"),
                ("외부 요인",  "shap_ext_pct",  "주변 폐업 밀도"),
            ])
            st.markdown("")

        # ── 관점 2: 같은 업종 점포와 비교 ────────────────────
        st.markdown("### 📊 같은 업종 점포들과 비교")
        st.caption("성동구 동일 업종 · 상권 내 점포들 사이에서 이 점포의 위치")
        _component_cards([
            ("내부 지표",  "s_int",  "매출 · 거래 · 고객"),
            ("경쟁 환경",  "s_comp", "업종 · 상권 내 순위"),
            ("외부 환경",  "s_ext",  "주변 폐업 밀도"),
        ])

        # ── 자동 종합 해석 ────────────────────────────────────
        ews_map = {
            "내부 지표 (매출·거래·고객)": float(row.get("s_int", float("nan"))),
            "경쟁 환경 (업종 내 위치)":   float(row.get("s_comp", float("nan"))),
            "외부 환경 (주변 폐업 밀도)": float(row.get("s_ext", float("nan"))),
        }
        ews_valid = {k: v for k, v in ews_map.items() if not np.isnan(v)}

        if len(ews_valid) >= 2:
            worst_k = max(ews_valid, key=ews_valid.get)
            best_k  = min(ews_valid, key=ews_valid.get)
            worst_g = classify(ews_valid[worst_k])
            best_g  = classify(ews_valid[best_k])
            worst_top = max(1, round(100 - ews_valid[worst_k]))
            best_top  = max(1, round(100 - ews_valid[best_k]))

            if worst_g != best_g:
                if best_g in ("정상", "주의") and worst_g in ("위험", "경고"):
                    badge_color = GRADE_META[worst_g]["color"]
                    safe_color  = GRADE_META[best_g]["color"]
                    extra = ""
                    if best_g == "정상":
                        extra = f" <b style='color:{safe_color};'>{best_k}의 안정세가 상황을 버텨주고 있습니다.</b> 이 부분이 흔들리면 등급이 올라갈 수 있습니다."
                    elif best_g == "주의" and worst_g == "위험":
                        extra = f" 전반적으로 주의가 필요한 상황입니다."
                    insight_html = (
                        f"<div style='background:var(--bg-subtle); border-left:4px solid {badge_color}; "
                        f"padding:12px 15px; border-radius:8px; font-size:0.9rem; color:var(--text-body); line-height:1.7;'>"
                        f"<b style='color:{badge_color};'>{worst_k}</b>가 업종 내 상위 {worst_top}%로 가장 우려됩니다. "
                        f"반면 <b style='color:{safe_color};'>{best_k}</b>는 상위 {best_top}%로 현재 안정적입니다."
                        f"{extra}</div>"
                    )
                    st.markdown(insight_html, unsafe_allow_html=True)
                    st.markdown("")

        # Top 5 위험 신호 수평 막대
        st.markdown("### ⚠️ 가장 우려되는 신호 Top 5")
        if top_signals:
            bar_colors = [GRADE_META[classify(row[c])]["color"] for c in top_signals]
            fig_bar = go.Figure(go.Bar(
                y=[FEAT_KO.get(c, c) for c in top_signals],
                x=[row[c] for c in top_signals],
                orientation="h",
                marker_color=bar_colors,
                text=[f"  상위 {max(1, round(100 - row[c]))}%" for c in top_signals],
                textposition="outside",
                textfont=dict(size=13),
            ))
            fig_bar.add_vline(x=65, line_dash="dash", line_color="#F59E0B", annotation_text="경고선")
            fig_bar.add_vline(x=85, line_dash="dash", line_color="#EF4444", annotation_text="위험선")
            fig_bar.update_layout(
                height=230, margin=dict(l=10, r=80, t=10, b=10),
                xaxis=dict(range=[0, 120], showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
                yaxis=dict(tickfont=dict(size=12)),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: 신호 분석
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.caption("📌 이 탭의 수치는 모두 **성동구 동일 업종·상권 내 상대 순위** 기준입니다. 높을수록 위험.")
    col_radar, col_detail = st.columns([1, 1.2], gap="large")

    with col_radar:
        st.markdown("### 🕸️ 위험 신호 레이더")
        r_labels = [FEAT_KO.get(f, f) for f in RADAR_FEATS]
        r_vals   = [
            float(row[f]) if f in row.index and pd.notna(row[f]) else 50.0
            for f in RADAR_FEATS
        ]
        r_vals_c   = r_vals + [r_vals[0]]
        r_labels_c = r_labels + [r_labels[0]]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=r_vals_c, theta=r_labels_c,
            fill="toself", fillcolor=hex_rgba(color, 0.15),
            line=dict(color=color, width=2.5),
            name="현재 점포",
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=[65] * 7, theta=r_labels_c,
            line=dict(color="#F59E0B", dash="dot", width=1.5),
            name="경고 기준 (상위 35%)",
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=[50] * 7, theta=r_labels_c,
            line=dict(color="#94A3B8", dash="dash", width=1),
            name="업종 평균",
        ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 100], tickfont=dict(size=9)),
                angularaxis=dict(tickfont=dict(size=11)),
            ),
            height=400, showlegend=True,
            legend=dict(font=dict(size=11)),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_detail:
        st.markdown("### 📋 세부 지표 (업종 내 상대 순위)")

        for group_name, feats in [
            ("내부 지표 — 매출·거래·고객", F_INT),
            ("경쟁 환경 — 업종·상권 내 위치", F_COMP),
            ("외부 환경 — 주변 폐업 밀도", F_EXT),
        ]:
            with st.expander(f"**{group_name}**", expanded=(group_name.startswith("내부"))):
                for f in feats:
                    if f not in row.index or pd.isna(row[f]):
                        continue
                    val  = float(row[f])
                    g_f  = classify(val)
                    c_f  = GRADE_META[g_f]["color"]
                    pct  = int(min(val, 100))
                    top_str = f"상위 {max(1, round(100 - val))}%"
                    st.markdown(f"""
                    <div style="margin-bottom:9px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                            <span style="font-size:0.88rem; font-weight:600;">{FEAT_KO.get(f, f)}</span>
                            <span style="font-size:0.82rem; color:{c_f}; font-weight:700;">
                                {top_str} · {g_f}
                            </span>
                        </div>
                        <div style="background:var(--bg-bar); border-radius:4px; height:7px; overflow:hidden;">
                            <div style="width:{pct}%; background:{c_f}; height:100%; border-radius:4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3: 맞춤 금융 서비스
# ═══════════════════════════════════════════════════════════════
with tab3:
    col_prod, col_ai = st.columns([1, 1.2], gap="large")

    # ── 왼쪽: 금융상품 추천 ────────────────────────────────────
    with col_prod:
        st.markdown("### 💳 위험 등급별 맞춤 금융상품")

        st.markdown(f"""
        <div style="background:{color}18; border-left:5px solid {color};
             padding:13px 16px; border-radius:8px; margin-bottom:18px;">
            <b>{meta['emoji']} 현재 등급: {grade}</b> — {meta['action']}
        </div>
        """, unsafe_allow_html=True)

        for p in PRODUCTS.get(grade, []):
            st.markdown(f"""
            <div style="background:var(--bg-subtle); border:1px solid var(--border);
                 border-radius:10px; padding:13px 16px; margin-bottom:10px;">
                <div style="font-weight:800; font-size:1.0rem; color:var(--text-primary);">{p['name']}</div>
                <div style="color:var(--text-muted); font-size:0.82rem; margin:3px 0;">{p['org']}</div>
                <div style="display:flex; gap:10px; margin-top:8px; flex-wrap:wrap;">
                    <span style="background:var(--badge-rate-bg); color:var(--badge-rate-fg); padding:2px 10px;
                          border-radius:20px; font-size:0.8rem; font-weight:600;">
                        💰 {p['rate']}
                    </span>
                    <span style="background:var(--badge-lim-bg); color:var(--badge-lim-fg); padding:2px 10px;
                          border-radius:20px; font-size:0.8rem; font-weight:600;">
                        📋 {p['limit']}
                    </span>
                </div>
                <div style="margin-top:8px; font-size:0.83rem; color:var(--text-detail);">
                    💡 {p['note']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 서비스 플로우
        st.markdown("### 금융 서비스 흐름")
        st.markdown("""
        <div style="background:var(--bg-subtle); padding:15px 18px; border-radius:10px;
             border:1px solid var(--border); font-size:0.88rem; line-height:2.1; color:var(--text-body);">
        📡 <b>데이터 수집</b> (매월 카드사·POS 데이터)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        🔔 <b>위험 등급 산정</b> — 위험 / 경고 / 주의 / 정상<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        📊 <b>AI 리포트</b> — 자동 생성<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        💳 <b>맞춤 금융상품</b> — 자동 매칭 & SMS 알림 발송
        </div>
        """, unsafe_allow_html=True)

    # ── 오른쪽: AI 리포트 ──────────────────────────────────────
    with col_ai:
        st.markdown("### AI 경영 진단 리포트")

        if st.button("✨ AI 리포트 생성 (GPT-4o-mini)"):
            with st.spinner("AI 분석 중 (약 10초)..."):

                top3_desc = "\n".join([
                    f"- {FEAT_KO.get(c, c)}: {row[c]:.1f}%ile"
                    for c in top_signals[:3]
                ])

                rank_src = f"위험 순위 {grade_rank:.1f}%ile"

                shap_line = ""
                if using_lgb and has_shap:
                    shap_line = f"- 모델 예측 근거(SHAP): 내부 {row.get('shap_int_pct', float('nan')):.1f}%ile  |  경쟁 {row.get('shap_comp_pct', float('nan')):.1f}%ile  |  외부 {row.get('shap_ext_pct', float('nan')):.1f}%ile"

                prompt = f"""당신은 소상공인 경영위기 조기경보 시스템(EWS)의 AI 분석관입니다.
이 시스템은 폐업 확률 예측이 아닌 상위 위험군 선별 목적의 순위 기반 조기경보입니다.
과장·단정·공포 조장 표현을 절대 금지합니다.

[점포 정보]
- 업종: {row['HPSN_MCT_ZCD_NM']}
- 상권: {row['HPSN_MCT_BZN_CD_NM']}
- 위험 등급: {grade} ({rank_src})
{shap_line}
- 업종 내 또래 비교(EWS): 내부 {row['s_int']:.1f}%ile  |  경쟁 {row['s_comp']:.1f}%ile  |  외부 {row['s_ext']:.1f}%ile

[가장 우려되는 상위 3개 신호 (업종 내 백분위)]
{top3_desc}

[모델 검증 정보 — 리포트에 직접 인용 금지]
LightGBM CV AUC=0.798, EWS AUC=0.737
Bootstrap 95%CI=[0.655, 0.818], Permutation p<0.001
전향적 검증(2023→2024): AUC=0.611, Lift@5%=2.0x

[출력 형식 — 한국어, 600~900자 순수 텍스트]
1. 진단 요약 (2줄)
2. 관측 (Observed): 내부 신호 기반 bullet 3개 (숫자 포함)
3. 해석 (Hypothesis): 가능한 원인 bullet 2개 (가설로만, 단정 금지)
4. 실행 계획: 2주 / 1달 / 3달 단계별 (각 2개)
5. 권장 금융 서비스: {grade} 등급 기준 1개 제안 (구체적 상품명 포함)"""

                ai_text = None
                try:
                    from openai import OpenAI
                    api_key = _get_api_key()
                    if not api_key:
                        raise ValueError("API 키 없음 — 기본 리포트로 대체합니다")
                    client  = OpenAI(api_key=api_key)
                    msg     = client.chat.completions.create(
                        model="gpt-4o-mini",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    ai_text = msg.choices[0].message.content
                except Exception as e:
                    err_msg = str(e).encode("utf-8", errors="replace").decode("utf-8")
                    first_prod = PRODUCTS[grade][0]["name"] if PRODUCTS.get(grade) else "정책자금"
                    top_ko     = FEAT_KO.get(top_signals[0], top_signals[0]) if top_signals else "매출 추세"
                    ai_text = f"""**[AI 경영 진단 리포트 — {row['HPSN_MCT_BZN_CD_NM']} {row['HPSN_MCT_ZCD_NM']}]**

**1. 진단 요약**
현재 상권 내 {row['risk_rank_opt']:.1f} 백분위로 **'{grade}'** 단계에 해당합니다.
가장 두드러진 신호는 **'{top_ko}'** ({row[top_signals[0]]:.1f}%ile)로 즉각적인 점검이 필요합니다.

**2. 관측 (Observed)**
- 내부 신호 {row['s_int']:.1f}%ile: 동종 업종 대비 내부 경영 지표 {'상위 위험군' if row['s_int']>=65 else '중간 수준'}
- 경쟁 신호 {row['s_comp']:.1f}%ile: 상권·업종 경쟁 구도 {'불리' if row['s_comp']>=65 else '보통'}한 위치
- 외부 신호 {row['s_ext']:.1f}%ile: 주변 폐업 밀도 {'높음 — 상권 위축 가능성' if row['s_ext']>=65 else '평균 수준'}

**3. 해석 (Hypothesis)**
- (가설) 거래 건수 및 재방문율 감소는 주변 경쟁점 증가 또는 계절적 요인과 연관 가능성이 있음
- (가설) 업종 내 상대 순위 하락은 고객 가치 인식 또는 가격 경쟁력 이슈를 시사할 수 있음

**4. 실행 계획**
- 2주: 핵심 메뉴 3개 집중 운영 + 재방문 고객 쿠폰 즉시 발행
- 1달: 객단가 개선 메뉴 구성 + 배달앱 리뷰 키워드 점검
- 3달: 고정비 재구조화 검토 + 정책자금 사전 심사 신청

**5. 권장 금융 서비스**
'{grade}' 등급 기준 **'{first_prod}'** 신청을 우선 검토하세요.

*(기본 리포트 표시 중: {err_msg})*"""

                with st.container(border=True):
                    st.markdown(ai_text)

        else:
            st.info("버튼을 클릭하면 AI가 현재 점포 상황을 분석하고 맞춤 금융상품을 추천합니다.")
            st.markdown(f"""
            <div style="background:var(--bg-subtle); padding:15px; border-radius:10px;
                 border:1px solid var(--border); margin-top:10px; font-size:0.9rem; color:var(--text-detail); line-height:1.9;">
            <b>📝 리포트 구성</b><br>
            &nbsp;① 진단 요약 — 현재 위험 수준 핵심 해석<br>
            &nbsp;② 관측 신호 — 데이터 기반 객관적 수치<br>
            &nbsp;③ 원인 가설 — 경영 환경 분석 (단정 금지)<br>
            &nbsp;④ 단계별 실행 계획 (2주 / 1달 / 3달)<br>
            &nbsp;⑤ 맞춤 금융상품 추천 (<b>{grade}</b> 등급 기준)
            </div>
            """, unsafe_allow_html=True)
