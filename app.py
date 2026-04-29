import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Game Predictor",
    page_icon="🏀",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem !important;
    letter-spacing: 3px;
    color: #F5A623;
}

.stSelectbox label, .stMarkdown p {
    color: #aaaaaa;
    font-size: 0.85rem;
}

div[data-baseweb="select"] > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #f0f0f0 !important;
    border-radius: 8px !important;
}

.predict-btn > button {
    background: linear-gradient(135deg, #F5A623, #e8890a) !important;
    color: #000 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.3rem !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.predict-btn > button:hover {
    opacity: 0.85 !important;
}

.result-box {
    background: #1a1a1a;
    border-radius: 14px;
    padding: 2rem;
    margin-top: 1.5rem;
    border: 1px solid #2a2a2a;
    text-align: center;
}

.result-winner {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5rem;
    color: #F5A623;
    letter-spacing: 2px;
}

.result-sub {
    color: #888;
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1.5rem;
    gap: 1rem;
}

.prob-card {
    flex: 1;
    background: #111;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #2a2a2a;
}

.prob-pct {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    margin-bottom: 0.2rem;
}

.prob-label {
    font-size: 0.75rem;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin-top: 1rem;
}

.stat-item {
    background: #111;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    border: 1px solid #1e1e1e;
}

.stat-key { color: #666; }
.stat-val { color: #f0f0f0; font-weight: 600; }

.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 1.5rem 0;
}

.last-game-note {
    font-size: 0.75rem;
    color: #555;
    text-align: center;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("nba_model.pkl")
    features = joblib.load("nba_features.pkl")
    return model, features

@st.cache_data
def load_data():
    df = pd.read_csv("nba_model_ready_v2.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df

model, FEATURES = load_model()
df = load_data()

# Team name mapping
TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BKN": "Brooklyn Nets", "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "LA Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"
}

all_teams = sorted(df["HOME_TEAM"].unique().tolist())
team_options = [f"{TEAM_NAMES.get(t, t)} ({t})" for t in all_teams]
abbr_map = {f"{TEAM_NAMES.get(t, t)} ({t})": t for t in all_teams}

def get_latest_stats(team, role):
    """Lấy chỉ số trận gần nhất của đội theo vai trò home/away"""
    prefix = "HOME_" if role == "home" else "AWAY_"
    col = "HOME_TEAM" if role == "home" else "AWAY_TEAM"
    rows = df[df[col] == team].sort_values("GAME_DATE", ascending=False)
    if rows.empty:
        return None, None
    latest = rows.iloc[0]
    stats = {
        "EMA_PTS": latest[f"{prefix}EMA_PTS"],
        "EMA_FG_PCT": latest[f"{prefix}EMA_FG_PCT"],
        "EMA_FG3_PCT": latest[f"{prefix}EMA_FG3_PCT"],
        "WIN_PCT": latest[f"{prefix}CURRENT_WIN_PCT"],
        "WIN_STREAK": latest[f"{prefix}WIN_STREAK"],
        "ELO": latest[f"{prefix}ELO"],
        "REST_DAYS": latest[f"{prefix}REST_DAYS"],
    }
    return stats, latest["GAME_DATE"].strftime("%d/%m/%Y")

def build_features(home_team, away_team):
    """Tính DIFF features từ trận gần nhất của 2 đội"""
    # Lấy hàng gần nhất của home team khi đá sân nhà
    home_rows = df[df["HOME_TEAM"] == home_team].sort_values("GAME_DATE", ascending=False)
    away_rows = df[df["AWAY_TEAM"] == away_team].sort_values("GAME_DATE", ascending=False)

    if home_rows.empty or away_rows.empty:
        return None

    h = home_rows.iloc[0]
    a = away_rows.iloc[0]

    row = {
        "DIFF_PTS":        h["HOME_EMA_PTS"]          - a["AWAY_EMA_PTS"],
        "DIFF_FG_PCT":     h["HOME_EMA_FG_PCT"]        - a["AWAY_EMA_FG_PCT"],
        "DIFF_FG3_PCT":    h["HOME_EMA_FG3_PCT"]       - a["AWAY_EMA_FG3_PCT"],
        "DIFF_FT_PCT":     h["HOME_EMA_FT_PCT"]        - a["AWAY_EMA_FT_PCT"],
        "DIFF_OREB":       h["HOME_EMA_OREB"]          - a["AWAY_EMA_OREB"],
        "DIFF_DREB":       h["HOME_EMA_DREB"]          - a["AWAY_EMA_DREB"],
        "DIFF_AST":        h["HOME_EMA_AST"]           - a["AWAY_EMA_AST"],
        "DIFF_STL":        h["HOME_EMA_STL"]           - a["AWAY_EMA_STL"],
        "DIFF_BLK":        h["HOME_EMA_BLK"]           - a["AWAY_EMA_BLK"],
        "DIFF_TOV":        h["HOME_EMA_TOV"]           - a["AWAY_EMA_TOV"],
        "DIFF_WIN_PCT":    h["HOME_CURRENT_WIN_PCT"]   - a["AWAY_CURRENT_WIN_PCT"],
        "DIFF_WIN_STREAK": h["HOME_WIN_STREAK"]        - a["AWAY_WIN_STREAK"],
        "DIFF_REST_DAYS":  h["HOME_REST_DAYS"]         - a["AWAY_REST_DAYS"],
        "DIFF_ELO":        h["HOME_ELO"]               - a["AWAY_ELO"],
        "DIFF_eFG_PCT":    h["HOME_EMA_eFG_PCT"]       - a["AWAY_EMA_eFG_PCT"],
        "DIFF_TO_RATIO":   h["HOME_EMA_TO_RATIO"]      - a["AWAY_EMA_TO_RATIO"],
        "DIFF_FT_RATE":    h["HOME_EMA_FT_RATE"]       - a["AWAY_EMA_FT_RATE"],
        "DIFF_OREB_PCT":   h["HOME_OREB_PCT"]          - a["AWAY_OREB_PCT"],
        "HOME_IS_B2B":     h["HOME_IS_B2B"],
        "AWAY_IS_B2B":     a["AWAY_IS_B2B"],
    }
    return pd.DataFrame([row])[FEATURES]

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("<h1>🏀 NBA PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#555; margin-top:-1rem; margin-bottom:2rem;'>Dự đoán kết quả trận đấu dựa trên chỉ số EMA gần nhất</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**🏠 Đội Nhà (Home)**")
    home_sel = st.selectbox("Home", team_options, index=team_options.index("LA Lakers (LAL)"), label_visibility="collapsed")

with col2:
    st.markdown("**✈️ Đội Khách (Away)**")
    default_away = team_options.index("Golden State Warriors (GSW)")
    away_sel = st.selectbox("Away", team_options, index=default_away, label_visibility="collapsed")

home_abbr = abbr_map[home_sel]
away_abbr = abbr_map[away_sel]

st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
predict = st.button("DỰ ĐOÁN KẾT QUẢ")
st.markdown("</div>", unsafe_allow_html=True)

if predict:
    if home_abbr == away_abbr:
        st.warning("Vui lòng chọn 2 đội khác nhau!")
    else:
        X = build_features(home_abbr, away_abbr)
        if X is None:
            st.error("Không đủ dữ liệu để dự đoán.")
        else:
            proba = model.predict_proba(X)[0]
            home_prob = proba[1]
            away_prob = proba[0]
            winner = home_abbr if home_prob >= 0.5 else away_abbr
            winner_name = TEAM_NAMES.get(winner, winner)
            winner_label = "🏠 Home Win" if home_prob >= 0.5 else "✈️ Away Win"

            home_stats, home_date = get_latest_stats(home_abbr, "home")
            away_stats, away_date = get_latest_stats(away_abbr, "away")

            st.markdown(f"""
            <div class='result-box'>
                <div class='result-sub'>Dự đoán thắng</div>
                <div class='result-winner'>{winner_label} — {winner_name}</div>
                <div class='prob-row'>
                    <div class='prob-card'>
                        <div class='prob-pct' style='color:#F5A623'>{home_prob:.1%}</div>
                        <div class='prob-label'>{TEAM_NAMES.get(home_abbr, home_abbr)}</div>
                        <div class='prob-label' style='font-size:0.65rem'>HOME</div>
                    </div>
                    <div style='color:#444; font-size:1.5rem; font-weight:bold;'>VS</div>
                    <div class='prob-card'>
                        <div class='prob-pct' style='color:#5B8CFF'>{away_prob:.1%}</div>
                        <div class='prob-label'>{TEAM_NAMES.get(away_abbr, away_abbr)}</div>
                        <div class='prob-label' style='font-size:0.65rem'>AWAY</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if home_stats and away_stats:
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                st.markdown("<p style='color:#555; font-size:0.8rem; text-align:center;'>CHỈ SỐ GẦN NHẤT</p>", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"<p style='color:#F5A623; font-weight:600; margin-bottom:0.3rem;'>{TEAM_NAMES.get(home_abbr)} 🏠</p>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='stat-grid'>
                        <div class='stat-item'><span class='stat-key'>PTS</span><span class='stat-val'>{home_stats['EMA_PTS']:.1f}</span></div>
                        <div class='stat-item'><span class='stat-key'>FG%</span><span class='stat-val'>{home_stats['EMA_FG_PCT']:.1%}</span></div>
                        <div class='stat-item'><span class='stat-key'>3P%</span><span class='stat-val'>{home_stats['EMA_FG3_PCT']:.1%}</span></div>
                        <div class='stat-item'><span class='stat-key'>WIN%</span><span class='stat-val'>{home_stats['WIN_PCT']:.1%}</span></div>
                        <div class='stat-item'><span class='stat-key'>ELO</span><span class='stat-val'>{home_stats['ELO']:.0f}</span></div>
                        <div class='stat-item'><span class='stat-key'>STREAK</span><span class='stat-val'>{int(home_stats['WIN_STREAK'])}</span></div>
                    </div>
                    <p class='last-game-note'>Trận gần nhất: {home_date}</p>
                    """, unsafe_allow_html=True)

                with c2:
                    st.markdown(f"<p style='color:#5B8CFF; font-weight:600; margin-bottom:0.3rem;'>{TEAM_NAMES.get(away_abbr)} ✈️</p>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='stat-grid'>
                        <div class='stat-item'><span class='stat-key'>PTS</span><span class='stat-val'>{away_stats['EMA_PTS']:.1f}</span></div>
                        <div class='stat-item'><span class='stat-key'>FG%</span><span class='stat-val'>{away_stats['EMA_FG_PCT']:.1%}</span></div>
                        <div class='stat-item'><span class='stat-key'>3P%</span><span class='stat-val'>{away_stats['EMA_FG3_PCT']:.1%}</span></div>
                        <div class='stat-item'><span class='stat-key'>WIN%</span><span class='stat-val'>{away_stats['WIN_PCT']:.1%}</span></div>
                        <div class='stat-item'><span class='stat-key'>ELO</span><span class='stat-val'>{away_stats['ELO']:.0f}</span></div>
                        <div class='stat-item'><span class='stat-key'>STREAK</span><span class='stat-val'>{int(away_stats['WIN_STREAK'])}</span></div>
                    </div>
                    <p class='last-game-note'>Trận gần nhất: {away_date}</p>
                    """, unsafe_allow_html=True)