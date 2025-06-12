import streamlit as st
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# PAGE CONFIG
st.set_page_config(
    page_title="Tennis Match Predictor 🎾",
    page_icon="🎾",
    layout="centered",
    initial_sidebar_state="auto"
)

# HEADER
st.markdown("""
    <div style="text-align: center;">
        <h1 style='color: #FF4B4B;'>🎾 Tennis Match Winner Predictor</h1>
        <h4 style='color: #444;'>Predict the winner of a tennis match using ML </h4>
    </div>
    <br>
""", unsafe_allow_html=True)

st.info("💡 *Don't know the stats?* Get player ranks, odds, surface win %, and H2H stats from ATP or WTA websites:")

col_atp, col_wta = st.columns(2)

with col_atp:
    st.link_button("🌍 Visit ATP Website", "https://www.atptour.com/")
with col_wta:
    st.link_button("🌍 Visit WTA Website", "https://www.wtatennis.com/")

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    rank_1 = st.number_input("👟 Player 1 Rank", min_value=1, max_value=10000, value=1)
    odd_1 = st.number_input("💰 Player 1 Odds", min_value=1.0, max_value=100.0, value=1.00)
    surface = st.selectbox("🏟️ Surface", ["Clay", "Grass", "Hard", "Carpet"])
    form_win_pct_diff = st.number_input("🔥 Form Win % Diff (P1 - P2)", value=0.0)
with col2:
    rank_2 = st.number_input("👟 Player 2 Rank", min_value=1, max_value=10000, value=1)
    odd_2 = st.number_input("💰 Player 2 Odds", min_value=1.0, max_value=100.0, value=1.00)
    h2h_diff = st.number_input("⚔️ H2H Win Diff (P1 - P2)", value=0.0)
    surface_win_diff = st.number_input("🎾 Surface Win % Diff (P1 - P2)", value=0.0)

is_top10_match = st.checkbox("🔟 Both players in Top 10?", value=False)

# Feature Engineering
rank_diff = rank_1 - rank_2
is_p1_higher = int(rank_1 < rank_2)
rank_ratio = rank_2 / (rank_1 + 1e-5)
odds_ratio = abs(odd_2 / (odd_1 + 1e-5))

surface_encoding = {
    "Clay": [1, 0, 0, 0],
    "Grass": [0, 1, 0, 0],
    "Hard": [0, 0, 1, 0],
    "Carpet": [0, 0, 0, 1]
}
surface_clay, surface_grass, surface_hard, surface_carpet = surface_encoding[surface]

features = [
    odds_ratio,
    rank_diff,
    is_p1_higher,
    rank_ratio,
    surface_win_diff,
    surface_clay,
    surface_grass,
    surface_hard,
    surface_carpet,
    form_win_pct_diff,
    h2h_diff,
    int(is_top10_match)
]

if st.button("📊 Predict Winner"):
    input_scaled = scaler.transform([features])
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]
    st.markdown("---")
    if prediction == 1:
        st.success(f"🎉 **Player 1 will win!** (Confidence: `{prob[1]*100:.2f}%`)")
        if prob[1] > 0.8:
            st.balloons()
    else:
        st.error(f"📉 **Player 2 will win!** (Confidence: `{prob[0]*100:.2f}%`)")
        if prob[0] > 0.8:
            st.snow()

