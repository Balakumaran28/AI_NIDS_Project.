import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CyberShield – AI NIDS",
    layout="wide"
)

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Orbitron', sans-serif;
}

.main {
    background-color: #0e1117;
}

h1 {
    color: #00e5ff;
}

h2, h3 {
    color: #90caf9;
}

div[data-testid="metric-container"] {
    background-color: #1e222d;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 0 10px rgba(0,229,255,0.2);
}

.stButton>button {
    background: linear-gradient(90deg, #00e5ff, #2979ff);
    color: black;
    border-radius: 8px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>CyberShield: AI-Based Intrusion Detection System</h1>", unsafe_allow_html=True)
st.markdown("""
<p style="color:#b0bec5;">
Real-time network traffic analysis using Machine Learning to identify
malicious activities and potential intrusions.
</p>
""", unsafe_allow_html=True)

# ---------------- DATA GENERATION ----------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 5000

    data = {
        "Destination_Port": np.random.randint(1, 65535, n_samples),
        "Flow_Duration": np.random.randint(100, 100000, n_samples),
        "Total_Fwd_Packets": np.random.randint(1, 100, n_samples),
        "Packet_Length_Mean": np.random.uniform(10, 1500, n_samples),
        "Active_Mean": np.random.uniform(0, 1000, n_samples),
        "Label": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    attack_rows = df["Label"] == 1
    df.loc[attack_rows, "Total_Fwd_Packets"] += np.random.randint(50, 200, attack_rows.sum())
    df.loc[attack_rows, "Flow_Duration"] = np.random.randint(1, 1000, attack_rows.sum())

    return df

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## Configuration")
split_size = st.sidebar.slider("Training Data (%)", 50, 90, 80)
n_estimators = st.sidebar.slider("Random Forest Trees", 10, 200, 120)

# ---------------- PREPROCESSING ----------------
X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(100 - split_size) / 100, random_state=42
)

# ---------------- TRAINING SECTION ----------------
st.divider()
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model Control")
    if st.button("Train Detection Model"):
        with st.spinner("Initializing CyberShield Engine..."):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=18,
                random_state=42
            )
            model.fit(X_train, y_train)
            st.session_state["model"] = model
            st.success("Model trained successfully")

with col2:
    st.subheader("Detection Performance")
    if "model" in st.session_state:
        preds = st.session_state["model"].predict(X_test)
        acc = accuracy_score(y_test, preds)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc * 100:.2f}%")
        m2.metric("Total Samples", len(df))
        m3.metric("Threats Detected", int(np.sum(preds)))

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Train the model to view performance metrics.")

# ---------------- LIVE TRAFFIC ANALYZER ----------------
st.divider()
st.subheader("Live Traffic Risk Analyzer")

c1, c2, c3, c4 = st.columns(4)
flow = c1.number_input("Flow Duration", 0, 100000, 600)
packets = c2.number_input("Packets", 0, 500, 120)
pkt_len = c3.number_input("Packet Size Mean", 0, 1500, 450)
active = c4.number_input("Active Time", 0, 1000, 80)

if st.button("Analyze Traffic"):
    if "model" in st.session_state:
        sample = np.array([[80, flow, packets, pkt_len, active]])
        pred = st.session_state["model"].predict(sample)

        if pred[0] == 1:
            st.markdown(
                "<h3 style='color:#ff5252;'>High Risk: Intrusion Detected</h3>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h3 style='color:#69f0ae;'>Low Risk: Normal Traffic</h3>",
                unsafe_allow_html=True
            )
    else:
        st.error("Please train the model before analyzing traffic.")
