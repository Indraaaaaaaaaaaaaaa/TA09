import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Simulasi Pertumbuhan Trending Video",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple styling to make the app feel more polished
st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at 20% 20%, #f0f7ff 0, #ffffff 40%);
    }
    .metric-card {
        padding: 14px 16px;
        border-radius: 12px;
        background: #ffffff;
        border: 1px solid #edf2f7;
        box-shadow: 0px 4px 24px rgba(15, 23, 42, 0.06);
    }
    .section-title {
        font-weight: 700;
        font-size: 20px;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# LOAD DATASET
# -----------------------
plt.style.use("seaborn-v0_8-whitegrid")

df = pd.read_csv("CAvideos.csv")


def format_number(num: float) -> str:
    """Pretty print large numbers with suffix."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    if num >= 1_000:
        return f"{num/1_000:.2f}K"
    return f"{num:.0f}"

# Pilih video otomatis
def pilih_video_otomatis(df, min_points=5):
    counts = df.groupby('video_id').size().sort_values(ascending=False)
    for vid, cnt in counts.items():
        if cnt < min_points:
            break
        sub = df[df['video_id'] == vid].copy()
        sub = sub.sort_values('trending_date')
        views = sub['views'].values
        if np.all(np.diff(views) > 0):
            return vid, sub
    vid = counts.index[0]
    sub = df[df['video_id'] == vid].sort_values('trending_date')
    return vid, sub

video_id, df_vid = pilih_video_otomatis(df)
df_vid = df_vid.sort_values('trending_date')

# Time series
t_data = np.arange(len(df_vid))
U_data = df_vid['views'].astype(float).values


# -----------------------
# PARAMETER ESTIMATION
# -----------------------
def estimate_params(t, U):
    Umax_est = U.max() * 1.1
    if np.any(U >= Umax_est):
        Umax_est = U.max() * 1.5
    y = np.log(Umax_est - U)
    slope, intercept = np.polyfit(t, y, 1)
    r_est = -slope
    return r_est, Umax_est

r_est, Umax_est = estimate_params(t_data, U_data)

# -----------------------
# ODE MODEL
# -----------------------
def model(U, t, r, Umax):
    return r * (Umax - U)

def euler(func, U0, t_points, params):
    U = np.zeros(len(t_points))
    U[0] = U0
    h = t_points[1] - t_points[0]
    for i in range(len(t_points)-1):
        slope = func(U[i], t_points[i], *params)
        U[i+1] = U[i] + h * slope
    return U

def rk4(func, U0, t_points, params):
    U = np.zeros(len(t_points))
    U[0] = U0
    h = t_points[1] - t_points[0]
    for i in range(len(t_points)-1):
        t = t_points[i]
        Ui = U[i]
        k1 = func(Ui, t, *params)
        k2 = func(Ui + 0.5*h*k1, t + 0.5*h, *params)
        k3 = func(Ui + 0.5*h*k2, t + 0.5*h, *params)
        k4 = func(Ui + h*k3, t + h, *params)
        U[i+1] = Ui + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return U

# -----------------------
# STREAMLIT UI
# -----------------------
st.title("Simulasi Pertumbuhan Eksponensial Terbatas")
st.caption("Dataset: YouTube Trending (CAvideos.csv) • Metode Euler & RK4")

st.sidebar.header("Pengaturan Model")
r = st.sidebar.slider(
    "Laju Pertumbuhan (r)",
    0.0001,
    float(r_est * 2),
    float(r_est),
    step=0.0001,
)
Umax = st.sidebar.slider(
    "Umax (batas tampilan)",
    float(U_data.max()),
    float(Umax_est * 2),
    float(Umax_est),
)
h = st.sidebar.slider("Step Size (h)", 0.1, 2.0, 1.0, help="Resolusi waktu simulasi")

with st.container():
    st.markdown('<div class="section-title">Video Terpilih</div>', unsafe_allow_html=True)
    info_col, metric_col = st.columns([1.7, 1.3])
    with info_col:
        st.write(f"**Video ID:** `{video_id}`")
        st.write("**Judul:**", df_vid["title"].iloc[0])
        st.write("**Channel:**", df_vid["channel_title"].iloc[0])
        st.write(
            "**Periode Trending:**",
            f"{df_vid['trending_date'].min()} → {df_vid['trending_date'].max()}",
        )
    with metric_col:
        m1, m2, m3 = st.columns(3)
        m1.metric("View Awal", format_number(U_data[0]))
        m2.metric("View Akhir", format_number(U_data[-1]))
        growth_pct = (
            ((U_data[-1] - U_data[0]) / U_data[0]) * 100 if U_data[0] else 0
        )
        m3.metric("Kenaikan", f"{growth_pct:.1f}%")
        st.markdown(
            f"Data poin: **{len(U_data)}** • Max view: **{format_number(U_data.max())}**"
        )

# Waktu simulasi
t_sim = np.arange(0, len(U_data), h)

# Simulasi
U0 = U_data[0]
params = (r, Umax)

U_euler = euler(model, U0, t_sim, params)
U_rk4 = rk4(model, U0, t_sim, params)

# Plot
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(t_data, U_data, "o-", color="#111827", label="Data Asli", markersize=5)
ax.plot(t_sim, U_euler, "--", color="#ef4444", label="Euler", linewidth=2)
ax.plot(t_sim, U_rk4, "-", color="#3b82f6", label="RK4", linewidth=2.5)
ax.set_xlabel("t")
ax.set_ylabel("Views")
ax.legend()

st.markdown('<div class="section-title">Perbandingan Model vs Data</div>', unsafe_allow_html=True)
st.pyplot(fig)

# Error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Interpolasi prediksi ke t_data
U_rk4_interp = np.interp(t_data, t_sim, U_rk4)
err = mae(U_data, U_rk4_interp)

st.markdown('<div class="section-title">Error RK4 terhadap Data Asli</div>', unsafe_allow_html=True)
st.write(f"MAE: **{err:,.2f}**")

with st.expander("Catatan Model"):
    st.write(
        "- Model menggunakan pertumbuhan eksponensial terbatas dengan parameter r dan Umax."
    )
    st.write(
        "- Anda dapat menyesuaikan step size (h) untuk melihat stabilitas numerik Euler vs RK4."
    )
    st.write(
        "- MAE dihitung setelah menginterpolasi hasil simulasi ke titik waktu data asli."
    )
