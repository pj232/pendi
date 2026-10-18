# -*- coding: utf-8 -*-
"""
==============================================================================
 DamLab - Collecte, analyse et dimensionnement d'un BARRAGE-POIDS (béton)
==============================================================================
 Installation :   pip install streamlit pandas numpy plotly
 Lancement    :   streamlit run main.py      (ou bouton Run de PyCharm via terminal)

 Modules :
   1. Tableau de bord
   2. Hydrologie & crue de projet (Gumbel, évacuateur, revanche)
   3. Dimensionnement & stabilité (glissement, renversement, contraintes,
      sous-pressions, vase, séisme Westergaard) + dimensionnement automatique
   4. Collecte des données d'auscultation (saisie, import CSV, démo)
   5. Analyse & surveillance (séries, seuils, anomalies, tendances, corrélations)
   6. Rapport & export

 NB : outil de PRÉ-DIMENSIONNEMENT pédagogique. Toute étude réelle doit être
 validée par un ingénieur qualifié et conforme aux normes en vigueur.
==============================================================================
"""
import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="DamLab - Barrage-poids",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1. STYLE (couleurs fluides et dynamiques)
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
  background: linear-gradient(-45deg, #0b132b, #12305c, #0f4c75, #3a0ca3, #00796b, #0b132b);
  background-size: 400% 400%;
  animation: gradientFlow 28s ease infinite;
}
@keyframes gradientFlow {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes shine { to { background-position: 300% center; } }
@keyframes floaty { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }
@keyframes pulseGlow {
  0%,100% { box-shadow: 0 0 0 0 rgba(34,211,238,.45); }
  50%     { box-shadow: 0 0 18px 4px rgba(34,211,238,.25); }
}

header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: linear-gradient(190deg, rgba(15,76,117,.92), rgba(58,12,163,.80) 70%, rgba(0,121,107,.80));
  backdrop-filter: blur(14px);
  border-right: 1px solid rgba(255,255,255,.12);
}
h2, h3, h4,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
  color: #eaf6ff !important;
}

/* Hero */
.hero {
  padding: 26px 30px; margin-bottom: 18px; border-radius: 22px;
  background: linear-gradient(120deg, rgba(34,211,238,.18), rgba(167,139,250,.18), rgba(244,114,182,.15));
  border: 1px solid rgba(255,255,255,.18); backdrop-filter: blur(10px);
}
.hero h1 {
  margin: 0; font-weight: 700; font-size: 2.2rem;
  background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6, #fbbf24, #22d3ee);
  background-size: 300% auto; -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; animation: shine 7s linear infinite;
}
.hero p { margin: 6px 0 0 0; color: #cfe8ff !important; font-weight: 300; }

/* Cartes métriques */
[data-testid="stMetric"] {
  background: linear-gradient(135deg, rgba(255,255,255,.10), rgba(255,255,255,.04));
  border: 1px solid rgba(255,255,255,.16); border-radius: 18px; padding: 14px 18px;
  backdrop-filter: blur(8px); transition: all .3s ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-4px); border-color: rgba(34,211,238,.7);
  box-shadow: 0 10px 28px rgba(34,211,238,.22);
}
[data-testid="stMetricValue"] { color: #5eead4 !important; font-weight: 700; }
[data-testid="stMetricLabel"] p { color: #bfe3ff !important; }

/* Boutons */
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(90deg, #06b6d4, #6366f1, #d946ef); background-size: 200% auto;
  color: white; border: none; border-radius: 14px; font-weight: 600; padding: .55rem 1.2rem;
  transition: all .35s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background-position: right center; transform: translateY(-2px) scale(1.02);
  box-shadow: 0 8px 24px rgba(99,102,241,.45); color: white;
}

/* Onglets */
button[data-baseweb="tab"] { border-radius: 12px 12px 0 0; transition: all .25s; }
button[data-baseweb="tab"] p { color: #cfe8ff !important; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] { background: rgba(34,211,238,.16); }
div[data-baseweb="tab-highlight"] { background: linear-gradient(90deg,#22d3ee,#a78bfa,#f472b6) !important; height: 3px; }

/* Champs de saisie */
.stNumberInput input, .stTextInput input, .stDateInput input, .stTimeInput input, textarea {
  background: rgba(255,255,255,.09) !important; color: #fff !important;
  border-radius: 10px !important; border: 1px solid rgba(255,255,255,.18) !important;
}
div[data-baseweb="select"] > div {
  background: rgba(255,255,255,.09) !important; color: #fff !important;
  border-radius: 10px !important; border: 1px solid rgba(255,255,255,.18) !important;
}
div[data-baseweb="select"] span { color: #fff !important; }
[data-testid="stExpander"] {
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.14); border-radius: 16px;
}
[data-testid="stExpander"] summary p { color: #eaf6ff !important; font-weight: 600; }

/* Badges & tableaux HTML */
.badge { padding: 3px 12px; border-radius: 999px; font-size: .78rem; font-weight: 700; color: #04121f; display: inline-block; }
.badge.ok   { background: linear-gradient(90deg,#34d399,#22d3ee); }
.badge.ko   { background: linear-gradient(90deg,#fb7185,#f472b6); animation: pulseGlow 2s infinite; }
.badge.warn { background: linear-gradient(90deg,#fbbf24,#fb923c); }
.tbl-wrap { overflow-x: auto; border-radius: 16px; border: 1px solid rgba(255,255,255,.16); margin: 6px 0 14px 0; }
.tbl { width: 100%; border-collapse: collapse; color: #eaf6ff; font-size: .9rem; }
.tbl th { background: linear-gradient(90deg, rgba(34,211,238,.35), rgba(167,139,250,.35)); padding: 10px 12px; text-align: left; }
.tbl td { padding: 8px 12px; border-top: 1px solid rgba(255,255,255,.08); }
.tbl tr:hover td { background: rgba(255,255,255,.06); }
.side-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 0; animation: floaty 4s ease-in-out infinite; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PALETTE = ["#22d3ee", "#a78bfa", "#f472b6", "#fbbf24", "#34d399", "#60a5fa", "#fb7185", "#f97316"]
CASE_COLORS = {"Normal": "#22d3ee", "Crue": "#f472b6", "Séisme": "#fbbf24"}


def hero(title, subtitle):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def style_fig(fig, height=430, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.04)",
        font=dict(family="Poppins", color="#eaf6ff"),
        height=height,
        margin=dict(l=20, r=20, t=55 if title else 20, b=20),
        title=dict(text=title, font=dict(size=16)) if title else None,
        legend=dict(bgcolor="rgba(0,0,0,0.25)", bordercolor="rgba(255,255,255,0.15)", borderwidth=1),
    )
    return fig


def badge(ok, txt_ok="CONFORME", txt_ko="NON CONFORME"):
    return f"<span class='badge {'ok' if ok else 'ko'}'>{txt_ok if ok else txt_ko}</span>"


def status_badge(s):
    cls = {"Normal": "ok", "ALERTE": "warn", "ALARME": "ko"}.get(s, "ok")
    return f"<span class='badge {cls}'>{s}</span>"


def html_table(headers, rows):
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    st.markdown(
        f"<div class='tbl-wrap'><table class='tbl'><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# 2. PARAMÈTRES PAR DÉFAUT & ÉTAT DE SESSION
# =============================================================================
DEFAULT_QMAX = [420, 515, 380, 610, 470, 720, 395, 560, 850, 445, 630, 505, 690, 410, 780, 545, 600, 490, 670, 455]

DEFAULT_PARAMS = dict(
    nom="Barrage de démonstration",
    L_crete=250.0,          # longueur en crête (m)
    h_normal=40.0,          # hauteur d'eau à la retenue normale (m)
    # Hydrologie / évacuateur
    T=1000.0, L_dev=60.0, C_dev=2.1, atten=15.0, vent=100.0, fetch=8.0, fb_min=1.0,
    # Géométrie
    H_auto=True, H=46.0, a=6.0, n=0.0, m=0.80,
    # Matériaux / fondation
    gamma_c=24.0, gamma_w=9.81, phi=38.0, coh=100.0, sig_adm=3000.0,
    # Actions
    ht_n=2.0, ht_f=8.0, eff_drain=0.67, drain_pos=0.10,
    hs=8.0, gamma_s=8.0, phi_s=25.0, kh=0.12,
    # Coût
    cout_beton=120.0,
)

DEFAULT_SEUILS = pd.DataFrame(
    [
        ["Niveau retenue", "m", 41.0, 43.0],
        ["Piézomètre", "m", 14.0, 17.0],
        ["Déplacement", "mm", 8.0, 12.0],
        ["Débit de fuite", "L/s", 12.0, 20.0],
        ["Température béton", "°C", 30.0, 35.0],
        ["Pluviométrie", "mm", 60.0, 100.0],
    ],
    columns=["Type", "Unité", "Seuil d'alerte", "Seuil d'alarme"],
)
UNITS = dict(zip(DEFAULT_SEUILS["Type"], DEFAULT_SEUILS["Unité"]))
DATA_COLS = ["Date", "Capteur", "Type", "Valeur", "Unité", "Opérateur", "Observation"]

CRITERIA = {
    "Normal": dict(fs_s=1.5, fs_o=1.5, e=1 / 6, sig=1.00, label="tiers central"),
    "Crue":   dict(fs_s=1.3, fs_o=1.3, e=1 / 4, sig=1.15, label="moitié centrale"),
    "Séisme": dict(fs_s=1.1, fs_o=1.1, e=1 / 3, sig=1.33, label="deux tiers centraux"),
}


def make_demo_data(days=365, seed=7):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now().normalize()
    dates = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D") + pd.Timedelta(hours=8)
    t = np.arange(days)
    level = 36 + 3.5 * np.sin(2 * np.pi * (t - 60) / 365) + rng.normal(0, 0.25, days)
    temp = 20 + 7 * np.sin(2 * np.pi * (t - 100) / 365) + rng.normal(0, 0.8, days)
    series = {}
    series[("NR-1", "Niveau retenue")] = level
    for name, base, k in [("PZ-1", 6.5, 0.45), ("PZ-2", 7.5, 0.40), ("PZ-3", 7.0, 0.50)]:
        v = base + k * (level - 36) + rng.normal(0, 0.12, days)
        if name == "PZ-3":                       # dérive simulée sur les 65 derniers jours
            v = v + np.clip(t - (days - 65), 0, None) * 0.13
            v[-20] += 1.5                        # pic isolé
        series[(name, "Piézomètre")] = v
    for name, base, k in [("DEP-1", 3.0, -0.35), ("DEP-2", 2.2, -0.30)]:
        v = base + 0.012 * t * 0.3 + k * (temp - 20) * 0.12 + 0.25 * (level - 36) + rng.normal(0, 0.12, days)
        series[(name, "Déplacement")] = v
    leak = 6 + 0.9 * (level - 36) + rng.normal(0, 0.35, days) + np.clip(t - (days - 45), 0, None) * 0.12
    series[("FUI-1", "Débit de fuite")] = leak
    series[("TB-1", "Température béton")] = 18 + 5 * np.sin(2 * np.pi * (t - 120) / 365) + rng.normal(0, 0.4, days)
    rain = np.where(rng.random(days) < 0.3, rng.exponential(6, days), 0.0)
    rain[rng.integers(0, days, 3)] = [55, 72, 66]
    series[("PLV-1", "Pluviométrie")] = rain

    frames = []
    for (cap, typ), vals in series.items():
        frames.append(pd.DataFrame({
            "Date": dates, "Capteur": cap, "Type": typ, "Valeur": np.round(vals, 3),
            "Unité": UNITS[typ], "Opérateur": "Auto", "Observation": "",
        }))
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)


def clean_df(df):
    df = df.copy()
    for c in DATA_COLS:
        if c not in df.columns:
            df[c] = "" if c in ("Unité", "Opérateur", "Observation") else np.nan
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Valeur"] = pd.to_numeric(df["Valeur"], errors="coerce")
    df["Capteur"] = df["Capteur"].astype(str).str.strip()
    df = df.dropna(subset=["Date", "Valeur"])
    df = df[df["Capteur"].str.len() > 0]
    df = df[df["Capteur"].str.lower() != "nan"]
    reg = {}
    if "Type" in df.columns:
        reg = df.dropna(subset=["Type"]).groupby("Capteur")["Type"].last().to_dict()
    df["Type"] = [t if isinstance(t, str) and t.strip() else reg.get(c, "Autre")
                  for t, c in zip(df["Type"], df["Capteur"])]
    df["Unité"] = [u if isinstance(u, str) and u.strip() else UNITS.get(t, "")
                   for u, t in zip(df["Unité"], df["Type"])]
    df["Opérateur"] = df["Opérateur"].fillna("").astype(str)
    df["Observation"] = df["Observation"].fillna("").astype(str)
    return df[DATA_COLS].sort_values("Date").reset_index(drop=True)


def init_state():
    ss = st.session_state
    if "p" not in ss:
        ss.p = dict(DEFAULT_PARAMS)
    if "crues" not in ss:
        ss.crues = pd.DataFrame({"Année": list(range(2006, 2006 + len(DEFAULT_QMAX))),
                                 "Qmax (m³/s)": [float(x) for x in DEFAULT_QMAX]})
    if "data" not in ss:
        ss.data = make_demo_data()
    if "seuils" not in ss:
        ss.seuils = DEFAULT_SEUILS.copy()


def num(label, key, mn, mx, step, fmt="%.2f", help=None, container=st):
    p = st.session_state.p
    p[key] = container.number_input(
        label, min_value=float(mn), max_value=float(mx), value=float(p[key]),
        step=float(step), format=fmt, key=f"w_{key}", help=help,
    )
    return p[key]


# =============================================================================
# 3. HYDROLOGIE
# =============================================================================
def gumbel_params(q):
    q = np.asarray(q, float)
    s = q.std(ddof=1)
    alpha = math.sqrt(6) / math.pi * s
    u = q.mean() - 0.5772 * alpha
    return u, alpha


def gumbel_quantile(u, alpha, T):
    T = max(float(T), 1.01)
    return u - alpha * math.log(-math.log(1 - 1 / T))


def compute_hydro(p, qvals):
    q = np.asarray(qvals, float)
    u, alpha = gumbel_params(q)
    QT = gumbel_quantile(u, alpha, p["T"])
    Qout = QT * (1 - p["atten"] / 100.0)
    Hs = (Qout / (p["C_dev"] * p["L_dev"])) ** (2 / 3)
    F, U = p["fetch"], p["vent"]
    hv = 0.032 * math.sqrt(U * F) + 0.763 - 0.271 * F ** 0.75 if F <= 18 else 0.032 * math.sqrt(U * F)
    hv = max(hv, 0.3)
    v = 1.5 + 2 * hv
    Fb = max(0.75 * hv + v * v / (2 * 9.81), p["fb_min"])
    H_req = p["h_normal"] + Hs + Fb
    return dict(u=u, alpha=alpha, QT=QT, Qout=Qout, Hs=Hs, hv=hv, Fb=Fb, H_req=H_req)


def get_hydro():
    p = st.session_state.p
    q = pd.to_numeric(st.session_state.crues.iloc[:, 1], errors="coerce").dropna()
    q = q[q > 0]
    if len(q) < 5:
        q = pd.Series(DEFAULT_QMAX, dtype=float)
    return compute_hydro(p, q.values), q.values


def get_H(p, hydro):
    if p["H_auto"]:
        return math.ceil(hydro["H_req"] * 2) / 2
    return float(p["H"])


# =============================================================================
# 4. STABILITÉ DU BARRAGE-POIDS (par mètre linéaire)
# =============================================================================
def piecewise_force(xs, ps):
    """Force (aire) et position du centre de poussée d'un diagramme linéaire par morceaux."""
    F, M = 0.0, 0.0
    for i in range(len(xs) - 1):
        x1, x2, p1, p2 = xs[i], xs[i + 1], ps[i], ps[i + 1]
        w = x2 - x1
        f = (p1 + p2) / 2 * w
        xc = x1 + w * (p1 + 2 * p2) / (3 * (p1 + p2)) if abs(p1 + p2) > 1e-12 else x1 + w / 2
        F += f
        M += f * xc
    return F, (M / F if abs(F) > 1e-12 else xs[-1] / 2)


def analyse_case(p, H, a, n, m, h, ht, kh, case):
    gc, gw = p["gamma_c"], p["gamma_w"]
    B = n * H + a + m * H
    h, ht = min(h, H), min(ht, H)
    hs = min(p["hs"], H)
    rows = []          # (Action, Effort, Bras, Moment, Type)
    Vstab = Hdest = Hstab = Mr = Mo = U = 0.0

    # --- Poids propre
    for name, W, x, _yc in [
        ("Poids propre - parement amont", 0.5 * n * H * H * gc, 2 * n * H / 3, H / 3),
        ("Poids propre - crête", a * H * gc, n * H + a / 2, H / 2),
        ("Poids propre - parement aval", 0.5 * m * H * H * gc, n * H + a + m * H / 3, H / 3),
    ]:
        if W > 0:
            arm = B - x
            rows.append((name, W, arm, W * arm, "Stabilisant"))
            Vstab += W
            Mr += W * arm
    W_self = Vstab

    # --- Eau amont (verticale sur parement incliné)
    Wv = gw * 0.5 * n * h * h
    if Wv > 0:
        arm = B - n * h / 3
        rows.append(("Poids de l'eau sur parement amont", Wv, arm, Wv * arm, "Stabilisant"))
        Vstab += Wv
        Mr += Wv * arm
    # --- Eau aval (verticale)
    Wt = gw * 0.5 * m * ht * ht
    if Wt > 0:
        arm = m * ht / 3
        rows.append(("Poids de l'eau aval sur parement", Wt, arm, Wt * arm, "Stabilisant"))
        Vstab += Wt
        Mr += Wt * arm

    # --- Poussée hydrostatique amont
    Pw = 0.5 * gw * h * h
    rows.append(("Poussée hydrostatique amont", Pw, h / 3, -Pw * h / 3, "Déstabilisant"))
    Hdest += Pw
    Mo += Pw * h / 3
    # --- Contre-poussée aval
    Pt = 0.5 * gw * ht * ht
    if Pt > 0:
        rows.append(("Contre-poussée hydrostatique aval", Pt, ht / 3, Pt * ht / 3, "Stabilisant"))
        Hstab += Pt
        Mr += Pt * ht / 3
    # --- Vase
    Ka = math.tan(math.radians(45 - p["phi_s"] / 2)) ** 2
    Ps = 0.5 * p["gamma_s"] * Ka * hs * hs
    if Ps > 0:
        rows.append(("Poussée des sédiments (vase)", Ps, hs / 3, -Ps * hs / 3, "Déstabilisant"))
        Hdest += Ps
        Mo += Ps * hs / 3
    # --- Sous-pressions
    if p["eff_drain"] > 0:
        xd = p["drain_pos"] * B
        hd = ht + (1 - p["eff_drain"]) * (h - ht)
        px, ph = [0.0, xd, B], [h, hd, ht]
    else:
        px, ph = [0.0, B], [h, ht]
    Fh, xc = piecewise_force(px, ph)
    U = gw * Fh
    arm_u = B - xc
    rows.append(("Sous-pressions", U, arm_u, -U * arm_u, "Déstabilisant"))
    Mo += U * arm_u
    # --- Séisme
    if kh > 0:
        Fi_total, Mi = 0.0, 0.0
        for W, yc in [(0.5 * n * H * H * gc, H / 3), (a * H * gc, H / 2), (0.5 * m * H * H * gc, H / 3)]:
            Fi_total += kh * W
            Mi += kh * W * yc
        rows.append(("Inertie sismique du barrage", Fi_total, Mi / Fi_total if Fi_total else 0, -Mi, "Déstabilisant"))
        Hdest += Fi_total
        Mo += Mi
        Pe = 0.583 * kh * gw * h * h
        rows.append(("Surpression hydrodynamique (Westergaard)", Pe, 0.4 * h, -Pe * 0.4 * h, "Déstabilisant"))
        Hdest += Pe
        Mo += Pe * 0.4 * h

    sumV = Vstab - U
    sumH = Hdest - Hstab
    tanphi = math.tan(math.radians(p["phi"]))
    fs_s = (max(sumV, 0) * tanphi + p["coh"] * B) / sumH if sumH > 0 else 99.0
    fs_o = Mr / Mo if Mo > 0 else 99.0
    xbar = (Mr - Mo) / sumV if sumV > 0 else 0.0
    e = B / 2 - xbar
    # Contraintes à la base (section fissurée hors noyau central)
    if sumV <= 0 or xbar <= 0 or xbar >= B:
        sig_toe, sig_heel = 1e9, 1e9
    elif abs(e) <= B / 6:
        s0 = sumV / B
        sig_toe, sig_heel = s0 * (1 + 6 * e / B), s0 * (1 - 6 * e / B)
    elif e > 0:
        sig_toe, sig_heel = 2 * sumV / (3 * xbar), 0.0
    else:
        sig_toe, sig_heel = 0.0, 2 * sumV / (3 * (B - xbar))

    crit = CRITERIA[case]
    sig_lim = p["sig_adm"] * crit["sig"]
    sig_max = max(sig_toe, sig_heel)
    checks = dict(
        s=fs_s >= crit["fs_s"], o=fs_o >= crit["fs_o"],
        e=abs(e) <= B * crit["e"], sig=sig_max <= sig_lim,
    )
    marge = min(fs_s / crit["fs_s"], fs_o / crit["fs_o"],
                (B * crit["e"]) / max(abs(e), 1e-6), sig_lim / max(sig_max, 1e-6))
    return dict(
        case=case, H=H, B=B, h=h, ht=ht, hs=hs, kh=kh, rows=rows, W=W_self, U=U,
        sumV=sumV, sumH=sumH, Mr=Mr, Mo=Mo, fs_s=fs_s, fs_o=fs_o, xbar=xbar, e=e,
        sig_toe=sig_toe, sig_heel=sig_heel, sig_lim=sig_lim, checks=checks,
        ok_all=all(checks.values()), marge=marge, px=px, ph=ph,
    )


def run_cases(p, hydro, H, m=None):
    m = p["m"] if m is None else m
    hf = min(p["h_normal"] + hydro["Hs"], H)
    return {
        "Normal": analyse_case(p, H, p["a"], p["n"], m, p["h_normal"], p["ht_n"], 0.0, "Normal"),
        "Crue": analyse_case(p, H, p["a"], p["n"], m, hf, p["ht_f"], 0.0, "Crue"),
        "Séisme": analyse_case(p, H, p["a"], p["n"], m, p["h_normal"], p["ht_n"], p["kh"], "Séisme"),
    }


def find_min_m(p, hydro, H):
    for m in np.arange(0.30, 1.601, 0.01):
        r = run_cases(p, hydro, H, float(m))
        if all(c["ok_all"] for c in r.values()):
            return round(float(m), 2)
    return None


def apply_m(val):
    st.session_state.p["m"] = float(val)
    st.session_state["w_m"] = float(val)


# --- Figures du barrage -------------------------------------------------------
def fig_section(res, p):
    H, B, h, ht, hs = res["H"], res["B"], res["h"], res["ht"], res["hs"]
    a, n, m = p["a"], p["n"], p["m"]
    pad = max(B * 0.45, 15)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[-pad, B + pad, B + pad, -pad, -pad], y=[0, 0, -0.18 * H, -0.18 * H, 0], fill="toself",
        fillcolor="rgba(161,98,7,0.55)", line=dict(width=0), name="Fondation", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[-pad, 0, n * h, -pad], y=[0, 0, h, h], fill="toself", fillcolor="rgba(34,211,238,0.35)",
        line=dict(color="#22d3ee", width=1), name="Retenue", hoverinfo="skip"))
    if hs > 0:
        fig.add_trace(go.Scatter(
            x=[-pad, 0, n * hs, -pad], y=[0, 0, hs, hs], fill="toself", fillcolor="rgba(180,140,90,0.75)",
            line=dict(width=0), name="Sédiments", hoverinfo="skip"))
    if ht > 0:
        fig.add_trace(go.Scatter(
            x=[B, B + pad, B + pad, B - m * ht], y=[0, 0, ht, ht], fill="toself",
            fillcolor="rgba(96,165,250,0.35)", line=dict(color="#60a5fa", width=1),
            name="Eau aval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[0, n * H, n * H + a, B, 0], y=[0, H, H, 0, 0], fill="toself",
        fillcolor="rgba(148,163,184,0.88)", line=dict(color="#f1f5f9", width=2.5), name="Barrage",
        hovertemplate=f"H = {H:.1f} m<br>B = {B:.1f} m<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[B / 3, 2 * B / 3], y=[-0.05 * H, -0.05 * H], mode="lines",
        line=dict(color="#34d399", width=7), name="Tiers central", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[B - res["xbar"]], y=[0], mode="markers",
        marker=dict(symbol="diamond", size=15, color="#fbbf24", line=dict(color="white", width=2)),
        name="Résultante", hovertemplate="Résultante à %{x:.1f} m du talon<extra></extra>"))
    fig.add_annotation(x=-pad * 0.55, y=h, text=f"Niveau d'eau {h:.1f} m", showarrow=False,
                       yshift=14, font=dict(color="#22d3ee"))
    fig.add_annotation(x=n * H + a / 2, y=H, text=f"Crête {a:.1f} m", showarrow=False,
                       yshift=14, font=dict(color="#e2e8f0"))
    fig.add_annotation(x=B / 2, y=-0.13 * H, text=f"Base B = {B:.1f} m", showarrow=False,
                       font=dict(color="#fde68a"))
    fig.update_xaxes(range=[-pad, B + pad], showgrid=False, zeroline=False, title="x (m)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[-0.2 * H, H * 1.18], showgrid=False,
                     zeroline=False, title="z (m)")
    return style_fig(fig, 520, f"Coupe type - cas {res['case']}")


def fig_base(res, p):
    B = res["B"]
    xs = np.linspace(0, B, 80)
    sig = res["sig_heel"] + (res["sig_toe"] - res["sig_heel"]) * xs / B
    up = p["gamma_w"] * np.interp(xs, res["px"], res["ph"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=sig, name="Contrainte σ (kPa)", fill="tozeroy",
                             line=dict(color="#22d3ee", width=3), fillcolor="rgba(34,211,238,0.25)"))
    fig.add_trace(go.Scatter(x=xs, y=up, name="Sous-pression u (kPa)", fill="tozeroy",
                             line=dict(color="#f472b6", width=3), fillcolor="rgba(244,114,182,0.22)"))
    fig.add_hline(y=res["sig_lim"], line=dict(color="#fbbf24", dash="dash"),
                  annotation_text="σ admissible", annotation_font_color="#fbbf24")
    fig.update_xaxes(title="Distance depuis le talon amont (m)")
    fig.update_yaxes(title="kPa")
    return style_fig(fig, 380, f"Contraintes et sous-pressions à la base - {res['case']}")


# =============================================================================
# 5. DONNÉES D'AUSCULTATION
# =============================================================================
def get_status(val, typ, seuils):
    row = seuils[seuils["Type"] == typ]
    if row.empty:
        return "Normal"
    a, b = row.iloc[0]["Seuil d'alerte"], row.iloc[0]["Seuil d'alarme"]
    return "ALARME" if val >= b else ("ALERTE" if val >= a else "Normal")


def sensor_summary(df, seuils):
    rows = []
    for cap, g in df.groupby("Capteur"):
        g = g.sort_values("Date")
        typ = g["Type"].iloc[-1]
        v = g["Valeur"].to_numpy(float)
        days = (g["Date"] - g["Date"].iloc[0]).dt.total_seconds().to_numpy() / 86400
        slope = float(np.polyfit(days, v, 1)[0]) if len(g) >= 3 and days.max() > 0 else 0.0
        last = float(v[-1])
        rows.append(dict(
            Capteur=cap, Type=typ, Unité=g["Unité"].iloc[-1], N=len(g), Min=float(v.min()),
            Moyenne=float(v.mean()), Max=float(v.max()), Ecart=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            Dernière=last, Date=g["Date"].iloc[-1].strftime("%d/%m/%Y"), Tendance=slope,
            Statut=get_status(last, typ, seuils),
        ))
    return pd.DataFrame(rows)


def robust_outliers(y, thr=3.5):
    y = np.asarray(y, float)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad < 1e-9:
        return np.zeros(len(y), dtype=bool)
    return np.abs(0.6745 * (y - med) / mad) > thr


# =============================================================================
# 6. PAGES
# =============================================================================
def page_dashboard():
    p = st.session_state.p
    hero(f"🌊 {p['nom']}", "Barrage-poids en béton - Pré-dimensionnement, collecte et surveillance en temps réel")
    hydro, _ = get_hydro()
    H = get_H(p, hydro)
    cases = run_cases(p, hydro, H)
    rn = cases["Normal"]
    vol_m2 = p["a"] * H + 0.5 * (p["n"] + p["m"]) * H * H
    df, seuils = st.session_state.data, st.session_state.seuils
    summ = sensor_summary(df, seuils) if not df.empty else pd.DataFrame()
    n_alert = int((summ["Statut"] == "ALERTE").sum()) if not summ.empty else 0
    n_alarm = int((summ["Statut"] == "ALARME").sum()) if not summ.empty else 0

    c = st.columns(5)
    c[0].metric("Hauteur du barrage", f"{H:.1f} m")
    c[1].metric("Largeur à la base", f"{rn['B']:.1f} m")
    c[2].metric("Crue de projet", f"{hydro['QT']:.0f} m³/s", f"T = {p['T']:.0f} ans", delta_color="off")
    c[3].metric("Béton (total)", f"{vol_m2 * p['L_crete'] / 1000:.1f} 10³ m³")
    c[4].metric("Conformité", "OUI ✅" if all(x["ok_all"] for x in cases.values()) else "NON ⚠️")

    c = st.columns(5)
    c[0].metric("FS glissement (normal)", f"{rn['fs_s']:.2f}", f"min {CRITERIA['Normal']['fs_s']}")
    c[1].metric("FS renversement (normal)", f"{rn['fs_o']:.2f}", f"min {CRITERIA['Normal']['fs_o']}")
    c[2].metric("Mesures enregistrées", f"{len(df):,}".replace(",", " "))
    c[3].metric("Capteurs en alerte", n_alert)
    c[4].metric("Capteurs en alarme", n_alarm)

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(fig_section(rn, p), key="dash_section")
    with right:
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=min(rn["fs_s"], 5.0),
            title={"text": "Coefficient de sécurité au glissement", "font": {"size": 15}},
            gauge={"axis": {"range": [0, 5]}, "bar": {"color": "#22d3ee"},
                   "steps": [{"range": [0, 1.0], "color": "rgba(251,113,133,.55)"},
                             {"range": [1.0, CRITERIA["Normal"]["fs_s"]], "color": "rgba(251,191,36,.55)"},
                             {"range": [CRITERIA["Normal"]["fs_s"], 5], "color": "rgba(52,211,153,.45)"}],
                   "threshold": {"line": {"color": "white", "width": 4}, "value": CRITERIA["Normal"]["fs_s"]}}))
        st.plotly_chart(style_fig(g, 300), key="dash_gauge")
        lev = df[df["Type"] == "Niveau retenue"] if not df.empty else df
        if not lev.empty:
            cap = lev["Capteur"].iloc[0]
            gg = lev[lev["Capteur"] == cap].sort_values("Date")
            f2 = go.Figure(go.Scatter(x=gg["Date"], y=gg["Valeur"], mode="lines", fill="tozeroy",
                                      line=dict(color="#a78bfa", width=2.5), fillcolor="rgba(167,139,250,.22)"))
            st.plotly_chart(style_fig(f2, 210, f"Niveau de la retenue ({cap})"), key="dash_level")

    st.markdown("#### Verdict des vérifications")
    rows = []
    for k, r in cases.items():
        rows.append([k, f"{r['fs_s']:.2f}", f"{r['fs_o']:.2f}", f"{r['e']:.2f} m",
                     f"{max(r['sig_toe'], r['sig_heel']):.0f} kPa", badge(r["ok_all"])])
    html_table(["Cas de charge", "FS glissement", "FS renversement", "Excentricité", "σ max", "Verdict"], rows)
    if not summ.empty and (n_alert + n_alarm) > 0:
        st.warning("Des capteurs dépassent leurs seuils : voir la page « Analyse & surveillance ».")


def page_hydro():
    p = st.session_state.p
    hero("🌊 Hydrologie & crue de projet",
         "Ajustement de Gumbel, laminage simplifié, dimensionnement de l'évacuateur et de la revanche")
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.markdown("#### 📥 Débits maximaux annuels")
        edited = st.data_editor(
            st.session_state.crues, num_rows="dynamic", key="ed_crues", hide_index=True,
            column_config={"Année": st.column_config.NumberColumn(format="%d"),
                           "Qmax (m³/s)": st.column_config.NumberColumn(format="%.1f", min_value=0.0)})
        if st.button("💾 Appliquer le tableau", key="btn_crues"):
            st.session_state.crues = edited.reset_index(drop=True)
            st.session_state.pop("ed_crues", None)
            st.rerun()
        st.markdown("#### ⚙️ Paramètres")
        c1, c2 = st.columns(2)
        num("Période de retour T (ans)", "T", 10, 100000, 10, "%.0f", container=c1)
        num("Hauteur d'eau normale (m)", "h_normal", 5, 300, 0.5, container=c2)
        num("Longueur du déversoir (m)", "L_dev", 5, 500, 1, container=c1)
        num("Coef. de débit C", "C_dev", 1.0, 2.5, 0.05, container=c2,
            help="Q = C · L · H^1,5 (SI) - seuil épais ≈ 1,7 ; seuil Creager ≈ 2,1-2,2")
        num("Écrêtement par la retenue (%)", "atten", 0, 80, 1, "%.0f", container=c1)
        num("Revanche minimale (m)", "fb_min", 0.0, 5.0, 0.1, container=c2)
        num("Vitesse du vent (km/h)", "vent", 10, 250, 5, "%.0f", container=c1)
        num("Fetch (km)", "fetch", 0.1, 100, 0.5, container=c2)
    hydro, q = get_hydro()
    with col_b:
        k = st.columns(3)
        k[0].metric("Q de projet", f"{hydro['QT']:.0f} m³/s")
        k[1].metric("Q évacué (laminé)", f"{hydro['Qout']:.0f} m³/s")
        k[2].metric("Charge sur déversoir", f"{hydro['Hs']:.2f} m")
        k = st.columns(3)
        k[0].metric("Hauteur de vagues", f"{hydro['hv']:.2f} m")
        k[1].metric("Revanche retenue", f"{hydro['Fb']:.2f} m")
        k[2].metric("Hauteur requise", f"{hydro['H_req']:.2f} m")

        qs = np.sort(q)
        N = len(qs)
        Pnon = np.arange(1, N + 1) / (N + 1)
        Temp = 1 / (1 - Pnon)
        Ts = np.logspace(math.log10(1.2), math.log10(max(p["T"] * 3, 100)), 120)
        Qs = [gumbel_quantile(hydro["u"], hydro["alpha"], t) for t in Ts]
        f = go.Figure()
        f.add_trace(go.Scatter(x=Ts, y=Qs, mode="lines", name="Loi de Gumbel",
                               line=dict(color="#22d3ee", width=3)))
        f.add_trace(go.Scatter(x=Temp, y=qs, mode="markers", name="Observations (Weibull)",
                               marker=dict(size=10, color="#f472b6", line=dict(color="white", width=1))))
        f.add_trace(go.Scatter(x=[p["T"]], y=[hydro["QT"]], mode="markers", name="Crue de projet",
                               marker=dict(size=17, symbol="star", color="#fbbf24", line=dict(color="white", width=2))))
        f.update_xaxes(type="log", title="Période de retour (ans)")
        f.update_yaxes(title="Débit (m³/s)")
        st.plotly_chart(style_fig(f, 360, "Courbe de fréquence des crues"), key="h_gumbel")

        Hh = np.linspace(0, max(hydro["Hs"] * 2, 3), 80)
        Qd = p["C_dev"] * p["L_dev"] * Hh ** 1.5
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=Hh, y=Qd, mode="lines", fill="tozeroy", name="Capacité du déversoir",
                                line=dict(color="#a78bfa", width=3), fillcolor="rgba(167,139,250,.2)"))
        f2.add_trace(go.Scatter(x=[hydro["Hs"]], y=[hydro["Qout"]], mode="markers", name="Point de fonctionnement",
                                marker=dict(size=15, color="#fbbf24", symbol="star", line=dict(color="white", width=2))))
        f2.update_xaxes(title="Charge H (m)")
        f2.update_yaxes(title="Débit (m³/s)")
        st.plotly_chart(style_fig(f2, 330, "Courbe de tarage de l'évacuateur"), key="h_rating")


def page_dim():
    p = st.session_state.p
    hero("🧱 Dimensionnement & stabilité",
         "Barrage-poids : glissement, renversement, contraintes, sous-pressions, vase et séisme")
    hydro, _ = get_hydro()

    with st.expander("⚙️ Paramètres de calcul", expanded=True):
        t1, t2, t3, t4 = st.tabs(["📐 Géométrie", "🪨 Matériaux & fondation", "🌧️ Actions", "💰 Coût"])
        with t1:
            c1, c2, c3 = st.columns(3)
            with c1:
                p["H_auto"] = st.checkbox("Hauteur automatique (retenue + crue + revanche)",
                                          value=p["H_auto"], key="w_H_auto")
                if p["H_auto"]:
                    st.metric("Hauteur retenue", f"{get_H(p, hydro):.1f} m")
                else:
                    num("Hauteur du barrage H (m)", "H", 5, 300, 0.5)
            with c2:
                num("Largeur en crête a (m)", "a", 2, 30, 0.5,
                    help="Indicatif : a ≈ max(3 m ; H/10)")
                num("Fruit amont n (H:V)", "n", 0.0, 0.6, 0.01)
            with c3:
                num("Fruit aval m (H:V)", "m", 0.30, 1.60, 0.01,
                    help="Pente du parement aval (horizontal pour 1 vertical)")
        with t2:
            c1, c2, c3 = st.columns(3)
            num("Poids volumique béton (kN/m³)", "gamma_c", 20, 27, 0.1, container=c1)
            num("Angle de frottement φ à l'interface (°)", "phi", 20, 60, 1, "%.0f", container=c2)
            num("Cohésion c (kPa)", "coh", 0, 2000, 10, "%.0f", container=c3)
            num("Contrainte admissible fondation (kPa)", "sig_adm", 300, 20000, 100, "%.0f", container=c1)
            num("Poids volumique eau (kN/m³)", "gamma_w", 9.5, 10.5, 0.01, container=c2)
        with t3:
            c1, c2, c3 = st.columns(3)
            num("Niveau aval normal (m)", "ht_n", 0, 50, 0.5, container=c1)
            num("Niveau aval en crue (m)", "ht_f", 0, 80, 0.5, container=c2)
            num("Efficacité du drainage (0-1)", "eff_drain", 0.0, 0.95, 0.01, container=c3,
                help="0 = pas de drains ; 0,67 = pression réduite au tiers de la charge")
            num("Position des drains (fraction de B)", "drain_pos", 0.02, 0.5, 0.01, container=c1)
            num("Hauteur de vase (m)", "hs", 0, 60, 0.5, container=c2)
            num("γ' de la vase (kN/m³)", "gamma_s", 4, 14, 0.5, container=c3)
            num("φ de la vase (°)", "phi_s", 5, 45, 1, "%.0f", container=c1)
            num("Coefficient sismique horizontal kh", "kh", 0.0, 0.5, 0.01, container=c2)
        with t4:
            c1, c2 = st.columns(2)
            num("Longueur en crête (m)", "L_crete", 20, 5000, 10, "%.0f", container=c1)
            num("Coût du béton (€/m³)", "cout_beton", 20, 1000, 5, "%.0f", container=c2)

    H = get_H(p, hydro)
    cases = run_cases(p, hydro, H)
    rn = cases["Normal"]

    m_min = find_min_m(p, hydro, H)
    m_rec = None if m_min is None else min(round(math.ceil(m_min * 1.05 / 0.05) * 0.05, 2), 1.6)
    st.markdown("### 🎯 Dimensionnement automatique")
    c = st.columns([1, 1, 1, 1.4])
    c[0].metric("Fruit aval actuel", f"{p['m']:.2f}")
    c[1].metric("Fruit minimal conforme", "—" if m_min is None else f"{m_min:.2f}")
    c[2].metric("Fruit recommandé (+5 %)", "—" if m_rec is None else f"{m_rec:.2f}")
    with c[3]:
        st.write("")
        if m_rec is not None:
            st.button("✅ Appliquer le fruit recommandé", on_click=apply_m, args=(m_rec,))
        else:
            st.error("Aucun fruit ≤ 1,60 ne convient : augmentez la crête, le drainage ou revoyez les hypothèses.")

    vol = p["a"] * H + 0.5 * (p["n"] + p["m"]) * H * H
    c = st.columns(4)
    c[0].metric("Hauteur H", f"{H:.1f} m")
    c[1].metric("Base B", f"{rn['B']:.1f} m")
    c[2].metric("Section", f"{vol:,.0f} m²/ml".replace(",", " "))
    c[3].metric("Coût estimé béton", f"{vol * p['L_crete'] * p['cout_beton'] / 1e6:.1f} M€")
    st.caption("Volume = section × longueur en crête (vallée supposée rectangulaire : estimation majorante).")

    tabs = st.tabs([f"💧 {k}" for k in cases])
    for tab, (name, r) in zip(tabs, cases.items()):
        with tab:
            crit = CRITERIA[name]
            lc, rc = st.columns([3, 2])
            with lc:
                st.plotly_chart(fig_section(r, p), key=f"sec_{name}")
            with rc:
                rows = [
                    ["Glissement (FS)", f"{r['fs_s']:.2f}", f"≥ {crit['fs_s']:.2f}", badge(r["checks"]["s"], "OK", "NON")],
                    ["Renversement (FS)", f"{r['fs_o']:.2f}", f"≥ {crit['fs_o']:.2f}", badge(r["checks"]["o"], "OK", "NON")],
                    [f"Excentricité ({crit['label']})", f"{r['e']:.2f} m", f"≤ {B_lim(r, crit):.2f} m", badge(r["checks"]["e"], "OK", "NON")],
                    ["σ max base", f"{max(r['sig_toe'], r['sig_heel']):.0f} kPa", f"≤ {r['sig_lim']:.0f} kPa", badge(r["checks"]["sig"], "OK", "NON")],
                    ["σ pied aval / talon amont", f"{r['sig_toe']:.0f} / {r['sig_heel']:.0f} kPa", "σ talon ≥ 0 idéal", badge(r["sig_heel"] > 0.5, "Compression", "Décompression", )],
                ]
                html_table(["Vérification", "Valeur", "Critère", "Statut"], rows)
                st.markdown(f"**Verdict {name} : **" + badge(r["ok_all"]), unsafe_allow_html=True)
                k = st.columns(2)
                k[0].metric("ΣV (kN/m)", f"{r['sumV']:.0f}")
                k[1].metric("ΣH (kN/m)", f"{r['sumH']:.0f}")
                k[0].metric("Sous-pression U", f"{r['U']:.0f} kN/m")
                k[1].metric("Marge gouvernante", f"{r['marge']:.2f}")
            st.plotly_chart(fig_base(r, p), key=f"base_{name}")
            tbl = pd.DataFrame(r["rows"], columns=["Action", "Effort (kN/m)", "Bras / toe (m)", "Moment (kN·m/m)", "Type"])
            st.dataframe(tbl, hide_index=True,
                         column_config={"Effort (kN/m)": st.column_config.NumberColumn(format="%.1f"),
                                        "Bras / toe (m)": st.column_config.NumberColumn(format="%.2f"),
                                        "Moment (kN·m/m)": st.column_config.NumberColumn(format="%.0f")})

    st.markdown("### 📊 Sensibilité au fruit aval")
    ms = np.arange(0.40, 1.41, 0.02)
    f = go.Figure()
    for name in CRITERIA:
        vals = []
        for mm in ms:
            hf = min(p["h_normal"] + hydro["Hs"], H)
            args = {"Normal": (p["h_normal"], p["ht_n"], 0.0), "Crue": (hf, p["ht_f"], 0.0),
                    "Séisme": (p["h_normal"], p["ht_n"], p["kh"])}[name]
            vals.append(analyse_case(p, H, p["a"], p["n"], float(mm), *args, name)["marge"])
        f.add_trace(go.Scatter(x=ms, y=vals, mode="lines", name=name, line=dict(width=3, color=CASE_COLORS[name])))
    f.add_hline(y=1.0, line=dict(color="white", dash="dash"), annotation_text="Limite de conformité")
    f.add_vline(x=p["m"], line=dict(color="#fbbf24", dash="dot"), annotation_text="m actuel")
    f.update_xaxes(title="Fruit aval m (H:V)")
    f.update_yaxes(title="Marge gouvernante (≥ 1 = conforme)")
    st.plotly_chart(style_fig(f, 400, "Marge de sécurité normalisée selon le fruit aval"), key="sens")


def B_lim(r, crit):
    return r["B"] * crit["e"]


def page_collecte():
    hero("📡 Collecte des données", "Auscultation : piézomètres, déplacements, fuites, niveau de retenue, température, pluie")
    df = st.session_state.data
    c = st.columns(4)
    c[0].metric("Mesures", f"{len(df):,}".replace(",", " "))
    c[1].metric("Capteurs", df["Capteur"].nunique() if not df.empty else 0)
    c[2].metric("Types", df["Type"].nunique() if not df.empty else 0)
    c[3].metric("Dernière mesure", df["Date"].max().strftime("%d/%m/%Y") if not df.empty else "—")

    t1, t2, t3 = st.tabs(["✍️ Saisie manuelle", "📂 Import / export CSV", "🗃️ Tableau éditable"])
    with t1:
        c1, c2, c3 = st.columns(3)
        types = list(UNITS.keys()) + ["Autre"]
        with c1:
            d = st.date_input("Date", date.today(), key="in_date")
            tm = st.time_input("Heure", datetime.now().time().replace(second=0, microsecond=0), key="in_time")
        with c2:
            typ = st.selectbox("Type de mesure", types, key="in_type")
            existing = sorted(df.loc[df["Type"] == typ, "Capteur"].unique()) if not df.empty else []
            choice = st.selectbox("Capteur", existing + ["➕ Nouveau capteur"], key="in_cap")
            cap = st.text_input("Nom du nouveau capteur", key="in_newcap") if choice.startswith("➕") else choice
        with c3:
            val = st.number_input("Valeur", value=0.0, format="%.3f", key="in_val")
            st.caption(f"Unité : {UNITS.get(typ, '')}")
            oper = st.text_input("Opérateur", key="in_oper")
        obs = st.text_input("Observation", key="in_obs")
        if st.button("➕ Enregistrer la mesure"):
            if not cap or not cap.strip():
                st.error("Indiquez le nom du capteur.")
            else:
                new = pd.DataFrame([{"Date": datetime.combine(d, tm), "Capteur": cap.strip(), "Type": typ,
                                     "Valeur": val, "Unité": UNITS.get(typ, ""), "Opérateur": oper,
                                     "Observation": obs}])
                st.session_state.data = clean_df(pd.concat([st.session_state.data, new], ignore_index=True))
                st.toast("Mesure enregistrée ✅")
                st.rerun()
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Importer")
            up = st.file_uploader("Fichier CSV (colonnes : Date, Capteur, Type, Valeur, …)", type=["csv"])
            if up is not None and st.button("📥 Fusionner avec les données"):
                try:
                    raw = pd.read_csv(up, sep=None, engine="python")
                    new = clean_df(raw)
                    st.session_state.data = clean_df(pd.concat([st.session_state.data, new], ignore_index=True))
                    st.toast(f"{len(new)} mesures importées ✅")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Import impossible : {exc}")
            tpl = pd.DataFrame([{"Date": "2026-01-15 08:00", "Capteur": "PZ-1", "Type": "Piézomètre",
                                 "Valeur": 7.2, "Unité": "m", "Opérateur": "Jean", "Observation": ""}])
            st.download_button("📄 Modèle CSV", tpl.to_csv(index=False).encode("utf-8"), "modele_mesures.csv", "text/csv")
        with c2:
            st.markdown("##### Exporter / réinitialiser")
            st.download_button("⬇️ Télécharger toutes les mesures", df.to_csv(index=False).encode("utf-8"),
                               "mesures_barrage.csv", "text/csv")
            if st.button("🎲 Recharger les données de démonstration"):
                st.session_state.data = make_demo_data()
                st.rerun()
            if st.button("🗑️ Vider toutes les mesures"):
                st.session_state.data = clean_df(pd.DataFrame(columns=DATA_COLS))
                st.rerun()
    with t3:
        st.caption("Modifiez, ajoutez ou supprimez des lignes puis cliquez sur « Enregistrer ».")
        view = df.sort_values("Date", ascending=False).reset_index(drop=True)
        edited = st.data_editor(
            view, num_rows="dynamic", key="ed_data", hide_index=True, height=420,
            column_config={
                "Date": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm"),
                "Type": st.column_config.SelectboxColumn(options=types),
                "Valeur": st.column_config.NumberColumn(format="%.3f"),
            })
        if st.button("💾 Enregistrer les modifications", key="btn_data"):
            st.session_state.data = clean_df(edited)
            st.session_state.pop("ed_data", None)
            st.rerun()


def page_analyse():
    hero("📈 Analyse & surveillance", "Séries temporelles, seuils, anomalies, tendances et corrélations")
    df = st.session_state.data
    if df.empty:
        st.info("Aucune donnée : ajoutez des mesures ou rechargez la démo dans « Collecte des données ».")
        return
    with st.expander("⚙️ Seuils d'alerte et d'alarme (modifiables)"):
        st.session_state.seuils = st.data_editor(
            st.session_state.seuils, hide_index=True, key="ed_seuils", disabled=["Type", "Unité"],
            column_config={"Seuil d'alerte": st.column_config.NumberColumn(format="%.2f"),
                           "Seuil d'alarme": st.column_config.NumberColumn(format="%.2f")})
    seuils = st.session_state.seuils

    f1, f2, f3 = st.columns([1.2, 1.6, 1.4])
    all_types = sorted(df["Type"].unique())
    sel_types = f1.multiselect("Types", all_types, default=all_types[:1], key="f_types")
    caps_avail = sorted(df.loc[df["Type"].isin(sel_types), "Capteur"].unique())
    sel_caps = f2.multiselect("Capteurs", caps_avail, default=caps_avail, key="f_caps")
    dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
    rng = f3.date_input("Période", (dmin, dmax), min_value=dmin, max_value=dmax, key="f_range")
    d0, d1 = (rng[0], rng[1]) if isinstance(rng, (list, tuple)) and len(rng) == 2 else (dmin, dmax)
    dff = df[(df["Capteur"].isin(sel_caps)) & (df["Date"].dt.date >= d0) & (df["Date"].dt.date <= d1)]
    if dff.empty:
        st.warning("Aucune mesure pour cette sélection.")
        return

    o1, o2 = st.columns(2)
    win = o1.slider("Moyenne mobile (jours, 0 = désactivée)", 0, 60, 7, key="f_win")
    show_out = o2.checkbox("Détecter les anomalies (MAD robuste)", value=True, key="f_out")

    fig = go.Figure()
    n_out = 0
    for i, (cap, g) in enumerate(dff.groupby("Capteur")):
        g = g.sort_values("Date")
        col = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=g["Date"], y=g["Valeur"], mode="lines", name=cap,
                                 line=dict(color=col, width=2), opacity=0.9 if win == 0 else 0.45))
        if win > 0:
            ma = g.set_index("Date")["Valeur"].rolling(f"{win}D", min_periods=1).mean()
            fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode="lines", name=f"{cap} (moy. {win} j)",
                                     line=dict(color=col, width=3.5, dash="solid")))
        if show_out and len(g) >= 10:
            mask = robust_outliers(g["Valeur"].values)
            n_out += int(mask.sum())
            if mask.any():
                fig.add_trace(go.Scatter(x=g["Date"][mask], y=g["Valeur"][mask], mode="markers",
                                         name=f"Anomalies {cap}", marker=dict(symbol="x", size=11, color="#ff4d6d",
                                                                              line=dict(width=2, color="white"))))
    types_in = dff["Type"].unique()
    if len(types_in) == 1:
        row = seuils[seuils["Type"] == types_in[0]]
        if not row.empty:
            fig.add_hline(y=float(row.iloc[0]["Seuil d'alerte"]), line=dict(color="#fbbf24", dash="dash"),
                          annotation_text="Alerte", annotation_font_color="#fbbf24")
            fig.add_hline(y=float(row.iloc[0]["Seuil d'alarme"]), line=dict(color="#fb7185", dash="dash"),
                          annotation_text="Alarme", annotation_font_color="#fb7185")
    fig.update_yaxes(title=" / ".join(sorted({UNITS.get(t, "") for t in types_in})))
    st.plotly_chart(style_fig(fig, 470, "Évolution temporelle"), key="an_ts")

    summ = sensor_summary(dff, seuils)
    c = st.columns(4)
    c[0].metric("Mesures filtrées", f"{len(dff):,}".replace(",", " "))
    c[1].metric("Capteurs", len(summ))
    c[2].metric("Anomalies détectées", n_out)
    c[3].metric("Capteurs hors seuil", int((summ["Statut"] != "Normal").sum()))

    st.markdown("#### 📋 Synthèse par capteur")
    rows = []
    for _, r in summ.iterrows():
        arrow = "↗" if r["Tendance"] > 1e-4 else ("↘" if r["Tendance"] < -1e-4 else "→")
        rows.append([r["Capteur"], r["Type"], r["N"], f"{r['Min']:.2f}", f"{r['Moyenne']:.2f}", f"{r['Max']:.2f}",
                     f"{r['Ecart']:.2f}", f"{r['Dernière']:.2f} {r['Unité']}", r["Date"],
                     f"{arrow} {r['Tendance']:+.4f} {r['Unité']}/j", status_badge(r["Statut"])])
    html_table(["Capteur", "Type", "N", "Min", "Moy.", "Max", "σ", "Dernière valeur", "Date", "Tendance", "Statut"], rows)

    t1, t2, t3 = st.tabs(["🔗 Corrélations", "🎯 Relation entre 2 capteurs", "📊 Distributions"])
    daily = df.assign(day=df["Date"].dt.floor("D")).pivot_table(index="day", columns="Capteur", values="Valeur", aggfunc="mean")
    with t1:
        cols = [c for c in daily.columns if c in sel_caps] if len(sel_caps) >= 2 else list(daily.columns)
        if len(cols) >= 2:
            corr = daily[cols].corr()
            hm = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                colorscale=[[0, "#f472b6"], [0.5, "#0b132b"], [1, "#22d3ee"]],
                text=np.round(corr.values, 2), texttemplate="%{text}", colorbar=dict(title="r")))
            st.plotly_chart(style_fig(hm, 480, "Matrice de corrélation (moyennes journalières)"), key="an_corr")
        else:
            st.info("Sélectionnez au moins 2 capteurs.")
    with t2:
        if daily.shape[1] >= 2:
            ca, cb = st.columns(2)
            xs_ = ca.selectbox("Capteur X (cause)", list(daily.columns), index=0, key="rel_x")
            ys_ = cb.selectbox("Capteur Y (réponse)", list(daily.columns), index=min(1, daily.shape[1] - 1), key="rel_y")
            pair = daily[[xs_, ys_]].dropna() if xs_ != ys_ else pd.DataFrame()
            if len(pair) >= 5:
                k, b0 = np.polyfit(pair[xs_], pair[ys_], 1)
                r2 = float(np.corrcoef(pair[xs_], pair[ys_])[0, 1] ** 2)
                xx = np.linspace(pair[xs_].min(), pair[xs_].max(), 50)
                sc = go.Figure()
                sc.add_trace(go.Scatter(x=pair[xs_], y=pair[ys_], mode="markers", name="Jours",
                                        marker=dict(size=7, color=np.arange(len(pair)), colorscale="Turbo",
                                                    showscale=True, colorbar=dict(title="Temps"), opacity=0.85)))
                sc.add_trace(go.Scatter(x=xx, y=k * xx + b0, mode="lines", name=f"y = {k:.3f}x + {b0:.2f}  (R²={r2:.2f})",
                                        line=dict(color="#fbbf24", width=3)))
                sc.update_xaxes(title=xs_)
                sc.update_yaxes(title=ys_)
                st.plotly_chart(style_fig(sc, 440, f"{ys_} en fonction de {xs_}"), key="an_scatter")
            else:
                st.info("Choisissez deux capteurs différents avec suffisamment de jours en commun.")
    with t3:
        bx = go.Figure()
        for i, (cap, g) in enumerate(dff.groupby("Capteur")):
            bx.add_trace(go.Violin(y=g["Valeur"], name=cap, box_visible=True, meanline_visible=True,
                                   line_color=PALETTE[i % len(PALETTE)], opacity=0.8))
        st.plotly_chart(style_fig(bx, 420, "Distribution des mesures"), key="an_dist")


def build_report(p, hydro, H, cases):
    lines = [
        f"# Rapport de pré-dimensionnement - {p['nom']}",
        f"_Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}_", "",
        "## 1. Hydrologie",
        f"- Période de retour : {p['T']:.0f} ans",
        f"- Débit de projet (Gumbel) : {hydro['QT']:.0f} m³/s ; débit évacué : {hydro['Qout']:.0f} m³/s",
        f"- Évacuateur : L = {p['L_dev']:.0f} m, C = {p['C_dev']:.2f} -> charge {hydro['Hs']:.2f} m",
        f"- Hauteur de vagues : {hydro['hv']:.2f} m ; revanche : {hydro['Fb']:.2f} m", "",
        "## 2. Géométrie",
        f"- Hauteur H = {H:.1f} m ; crête a = {p['a']:.1f} m ; fruit amont n = {p['n']:.2f} ; fruit aval m = {p['m']:.2f}",
        f"- Base B = {cases['Normal']['B']:.1f} m ; longueur en crête = {p['L_crete']:.0f} m", "",
        "## 3. Vérifications de stabilité",
        "| Cas | FS glissement | FS renversement | Excentricité (m) | σ max (kPa) | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for k, r in cases.items():
        lines.append(f"| {k} | {r['fs_s']:.2f} | {r['fs_o']:.2f} | {r['e']:.2f} | "
                     f"{max(r['sig_toe'], r['sig_heel']):.0f} | {'CONFORME' if r['ok_all'] else 'NON CONFORME'} |")
    vol = p["a"] * H + 0.5 * (p["n"] + p["m"]) * H * H
    lines += ["", "## 4. Quantités",
              f"- Section : {vol:.0f} m²/ml ; volume béton ≈ {vol * p['L_crete']:.0f} m³ ; "
              f"coût ≈ {vol * p['L_crete'] * p['cout_beton'] / 1e6:.1f} M€"]
    df = st.session_state.data
    if not df.empty:
        s = sensor_summary(df, st.session_state.seuils)
        lines += ["", "## 5. Auscultation (dernier état)", "| Capteur | Type | Dernière valeur | Tendance/j | Statut |", "|---|---|---|---|---|"]
        for _, r in s.iterrows():
            lines.append(f"| {r['Capteur']} | {r['Type']} | {r['Dernière']:.2f} {r['Unité']} | {r['Tendance']:+.4f} | {r['Statut']} |")
    lines += ["", "> Outil de pré-dimensionnement : à valider par un ingénieur qualifié selon les normes en vigueur."]
    return "\n".join(lines)


def page_rapport():
    p = st.session_state.p
    hero("📄 Rapport & export", "Synthèse téléchargeable du dimensionnement et de l'auscultation")
    hydro, _ = get_hydro()
    H = get_H(p, hydro)
    cases = run_cases(p, hydro, H)
    md = build_report(p, hydro, H, cases)
    c = st.columns(3)
    c[0].download_button("⬇️ Rapport (Markdown)", md.encode("utf-8"), "rapport_barrage.md", "text/markdown")
    res = pd.DataFrame([{"Cas": k, "FS glissement": r["fs_s"], "FS renversement": r["fs_o"],
                         "Excentricité (m)": r["e"], "σ pied (kPa)": r["sig_toe"], "σ talon (kPa)": r["sig_heel"],
                         "Conforme": r["ok_all"]} for k, r in cases.items()])
    c[1].download_button("⬇️ Résultats (CSV)", res.to_csv(index=False).encode("utf-8"), "resultats_stabilite.csv", "text/csv")
    c[2].download_button("⬇️ Paramètres (JSON)", pd.Series(p).to_json(indent=2, force_ascii=False).encode("utf-8"),
                         "parametres.json", "application/json")
    st.markdown(md)


# =============================================================================
# 7. NAVIGATION
# =============================================================================
def main():
    init_state()
    with st.sidebar:
        st.markdown("<p class='side-title'>🌊 DamLab</p>", unsafe_allow_html=True)
        st.caption("Barrage-poids · Collecte & analyse")
        st.session_state.p["nom"] = st.text_input("Nom du projet", st.session_state.p["nom"], key="w_nom")
        page = st.radio(
            "Navigation",
            ["🏠 Tableau de bord", "🌊 Hydrologie & crue", "🧱 Dimensionnement & stabilité",
             "📡 Collecte des données", "📈 Analyse & surveillance", "📄 Rapport & export"],
        )
        st.divider()
        st.caption("Pré-dimensionnement pédagogique - à valider par un ingénieur.")
    {
        "🏠 Tableau de bord": page_dashboard,
        "🌊 Hydrologie & crue": page_hydro,
        "🧱 Dimensionnement & stabilité": page_dim,
        "📡 Collecte des données": page_collecte,
        "📈 Analyse & surveillance": page_analyse,
        "📄 Rapport & export": page_rapport,
    }[page]()


main()




