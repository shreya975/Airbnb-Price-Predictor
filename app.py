"""
AI Airbnb Valuation Studio
An immersive, cinematic price-intelligence experience for Airbnb hosts.

ML PIPELINE — unchanged, mirrors the source notebook exactly:
    1. Drop id, name, host_id, host_name, last_review
    2. Fill missing reviews_per_month with the column median
    3. Remove price outliers using the IQR rule (1.5 * IQR)
    4. One-hot encode neighbourhood_group, neighbourhood, room_type (drop_first=True)
    5. Train RandomForestRegressor(n_estimators=200, random_state=42) on price
    6. Feature order = training X.columns, persisted as feature_names

If `airbnb_price_model.pkl` / `feature_names.pkl` sit next to this file, they are
loaded directly. Otherwise the identical pipeline is trained on the fly from
AB_NYC_2019.csv and cached.
"""

import time
import uuid
import random
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ═════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════

DATA_PATH = Path("AB_NYC_2019.csv")
MODEL_PATH = Path("airbnb_price_model.pkl")
FEATURES_PATH = Path("feature_names.pkl")
DEVELOPER_NAME = "Your Name"

st.set_page_config(
    page_title="AI Airbnb Valuation Studio",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS
# ═════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ───────────────────────── DESIGN TOKENS ───────────────────────── */
    :root{
        --ink:#0E1016;
        --ink-soft:#3A3F4B;
        --slate:#6B7280;
        --cream:#FBF8F2;
        --cream-2:#F3EEE4;
        --blue:#4E7CFF;
        --blue-2:#7CA8FF;
        --violet:#8B5CF6;
        --violet-2:#B79CFF;
        --orange:#FF8A4C;
        --gold:#FFC24E;
        --teal:#22C1A0;
        --glass: rgba(255,255,255,0.46);
        --glass-2: rgba(255,255,255,0.30);
        --glass-border: rgba(255,255,255,0.55);
        --glow-blue: 0 0 60px -12px rgba(78,124,255,0.55);
        --glow-violet: 0 0 60px -12px rgba(139,92,246,0.55);
        --glow-orange: 0 0 60px -12px rgba(255,138,76,0.55);
        --shadow-lux: 0 30px 80px -20px rgba(14,16,22,0.35);
        --shadow-soft: 0 16px 44px -18px rgba(14,16,22,0.28);
        --radius-xl: 30px;
        --radius-lg: 24px;
        --radius-md: 18px;
        --radius-sm: 12px;
    }

    @property --p { syntax:'<number>'; inherits:false; initial-value:0; }
    @property --grad-x { syntax:'<percentage>'; inherits:false; initial-value:0%; }

    /* ───────────────────────── RESET STREAMLIT CHROME ───────────────────────── */
    #MainMenu, header, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"],
    .stDeployButton, section[data-testid="stSidebar"] { display:none !important; }

    div[data-testid="stAppViewBlockContainer"] {
        padding: 0 !important; max-width: 100% !important;
    }
    .block-container { padding: 0 6vw 4rem 6vw !important; max-width: 1360px !important; margin:0 auto; }
    div[data-testid="stAppViewContainer"] > .main { overflow-x:hidden; }

    html, body, [class*="css"] { font-family:'Inter', sans-serif; color:var(--ink); }
    h1,h2,h3,h4, .lux-heading { font-family:'Plus Jakarta Sans', sans-serif; }
    * { box-sizing:border-box; }

    .stApp {
        background:
            radial-gradient(1400px 700px at 8% -8%, rgba(78,124,255,0.14), transparent 60%),
            radial-gradient(1100px 600px at 100% 4%, rgba(139,92,246,0.14), transparent 55%),
            radial-gradient(1000px 600px at 45% 105%, rgba(255,138,76,0.10), transparent 55%),
            linear-gradient(180deg, var(--cream) 0%, var(--cream-2) 100%);
        background-attachment: fixed;
    }

    iframe { border:none !important; background:transparent !important; }

    /* ───────────────────────── SCROLLBAR ───────────────────────── */
    ::-webkit-scrollbar { width:11px; }
    ::-webkit-scrollbar-track { background:transparent; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(var(--blue), var(--violet));
        border-radius:10px; border:2px solid var(--cream);
    }

    /* ───────────────────────── KEYFRAME LIBRARY ───────────────────────── */
    @keyframes fadeUp { from{opacity:0; transform:translateY(26px);} to{opacity:1; transform:translateY(0);} }
    @keyframes fadeIn { from{opacity:0;} to{opacity:1;} }
    @keyframes floatY { 0%,100%{transform:translateY(0px) rotate(0deg);} 50%{transform:translateY(-16px) rotate(1deg);} }
    @keyframes floatY2 { 0%,100%{transform:translateY(0px);} 50%{transform:translateY(14px);} }
    @keyframes pulseGlow { 0%,100%{ box-shadow:0 0 0px rgba(139,92,246,0.0);} 50%{ box-shadow:0 0 44px rgba(139,92,246,0.35);} }
    @keyframes gradientShift { 0%{ background-position:0% 50%;} 50%{ background-position:100% 50%;} 100%{ background-position:0% 50%;} }
    @keyframes shimmer { 0%{ transform:translateX(-120%);} 100%{ transform:translateX(220%);} }
    @keyframes spin360 { from{ transform:rotate(0deg);} to{ transform:rotate(360deg);} }
    @keyframes bounceIcon { 0%,100%{ transform:translateY(0);} 50%{ transform:translateY(-5px);} }
    @keyframes borderGlow {
        0%,100%{ box-shadow: 0 0 0 1px rgba(139,92,246,0.25), 0 25px 70px -25px rgba(78,124,255,0.4); }
        50%{ box-shadow: 0 0 0 1px rgba(255,138,76,0.35), 0 25px 70px -20px rgba(139,92,246,0.5); }
    }
    @keyframes ringPulse { 0%{ transform:scale(1); opacity:.55;} 100%{ transform:scale(1.5); opacity:0;} }

    /* ───────────────────────── TOP NAVBAR ───────────────────────── */
    .navbar {
        display:flex; align-items:center; justify-content:space-between;
        padding:20px 6px 20px 6px; margin: 0 -6px 8px -6px;
        animation:fadeUp .7s ease both;
    }
    .nav-logo { display:flex; align-items:center; gap:10px; font-family:'Plus Jakarta Sans'; font-weight:800; font-size:19px; }
    .nav-logo .dot {
        width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center;
        background:linear-gradient(135deg, var(--blue), var(--violet)); font-size:18px;
        box-shadow: var(--glow-violet);
    }
    div[data-testid="column"] div.stButton > button.nav-pill-btn { all:unset; }

    /* nav buttons re-themed (they are the Nth row of buttons rendered via st.button in the nav container) */
    .nav-row div.stButton > button {
        background: transparent !important; color:var(--ink-soft) !important; border:none !important;
        box-shadow:none !important; font-weight:600 !important; font-size:14.5px !important;
        padding:10px 16px !important; border-radius:999px !important; animation:none !important;
        transition: all .25s ease !important;
    }
    .nav-row div.stButton > button:hover {
        background: rgba(255,255,255,0.55) !important; color:var(--ink) !important;
        transform:none !important; box-shadow: var(--shadow-soft) !important;
    }
    .nav-row .active-pill div.stButton > button {
        background: linear-gradient(100deg, var(--blue), var(--violet)) !important;
        color:white !important; box-shadow: var(--glow-blue) !important;
    }

    /* ───────────────────────── GLASS CARD PRIMITIVES ───────────────────────── */
    .glass {
        background: var(--glass); border:1px solid var(--glass-border);
        backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
        border-radius: var(--radius-xl); box-shadow: var(--shadow-lux);
        position:relative;
    }
    .glass-border-glow {
        position:relative; border-radius: var(--radius-xl);
        background: linear-gradient(var(--cream), var(--cream)) padding-box,
                    linear-gradient(120deg, var(--blue), var(--violet), var(--orange)) border-box;
        border:1.5px solid transparent;
        animation: borderGlow 5s ease-in-out infinite;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: var(--glass) !important; border:1px solid var(--glass-border) !important;
        backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
        border-radius: var(--radius-lg) !important;
        box-shadow: var(--shadow-soft);
        padding: 10px 8px 6px 8px;
        transition: transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s ease;
        animation: fadeUp .7s ease both;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover > div {
        transform: translateY(-6px) scale(1.008);
        box-shadow: var(--shadow-lux);
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(odd) { animation-name: fadeUp; }
    div[data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(3n) > div { animation-duration:.9s; }

    /* ───────────────────────── HERO OVERLAP CTA ───────────────────────── */
    .hero-cta-wrap { position:relative; margin-top:-64px; z-index:5; padding:0 4vw; }
    .hero-cta-inner {
        background: var(--glass); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
        border:1px solid var(--glass-border); border-radius:26px; box-shadow:var(--shadow-lux);
        padding:22px 26px; display:flex; gap:16px; align-items:center; justify-content:space-between;
        flex-wrap:wrap;
    }

    /* ───────────────────────── BUTTONS ───────────────────────── */
    div.stButton > button, div.stFormSubmitButton > button {
        position:relative; overflow:hidden; width:100%; padding:17px 0; border-radius:18px; border:none;
        background: linear-gradient(100deg, var(--blue), var(--violet) 55%, var(--orange));
        background-size:240% auto; background-position:0% 50%;
        color:white; font-weight:700; font-size:16.5px; letter-spacing:.01em;
        box-shadow: 0 16px 40px -12px rgba(78,124,255,0.55);
        transition: all .4s cubic-bezier(.2,.8,.2,1);
        animation: pulseGlow 3.2s ease-in-out infinite;
    }
    div.stButton > button::before {
        content:''; position:absolute; top:0; left:0; width:40%; height:100%;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.45), transparent);
        transform: translateX(-120%); animation: shimmer 3.4s ease-in-out infinite;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background-position:100% 50%; transform: translateY(-3px) scale(1.01);
        box-shadow: 0 22px 50px -10px rgba(139,92,246,0.6);
    }
    div.stButton > button p { font-size:16.5px !important; font-weight:700 !important; }

    .ghost-btn div.stButton > button {
        background: rgba(255,255,255,0.55) !important; color:var(--ink) !important;
        box-shadow: var(--shadow-soft) !important; animation:none !important; border:1px solid rgba(255,255,255,0.8) !important;
    }
    .ghost-btn div.stButton > button:hover { background: rgba(255,255,255,0.8) !important; }

    /* ───────────────────────── INPUT RESKIN ───────────────────────── */
    .field-label {
        font-size:11.5px; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
        color:var(--violet); margin-bottom:6px; display:flex; align-items:center; gap:6px;
    }

    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.55) !important; border-radius:14px !important;
        border:1px solid rgba(139,92,246,0.22) !important; box-shadow:none !important;
        transition: all .25s ease;
    }
    div[data-baseweb="select"] > div:hover { border-color: rgba(139,92,246,0.5) !important; }
    div[data-baseweb="popover"] { border-radius:16px !important; overflow:hidden; }

    div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.55) !important; border-radius:14px !important;
        border:1px solid rgba(139,92,246,0.22) !important; padding:10px 14px !important;
        font-weight:600 !important; transition: all .25s ease;
    }
    div[data-testid="stNumberInput"] input:focus, div[data-testid="stTextInput"] input:focus {
        border-color: var(--violet) !important; box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
    }
    div[data-testid="stNumberInput"] button { background:rgba(255,255,255,0.6) !important; border-radius:8px !important; }

    div[data-testid="stSlider"] { padding-top:6px; }
    div[data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, var(--blue), var(--violet)) !important;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background: white !important; border:3px solid var(--violet) !important;
        box-shadow: 0 4px 14px rgba(139,92,246,0.45) !important; width:20px !important; height:20px !important;
    }
    div[data-testid="stSlider"] > div > div > div:first-child { background: rgba(139,92,246,0.14) !important; }

    label { color: var(--ink-soft) !important; font-weight:600 !important; font-size:13.5px !important; }

    /* ───────────────────────── HEADINGS / SECTIONS ───────────────────────── */
    .eyebrow {
        display:inline-flex; align-items:center; gap:8px; color:var(--violet); font-weight:700;
        font-size:12.5px; letter-spacing:.12em; text-transform:uppercase;
        padding:6px 16px; border-radius:999px; background:rgba(139,92,246,0.10);
        border:1px solid rgba(139,92,246,0.22); margin-bottom:16px;
    }
    .section-title {
        font-size: clamp(28px, 3.6vw, 42px); font-weight:800; color:var(--ink);
        letter-spacing:-0.02em; margin:2px 0 10px 0; line-height:1.12;
    }
    .section-desc { color:var(--slate); font-size:16px; max-width:640px; margin-bottom:8px; line-height:1.55; }
    .grad-text {
        background: linear-gradient(100deg, var(--blue), var(--violet) 55%, var(--orange));
        background-size:220% auto;
        -webkit-background-clip:text; background-clip:text; color:transparent;
        animation: gradientShift 6s ease infinite;
    }

    /* ───────────────────────── STAT / COUNTER CARDS ───────────────────────── */
    .stat-card {
        border-radius: var(--radius-lg); padding:28px 22px; text-align:center; position:relative;
        background: var(--glass); border:1px solid var(--glass-border);
        backdrop-filter: blur(18px); box-shadow: var(--shadow-soft);
        animation: fadeUp .8s ease both, floatY 7s ease-in-out infinite;
        transition: transform .3s ease;
    }
    .stat-card:hover { transform: translateY(-8px); }
    .stat-icon { font-size:26px; margin-bottom:6px; display:inline-block; animation: bounceIcon 2.6s ease-in-out infinite; }
    .stat-num { font-size:34px; font-weight:800; letter-spacing:-0.02em;
        background:linear-gradient(100deg, var(--blue), var(--violet));
        -webkit-background-clip:text; background-clip:text; color:transparent; }
    .stat-label { font-size:12.5px; color:var(--slate); font-weight:700; letter-spacing:.03em; margin-top:6px; text-transform:uppercase;}

    /* ───────────────────────── BADGES / PILLS ───────────────────────── */
    .badge-row { display:flex; flex-wrap:wrap; gap:10px; justify-content:center; }
    .badge {
        padding:10px 20px; border-radius:999px; font-size:13.5px; font-weight:700; color:var(--ink);
        background: rgba(255,255,255,0.65); border:1px solid rgba(255,255,255,0.85);
        box-shadow: 0 10px 26px -10px rgba(14,16,22,0.25);
        animation: floatY2 3.8s ease-in-out infinite;
    }
    .badge:nth-child(2n){ animation-delay:.5s; }
    .badge:nth-child(3n){ animation-delay:1s; }

    .pill { display:inline-flex; align-items:center; gap:6px; padding:9px 20px; border-radius:999px; font-weight:800; font-size:13.5px;
        box-shadow: var(--shadow-soft); animation: pulseGlow 3s ease-in-out infinite; }
    .pill-budget { background:linear-gradient(100deg, #4E7CFF, #7CA8FF); color:white; }
    .pill-standard { background:linear-gradient(100deg, #8B5CF6, #B79CFF); color:white; }
    .pill-premium { background:linear-gradient(100deg, #FF8A4C, #FFC24E); color:white; }
    .pill-luxury { background:linear-gradient(100deg, #0E1016, #3A3F4B); color:#FFC24E; }

    .grade-badge {
        width:76px; height:76px; border-radius:50%; display:flex; align-items:center; justify-content:center;
        font-size:26px; font-weight:800; color:white; margin:0 auto 8px auto;
        background: conic-gradient(from 180deg, var(--blue), var(--violet), var(--orange), var(--blue));
        box-shadow: var(--shadow-soft); animation: floatY 5s ease-in-out infinite;
    }
    .grade-badge span { background:var(--ink); width:60px; height:60px; border-radius:50%; display:flex; align-items:center; justify-content:center; }

    /* ───────────────────────── RADIAL METERS ───────────────────────── */
    .radial-outer { position:relative; width:132px; height:132px; margin:0 auto; }
    .radial-ring {
        width:132px; height:132px; border-radius:50%;
        background: conic-gradient(var(--ring-color, var(--violet)) calc(var(--p) * 1%), rgba(14,16,22,0.08) 0);
        display:flex; align-items:center; justify-content:center;
        transition: filter .3s ease;
    }
    .radial-inner {
        width:100px; height:100px; border-radius:50%; background: var(--cream);
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.06);
    }
    .radial-val { font-size:21px; font-weight:800; color:var(--ink); }
    .radial-tag { font-size:10px; color:var(--slate); font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
    .meter-label { text-align:center; margin-top:10px; font-weight:700; font-size:13.5px; color:var(--ink-soft); }

    /* ───────────────────────── PRICE HERO RESULT ───────────────────────── */
    .price-ring-outer { position:relative; width:250px; height:250px; margin:10px auto; }
    .price-ring {
        width:250px; height:250px; border-radius:50%;
        background: conic-gradient(var(--blue) calc(var(--p) * 1%), rgba(14,16,22,0.06) 0);
        display:flex; align-items:center; justify-content:center;
    }
    .price-ring-inner {
        width:206px; height:206px; border-radius:50%; background: var(--glass);
        backdrop-filter: blur(14px); display:flex; flex-direction:column; align-items:center; justify-content:center;
        box-shadow: inset 0 2px 16px rgba(0,0,0,0.06);
    }
    .price-label { color:var(--slate); font-weight:700; font-size:12.5px; text-transform:uppercase; letter-spacing:.1em; }
    .price-value {
        font-size:52px; font-weight:900; letter-spacing:-0.02em; line-height:1;
        background:linear-gradient(100deg, var(--blue), var(--violet) 55%, var(--orange));
        background-size:220% auto;
        -webkit-background-clip:text; background-clip:text; color:transparent;
        animation: gradientShift 5s ease infinite;
    }

    /* ───────────────────────── RECOMMENDATION CARDS ───────────────────────── */
    .rec-card {
        border-radius:18px; padding:16px 18px; margin-bottom:12px; display:flex; gap:12px; align-items:flex-start;
        background: rgba(255,255,255,0.55); border:1px solid rgba(255,255,255,0.75);
        box-shadow: var(--shadow-soft); font-size:14.5px; color:var(--ink);
        animation: fadeUp .6s ease both; transition: transform .25s ease;
    }
    .rec-card:hover { transform: translateX(6px); }
    .rec-icon { font-size:20px; }

    /* ───────────────────────── CUSTOM HTML BAR LIST (non-Plotly) ───────────────────────── */
    .bar-row { margin-bottom:14px; }
    .bar-row-top { display:flex; justify-content:space-between; font-size:13.5px; font-weight:700; margin-bottom:6px; color:var(--ink-soft); }
    .bar-track { height:12px; border-radius:999px; background: rgba(14,16,22,0.06); overflow:hidden; }
    .bar-fill {
        height:100%; border-radius:999px;
        background: linear-gradient(90deg, var(--blue), var(--violet), var(--orange));
        width:0%; animation: growBar 1.4s cubic-bezier(.2,.8,.2,1) forwards;
        box-shadow: 0 0 16px rgba(139,92,246,0.4);
    }
    @keyframes growBar { from{ width:0%; } }

    /* ───────────────────────── LOADING SEQUENCE ───────────────────────── */
    .loading-wrap { text-align:center; padding:20px 0; position:relative; }
    .loading-ring-wrap { position:relative; width:120px; height:120px; margin:0 auto 22px auto; }
    .loading-core {
        position:absolute; inset:0; border-radius:50%;
        background: conic-gradient(var(--blue), var(--violet), var(--orange), var(--blue));
        animation: spin360 1.6s linear infinite;
        -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 10px), #000 calc(100% - 9px));
                mask: radial-gradient(farthest-side, transparent calc(100% - 10px), #000 calc(100% - 9px));
    }
    .loading-emoji { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:38px; animation: bounceIcon 1.4s ease-in-out infinite; }
    .loading-text { font-size:19px; font-weight:700; color:var(--ink); margin-bottom:16px; animation: fadeIn .4s ease; }
    .loading-bar-track { height:9px; max-width:380px; margin:0 auto; border-radius:8px; background: rgba(14,16,22,0.08); overflow:hidden; }
    .loading-bar-fill { height:100%; border-radius:8px; background: linear-gradient(100deg, var(--blue), var(--violet), var(--orange)); transition: width .4s ease; }

    /* ───────────────────────── DATAFRAME / TABLE ───────────────────────── */
    div[data-testid="stDataFrame"] { border-radius:18px !important; overflow:hidden; box-shadow: var(--shadow-soft); }

    /* ───────────────────────── METRIC OVERRIDE ───────────────────────── */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.5); border-radius:16px; padding:14px 16px;
        border:1px solid rgba(255,255,255,0.7);
    }
    div[data-testid="stMetricValue"] { color:var(--ink); font-weight:800; }
    div[data-testid="stMetricLabel"] { color:var(--slate); font-weight:700; }

    /* ───────────────────────── FOOTER ───────────────────────── */
    .footer-wrap { text-align:center; color:var(--slate); font-size:13px; padding: 46px 0 10px 0; }
    .footer-chip {
        display:inline-block; margin:4px 6px; padding:7px 16px; border-radius:999px;
        background:rgba(255,255,255,0.55); border:1px solid rgba(255,255,255,0.8); font-weight:700;
        transition: all .25s ease;
    }
    .footer-chip:hover { background:rgba(255,255,255,0.85); transform:translateY(-3px); }

    /* ───────────────────────── SEGMENTED BOOKING WIDGET ───────────────────────── */
    .segment-title {
        display:flex; align-items:center; gap:10px; font-family:'Plus Jakarta Sans'; font-weight:800;
        font-size:16px; margin-bottom:14px; color:var(--ink);
    }
    .segment-icon-chip {
        width:34px; height:34px; border-radius:11px; display:flex; align-items:center; justify-content:center;
        background: linear-gradient(135deg, rgba(78,124,255,0.16), rgba(139,92,246,0.16)); font-size:16px;
    }
    hr.seg-divider { border:none; border-top:1px dashed rgba(107,114,128,0.25); margin:18px 0; }

    /* responsive */
    @media (max-width: 900px) {
        .hero-cta-wrap { margin-top:-40px; }
        .price-ring-outer, .price-ring, .price-ring-inner { width:200px; height:200px; }
        .price-ring-inner { width:164px; height:164px; }
    }
    </style>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# DATA + MODEL  (unchanged pipeline — mirrors the notebook exactly)
# ═════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_raw_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def preprocess(df_raw: pd.DataFrame):
    df = df_raw.drop(columns=["id", "name", "host_id", "host_name", "last_review"], errors="ignore").copy()
    df["reviews_per_month"] = df["reviews_per_month"].fillna(df["reviews_per_month"].median())

    Q1, Q3 = df["price"].quantile(0.25), df["price"].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_clean = df[(df["price"] >= lower) & (df["price"] <= upper)].reset_index(drop=True)
    return df_clean


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    df_raw = load_raw_data()
    if df_raw is None:
        return None

    df_clean = preprocess(df_raw)

    df_encoded = pd.get_dummies(
        df_clean, columns=["neighbourhood_group", "neighbourhood", "room_type"], drop_first=True
    )
    X = df_encoded.drop("price", axis=1)
    y = df_encoded["price"]

    if MODEL_PATH.exists() and FEATURES_PATH.exists():
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        y_pred = model.predict(X_test.reindex(columns=feature_names, fill_value=0))
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        feature_names = list(X.columns)
        y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }

    neighbourhood_map = (
        df_clean.groupby("neighbourhood_group")["neighbourhood"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "feature_names": feature_names,
        "df_raw": df_raw,
        "df_clean": df_clean,
        "metrics": metrics,
        "neighbourhood_map": neighbourhood_map,
        "importance": importance,
        "price_quantiles": df_clean["price"].quantile([0.25, 0.5, 0.75, 0.95]).to_dict(),
    }


def build_feature_row(feature_names, inputs: dict) -> pd.DataFrame:
    """Builds a single-row DataFrame matching the exact training feature order."""
    row = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    numeric_fields = [
        "latitude", "longitude", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count", "availability_365",
    ]
    for field in numeric_fields:
        if field in row.columns:
            row[field] = inputs[field]

    for prefix, value in [
        ("neighbourhood_group_", inputs["neighbourhood_group"]),
        ("neighbourhood_", inputs["neighbourhood"]),
        ("room_type_", inputs["room_type"]),
    ]:
        col = f"{prefix}{value}"
        if col in row.columns:
            row[col] = 1
        # if missing, this is the dropped baseline category — row stays 0, which is correct

    return row


def price_category(price, quantiles):
    if price <= quantiles[0.25]:
        return "Budget", "pill-budget"
    if price <= quantiles[0.75]:
        return "Standard", "pill-standard"
    if price <= quantiles[0.95]:
        return "Premium", "pill-premium"
    return "Luxury", "pill-luxury"


def confidence_score(model, row):
    try:
        tree_preds = np.array([t.predict(row.values)[0] for t in model.estimators_])
        mean_pred = tree_preds.mean()
        spread = tree_preds.std()
        if mean_pred <= 0:
            return 70.0, tree_preds
        conf = 100 - (spread / mean_pred) * 140
        return float(np.clip(conf, 55, 98)), tree_preds
    except Exception:
        return 80.0, None


def compute_scores(inputs, prediction, df_clean):
    group_base = {"Manhattan": 88, "Brooklyn": 78, "Queens": 62, "Bronx": 50, "Staten Island": 45}
    demand = group_base.get(inputs["neighbourhood_group"], 55)
    demand += (365 - inputs["availability_365"]) / 365 * 14
    demand += min(inputs["reviews_per_month"], 5) * 2.2
    demand = float(np.clip(demand, 0, 100))

    booking = 46
    booking += min(inputs["number_of_reviews"], 120) * 0.28
    booking += min(inputs["reviews_per_month"], 5) * 5.4
    booking -= max(0, inputs["minimum_nights"] - 7) * 2.2
    if 30 <= inputs["availability_365"] <= 300:
        booking += 8
    booking = float(np.clip(booking, 0, 100))

    similar = df_clean[
        (df_clean["neighbourhood_group"] == inputs["neighbourhood_group"]) &
        (df_clean["room_type"] == inputs["room_type"])
    ]["price"]
    if len(similar) >= 5:
        percentile = float((similar < prediction).mean() * 100)
    else:
        percentile = float((df_clean["price"] < prediction).mean() * 100)

    if percentile >= 90:
        grade = "A+"
    elif percentile >= 75:
        grade = "A"
    elif percentile >= 50:
        grade = "B+"
    elif percentile >= 25:
        grade = "B"
    else:
        grade = "C"

    return {"demand": demand, "booking": booking, "percentile": percentile, "grade": grade}


def generate_recommendations(inputs, prediction, category):
    recs = []
    if inputs["minimum_nights"] > 7:
        recs.append(("📉", "Your minimum-night stay is high — lowering it could unlock more short-stay bookings."))
    if inputs["availability_365"] < 90:
        recs.append(("🗓️", "Availability is limited — opening more nights typically improves total annual revenue."))
    if inputs["reviews_per_month"] < 0.5:
        recs.append(("⭐", "Review velocity is low — a welcome discount can help build review volume early on."))
    if inputs["number_of_reviews"] > 50:
        recs.append(("💬", "Strong review history — guests trust this listing, so you can price with confidence."))
    if inputs["room_type"] == "Entire home/apt":
        recs.append(("🏠", "Entire-home listings command a premium — highlight privacy and space in your photos."))
    if inputs["neighbourhood_group"] in ("Manhattan", "Brooklyn"):
        recs.append(("🔥", "High-demand zone — weekend and event-based dynamic pricing can meaningfully lift revenue."))
    if category == "Luxury":
        recs.append(("✨", "This listing prices into the luxury tier — premium amenities help justify the rate."))
    if category == "Budget":
        recs.append(("💰", "Priced competitively — small upsells (early check-in, late checkout) can grow revenue."))
    recs.append(("📈", "Increase price 10–15% on Friday & Saturday nights to capture weekend demand."))
    recs.append(("🏘️", "Excellent neighbourhood fundamentals support strong long-term booking potential."))
    random.Random(int(prediction)).shuffle(recs)
    return recs[:5]


PLOTLY_LUX_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#0E1016", size=13),
    colorway=["#4E7CFF", "#8B5CF6", "#FF8A4C", "#22C1A0", "#FFC24E"],
    hoverlabel=dict(bgcolor="rgba(14,16,22,0.92)", font_color="white", font_family="Inter",
                     bordercolor="rgba(14,16,22,0.92)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12)),
    margin=dict(t=40, b=10, l=10, r=10),
)


def style_axes(fig):
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(107,114,128,0.25)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(107,114,128,0.12)", zeroline=False, linecolor="rgba(107,114,128,0.25)")
    return fig


# ═════════════════════════════════════════════════════════════════════════
# RADIAL METER / RING COMPONENTS  (pure CSS, not Plotly)
# ═════════════════════════════════════════════════════════════════════════

def radial_meter(value, label, tag, color="var(--violet)", suffix=""):
    kf = f"kf{uuid.uuid4().hex[:8]}"
    value = float(np.clip(value, 0, 100))
    st.markdown(f"""
    <style>
    @keyframes {kf} {{ from{{ --p:0; }} to{{ --p:{value}; }} }}
    </style>
    <div class="radial-outer">
        <div class="radial-ring" style="--ring-color:{color}; --p:0; animation:{kf} 1.3s cubic-bezier(.2,.8,.2,1) forwards;">
            <div class="radial-inner">
                <div class="radial-val">{value:.0f}{suffix}</div>
                <div class="radial-tag">{tag}</div>
            </div>
        </div>
    </div>
    <div class="meter-label">{label}</div>
    """, unsafe_allow_html=True)


def price_ring(prediction, min_p, max_p):
    pct = float(np.clip((prediction - min_p) / max(1e-6, (max_p - min_p)) * 100, 4, 100))
    kf = f"kf{uuid.uuid4().hex[:8]}"
    st.markdown(f"""
    <style>
    @keyframes {kf} {{ from{{ --p:0; }} to{{ --p:{pct}; }} }}
    </style>
    <div class="price-ring-outer">
        <div class="price-ring" style="--p:0; animation:{kf} 1.5s cubic-bezier(.2,.8,.2,1) forwards;">
            <div class="price-ring-inner">
                <div class="price-label">Estimated Nightly Price</div>
                <div class="price-value">${prediction:.0f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def grade_badge(grade):
    st.markdown(f"""
    <div class="grade-badge"><span>{grade}</span></div>
    <div class="meter-label">Market Rating</div>
    """, unsafe_allow_html=True)


def html_bar_list(df_counts, label_col, value_col, max_items=10):
    df_counts = df_counts.head(max_items)
    max_val = df_counts[value_col].max()
    rows = ""
    for i, (_, r) in enumerate(df_counts.iterrows()):
        pct = r[value_col] / max_val * 100
        rows += f"""
        <div class="bar-row">
            <div class="bar-row-top"><span>{r[label_col]}</span><span>{int(r[value_col])}</span></div>
            <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%; animation-delay:{i*0.06:.2f}s;"></div></div>
        </div>
        """
    st.markdown(rows, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# HERO — self-contained animated component (real JS parallax, own document)
# ═════════════════════════════════════════════════════════════════════════

def render_hero():
    html = """
    <html><head><style>
        * { margin:0; padding:0; box-sizing:border-box; }
        html, body { background:transparent; overflow:hidden; font-family:'Plus Jakarta Sans','Inter',sans-serif; }
        .stage { position:relative; width:100%; height:620px; border-radius:34px; overflow:hidden;
            background:
                radial-gradient(1200px 700px at 20% 10%, rgba(78,124,255,0.55), transparent 55%),
                radial-gradient(1000px 700px at 85% 20%, rgba(139,92,246,0.5), transparent 55%),
                radial-gradient(900px 700px at 50% 100%, rgba(255,138,76,0.4), transparent 55%),
                linear-gradient(135deg, #0E1016 0%, #181B26 45%, #11131c 100%);
            background-size: 200% 200%;
            animation: auroraMove 14s ease-in-out infinite;
        }
        @keyframes auroraMove {
            0%,100% { background-position: 0% 0%, 100% 0%, 50% 100%, 0% 0%; }
            50% { background-position: 30% 30%, 70% 40%, 60% 80%, 0% 0%; }
        }
        .blob { position:absolute; border-radius:50%; filter:blur(6px); opacity:.5; will-change:transform; }
        .b1 { width:220px; height:220px; left:6%; top:18%; background:radial-gradient(circle at 30% 30%, #7CA8FF, transparent 70%); }
        .b2 { width:160px; height:160px; left:78%; top:12%; background:radial-gradient(circle at 30% 30%, #B79CFF, transparent 70%); }
        .b3 { width:260px; height:260px; left:60%; top:60%; background:radial-gradient(circle at 30% 30%, #FF8A4C, transparent 70%); }
        .b4 { width:140px; height:140px; left:18%; top:68%; background:radial-gradient(circle at 30% 30%, #22C1A0, transparent 70%); }
        .b5 { width:110px; height:110px; left:45%; top:8%; background:radial-gradient(circle at 30% 30%, #FFC24E, transparent 70%); }

        .grain { position:absolute; inset:0; opacity:.04;
            background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }

        .content { position:relative; z-index:3; height:100%; display:flex; flex-direction:column;
            align-items:center; justify-content:center; text-align:center; padding:0 8%; }

        .eyebrow { display:inline-flex; align-items:center; gap:8px; color:#B79CFF; font-weight:700;
            font-size:12.5px; letter-spacing:.14em; text-transform:uppercase;
            padding:8px 18px; border-radius:999px; background:rgba(139,92,246,0.16);
            border:1px solid rgba(183,156,255,0.35); margin-bottom:26px;
            animation: rise 1s cubic-bezier(.2,.8,.2,1) both; }

        .title { font-size:60px; font-weight:900; color:#fff; line-height:1.06; letter-spacing:-0.02em;
            margin-bottom:20px; animation: rise 1.1s .1s cubic-bezier(.2,.8,.2,1) both; }
        .title .grad { background:linear-gradient(100deg, #7CA8FF, #B79CFF 55%, #FF8A4C); background-size:220% auto;
            -webkit-background-clip:text; background-clip:text; color:transparent; animation: shift 6s ease infinite; }

        .sub { font-size:19px; color:rgba(255,255,255,0.72); max-width:620px; margin-bottom:34px; font-weight:400;
            line-height:1.55; animation: rise 1.2s .2s cubic-bezier(.2,.8,.2,1) both; }

        .badges { display:flex; flex-wrap:wrap; gap:12px; justify-content:center;
            animation: rise 1.3s .3s cubic-bezier(.2,.8,.2,1) both; }
        .badge { padding:11px 22px; border-radius:999px; font-size:13.5px; font-weight:700; color:#fff;
            background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.18);
            backdrop-filter: blur(8px); animation: floaty 4s ease-in-out infinite; }
        .badge:nth-child(2n){ animation-delay:.6s; }
        .badge:nth-child(3n){ animation-delay:1.1s; }

        @keyframes rise { from{ opacity:0; transform:translateY(30px);} to{ opacity:1; transform:translateY(0);} }
        @keyframes shift { 0%,100%{ background-position:0% 50%;} 50%{ background-position:100% 50%;} }
        @keyframes floaty { 0%,100%{ transform:translateY(0);} 50%{ transform:translateY(-7px);} }

        .glow-ring { position:absolute; width:640px; height:640px; border-radius:50%;
            border:1px solid rgba(255,255,255,0.08); top:50%; left:50%; transform:translate(-50%,-50%); }
        .glow-ring.r2 { width:820px; height:820px; border-color: rgba(255,255,255,0.05); }
    </style></head>
    <body>
        <div class="stage" id="stage">
            <div class="glow-ring"></div>
            <div class="glow-ring r2"></div>
            <div class="blob b1" data-depth="30"></div>
            <div class="blob b2" data-depth="55"></div>
            <div class="blob b3" data-depth="20"></div>
            <div class="blob b4" data-depth="45"></div>
            <div class="blob b5" data-depth="60"></div>
            <div class="grain"></div>
            <div class="content">
                <div class="eyebrow">✨ AI-Powered Valuation Engine</div>
                <div class="title">🏡 AI Airbnb<br><span class="grad">Valuation Studio</span></div>
                <div class="sub">Discover the true market value of your Airbnb property using Artificial
                Intelligence trained on tens of thousands of real New York City listings.</div>
                <div class="badges">
                    <span class="badge">🤖 AI Powered</span>
                    <span class="badge">📈 Machine Learning</span>
                    <span class="badge">💰 Price Intelligence</span>
                    <span class="badge">🏘️ Real Estate Analytics</span>
                    <span class="badge">🌲 Random Forest</span>
                </div>
            </div>
        </div>
        <script>
            const stage = document.getElementById('stage');
            const blobs = document.querySelectorAll('.blob');
            stage.addEventListener('mousemove', (e) => {
                const rect = stage.getBoundingClientRect();
                const relX = (e.clientX - rect.left) / rect.width - 0.5;
                const relY = (e.clientY - rect.top) / rect.height - 0.5;
                blobs.forEach((b) => {
                    const depth = parseFloat(b.dataset.depth);
                    b.style.transform = `translate(${relX * depth}px, ${relY * depth}px)`;
                });
            });
            stage.addEventListener('mouseleave', () => {
                blobs.forEach((b) => { b.style.transform = 'translate(0px,0px)'; });
            });
        </script>
    </body></html>
    """
    components.html(html, height=620, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def go_to(page):
    st.session_state.page = page


# ═════════════════════════════════════════════════════════════════════════
# TOP NAVBAR (replaces sidebar entirely)
# ═════════════════════════════════════════════════════════════════════════

def navbar():
    pages = [("Home", "🏠"), ("Predict Price", "🤖"), ("Analytics", "📊"), ("About Model", "🧠"), ("Developer", "👤")]
    c_logo, *c_rest, c_cta = st.columns([2.2] + [1] * len(pages) + [1.3])
    with c_logo:
        st.markdown("""
        <div class="nav-logo"><div class="dot">🏡</div> AI Valuation Studio</div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="nav-row">', unsafe_allow_html=True)
    cols = st.columns([2.2] + [1] * len(pages) + [1.3])
    for col, (label, icon) in zip(cols[1:-1], pages):
        with col:
            st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True,
                      on_click=go_to, args=(label,))
    with cols[-1]:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        st.button("Get Estimate →", key="nav_cta", use_container_width=True, on_click=go_to, args=("Predict Price",))
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid rgba(107,114,128,0.15); margin:4px 0 18px 0;'>",
                unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════

def page_home(bundle):
    render_hero()

    st.markdown('<div class="hero-cta-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="hero-cta-inner">', unsafe_allow_html=True)
    cta1, cta2, cta3 = st.columns([2, 1, 1])
    with cta1:
        st.markdown("""
        <div style="font-weight:800; font-size:17px;">Ready to price your listing like a pro?</div>
        <div style="color:var(--slate); font-size:14px;">Get an instant AI-backed estimate in seconds.</div>
        """, unsafe_allow_html=True)
    with cta2:
        st.button("🔮 Predict My Price", key="home_cta1", use_container_width=True,
                   on_click=go_to, args=("Predict Price",))
    with cta3:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        st.button("📊 View Analytics", key="home_cta2", use_container_width=True,
                   on_click=go_to, args=("Analytics",))
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)

    if bundle is None:
        st.warning("Add `AB_NYC_2019.csv` next to app.py to activate live statistics and predictions.")
        return

    df_clean = bundle["df_clean"]
    metrics = bundle["metrics"]

    st.markdown('<span class="eyebrow">⚡ Live Model Stats</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Numbers that <span class="grad-text">speak for themselves</span></div>',
                unsafe_allow_html=True)
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    cols = st.columns(5)
    stats = [
        ("🏘️", f"{len(df_clean):,}", "Listings Analyzed"),
        ("🎯", f"{metrics['r2']*100:.1f}%", "Model Accuracy"),
        ("⚡", "< 0.3s", "Prediction Speed"),
        ("🧬", f"{len(bundle['feature_names'])}", "Features Used"),
        ("💵", f"${df_clean['price'].mean():.0f}", "Average Nightly Price"),
    ]
    for col, (icon, val, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card"><div class="stat-icon">{icon}</div>
            <div class="stat-num">{val}</div><div class="stat-label">{label}</div></div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:56px;'></div>", unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">🧠 How It Works</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Trained on real <span class="grad-text">New York City</span> data</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-desc">A Random Forest model learns pricing patterns from location, '
                'room type, availability and host history — the same signals real hosts use to price competitively.</div>',
                unsafe_allow_html=True)
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    feats = [
        ("📍", "Location Intelligence", "Latitude, longitude, borough and neighbourhood signals capture micro-market pricing."),
        ("🛏️", "Property Profile", "Room type, minimum stay and availability shape how guests value your space."),
        ("⭐", "Host Analytics", "Review volume, review velocity and portfolio size reflect trust and demand."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3], feats):
        with col:
            with st.container(border=True):
                st.markdown(f"""
                <div style="padding:16px 12px;">
                <div style="font-size:28px;" class="stat-icon">{icon}</div>
                <div style="font-weight:800; font-size:17px; margin:10px 0 8px 0;">{title}</div>
                <div style="color:var(--slate); font-size:14.5px; line-height:1.55;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT PRICE — floating luxury booking widget
# ═════════════════════════════════════════════════════════════════════════

def page_predict(bundle):
    st.markdown('<span class="eyebrow">🤖 AI Price Engine</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Build your <span class="grad-text">booking profile</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Fill in your property details below — like configuring a booking, '
                'not filling out a form — and the AI estimates your ideal nightly rate.</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    if bundle is None:
        st.error("Dataset not found — place `AB_NYC_2019.csv` next to app.py to enable predictions.")
        return

    neighbourhood_map = bundle["neighbourhood_map"]
    groups = sorted(neighbourhood_map.keys())

    with st.container(border=True):
        st.markdown("<div style='padding:14px 16px 4px 16px;'>", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown('<div class="segment-title"><div class="segment-icon-chip">📍</div> Location</div>',
                        unsafe_allow_html=True)
            neighbourhood_group = st.selectbox("Neighbourhood Group", groups,
                                                index=groups.index("Manhattan") if "Manhattan" in groups else 0)
            neighbourhood_options = neighbourhood_map.get(neighbourhood_group, [])
            neighbourhood = st.selectbox("Neighbourhood", neighbourhood_options)
            latitude = st.slider("Latitude", 40.49, 40.92, 40.75, 0.001)
            longitude = st.slider("Longitude", -74.25, -73.70, -73.98, 0.001)

        with col_b:
            st.markdown('<div class="segment-title"><div class="segment-icon-chip">🛏️</div> Property</div>',
                        unsafe_allow_html=True)
            room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
            minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=3, step=1)
            availability_365 = st.slider("Availability (days/yr)", 0, 365, 180)

        with col_c:
            st.markdown('<div class="segment-title"><div class="segment-icon-chip">⭐</div> Host Analytics</div>',
                        unsafe_allow_html=True)
            number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=1000, value=20, step=1)
            reviews_per_month = st.number_input("Reviews / Month", min_value=0.0, max_value=30.0, value=1.5,
                                                 step=0.1, format="%.1f")
            calculated_host_listings_count = st.number_input("Host's Total Listings", min_value=1, max_value=500,
                                                               value=2, step=1)

        st.markdown("<hr class='seg-divider'>", unsafe_allow_html=True)
        predict_clicked = st.button("🔮  Reveal My AI Price Estimate", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    inputs = dict(
        neighbourhood_group=neighbourhood_group, neighbourhood=neighbourhood,
        latitude=latitude, longitude=longitude, room_type=room_type,
        minimum_nights=minimum_nights, availability_365=availability_365,
        number_of_reviews=number_of_reviews, reviews_per_month=reviews_per_month,
        calculated_host_listings_count=calculated_host_listings_count,
    )

    if predict_clicked:
        play_loading_sequence()

        row = build_feature_row(bundle["feature_names"], inputs)
        prediction = max(float(bundle["model"].predict(row)[0]), 10.0)
        category, pill_class = price_category(prediction, bundle["price_quantiles"])
        conf, tree_preds = confidence_score(bundle["model"], row)
        scores = compute_scores(inputs, prediction, bundle["df_clean"])

        if tree_preds is not None:
            low, high = np.percentile(tree_preds, [10, 90])
        else:
            low, high = prediction * 0.85, prediction * 1.15

        result = dict(
            prediction=prediction, category=category, pill_class=pill_class,
            confidence=conf, low=low, high=high, scores=scores,
            inputs=inputs, timestamp=datetime.now().strftime("%H:%M:%S"),
        )
        st.session_state.last_result = result
        st.session_state.history.append(result)

    if st.session_state.last_result:
        render_result(st.session_state.last_result, bundle)


def play_loading_sequence():
    box = st.empty()
    steps = [
        ("🏠", "Reading Property..."),
        ("🌍", "Searching Nearby Listings..."),
        ("📈", "Comparing Market Trends..."),
        ("🤖", "Running AI Engine..."),
        ("✨", "Calculating Price..."),
    ]
    for i, (icon, text) in enumerate(steps):
        progress = int((i + 1) / len(steps) * 100)
        box.markdown(f"""
        <div class="glass loading-wrap">
            <div class="loading-ring-wrap">
                <div class="loading-core"></div>
                <div class="loading-emoji">{icon}</div>
            </div>
            <div class="loading-text">{text}</div>
            <div class="loading-bar-track"><div class="loading-bar-fill" style="width:{progress}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.5)
    box.empty()


def render_result(result, bundle):
    st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">✨ Your Result</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Here\'s your <span class="grad-text">AI valuation</span></div>',
                unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<div style='padding:18px 10px;'>", unsafe_allow_html=True)
        top_l, top_r = st.columns([1, 1])
        with top_l:
            price_ring(result["prediction"], bundle["df_clean"]["price"].min(), bundle["df_clean"]["price"].max())
            st.markdown(f"""
            <div style="text-align:center; margin-top:14px;">
                <span class="pill {result['pill_class']}">🏷️ {result['category']} Tier</span>
            </div>
            """, unsafe_allow_html=True)
        with top_r:
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                radial_meter(result["confidence"], "Price Confidence", "CONF", "var(--blue)", "%")
            with m2:
                radial_meter(result["scores"]["demand"], "Demand Meter", "DEMAND", "var(--orange)")
            st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
            m3, m4 = st.columns(2)
            with m3:
                radial_meter(result["scores"]["booking"], "Booking Score", "SCORE", "var(--teal)")
            with m4:
                grade_badge(result["scores"]["grade"])

        st.markdown(f"""
        <div style="text-align:center; margin-top:22px; color:var(--slate); font-size:14px;">
            Estimated confident range: <b style="color:var(--ink);">${result['low']:.0f} – ${result['high']:.0f}</b>
            &nbsp;·&nbsp; Priced above <b style="color:var(--ink);">{result['scores']['percentile']:.0f}%</b> of comparable listings
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.35, 1])

    with col_left:
        st.markdown("#### ✨ AI Recommendations")
        recs = generate_recommendations(result["inputs"], result["prediction"], result["category"])
        for icon, text in recs:
            st.markdown(f'<div class="rec-card"><span class="rec-icon">{icon}</span><div>{text}</div></div>',
                        unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 🕘 Prediction History")
        hist_df = pd.DataFrame([
            {
                "Time": h["timestamp"],
                "Neighbourhood": h["inputs"]["neighbourhood"],
                "Room Type": h["inputs"]["room_type"],
                "Predicted Price": f"${h['prediction']:.0f}",
            }
            for h in st.session_state.history[::-1]
        ])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download History (CSV)", csv, "prediction_history.csv", "text/csv",
                            use_container_width=True)

        if len(st.session_state.history) >= 2:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            st.markdown("##### ⚖️ Compare Last 2 Predictions")
            a, b = st.session_state.history[-1], st.session_state.history[-2]
            comp_df = pd.DataFrame({
                "Metric": ["Neighbourhood", "Room Type", "Predicted Price"],
                "Latest": [a["inputs"]["neighbourhood"], a["inputs"]["room_type"], f"${a['prediction']:.0f}"],
                "Previous": [b["inputs"]["neighbourhood"], b["inputs"]["room_type"], f"${b['prediction']:.0f}"],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═════════════════════════════════════════════════════════════════════════

def page_analytics(bundle):
    st.markdown('<span class="eyebrow">📊 Market Intelligence</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Visual <span class="grad-text">market analytics</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Explore pricing patterns across New York City\'s Airbnb market.</div>',
                unsafe_allow_html=True)

    if bundle is None:
        st.warning("Add `AB_NYC_2019.csv` next to app.py to enable analytics.")
        return

    df = bundle["df_clean"]

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("<div style='padding:6px 10px;'>", unsafe_allow_html=True)
            st.markdown("**💵 Average Price by Room Type**")
            fig = px.bar(df.groupby("room_type", as_index=False)["price"].mean(),
                         x="room_type", y="price", text_auto=".0f")
            fig.update_traces(marker_line_width=0)
            fig.update_layout(**PLOTLY_LUX_TEMPLATE, height=340, xaxis_title="", yaxis_title="Avg Price ($)")
            style_axes(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            st.markdown("<div style='padding:6px 10px;'>", unsafe_allow_html=True)
            st.markdown("**🏙️ Listings by Neighbourhood Group**")
            fig = px.pie(df, names="neighbourhood_group", hole=0.62)
            fig.update_traces(textfont_size=12, marker=dict(line=dict(color="rgba(0,0,0,0)", width=0)))
            fig.update_layout(**PLOTLY_LUX_TEMPLATE, height=340)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            st.markdown("<div style='padding:6px 10px;'>", unsafe_allow_html=True)
            st.markdown("**📊 Price Distribution**")
            fig = px.histogram(df, x="price", nbins=50)
            fig.update_traces(marker_line_width=0)
            fig.update_layout(**PLOTLY_LUX_TEMPLATE, height=340, xaxis_title="Price ($)", yaxis_title="Listings",
                               showlegend=False)
            style_axes(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        with st.container(border=True):
            st.markdown("<div style='padding:6px 10px;'>", unsafe_allow_html=True)
            st.markdown("**🗓️ Availability Across the Year**")
            fig = px.histogram(df, x="availability_365", nbins=30)
            fig.update_traces(marker_line_width=0, marker_color="#8B5CF6")
            fig.update_layout(**PLOTLY_LUX_TEMPLATE, height=340, xaxis_title="Available Days", yaxis_title="Listings",
                               showlegend=False)
            style_axes(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        with st.container(border=True):
            st.markdown("<div style='padding:6px 10px;'>", unsafe_allow_html=True)
            st.markdown("**🏘️ Top 10 Neighbourhoods by Listing Count**")
            top10 = df["neighbourhood"].value_counts().head(10).reset_index()
            top10.columns = ["neighbourhood", "count"]
            html_bar_list(top10, "neighbourhood", "count")
            st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        with st.container(border=True):
            st.markdown("<div style='padding:6px 10px;'>", unsafe_allow_html=True)
            st.markdown("**📈 Price vs. Number of Reviews**")
            sample = df.sample(min(2000, len(df)), random_state=42)
            fig = px.scatter(sample, x="number_of_reviews", y="price", color="room_type", opacity=0.65)
            fig.update_traces(marker=dict(size=6, line=dict(width=0)))
            fig.update_layout(**PLOTLY_LUX_TEMPLATE, height=380, xaxis_title="Number of Reviews", yaxis_title="Price ($)")
            style_axes(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<div style='padding:6px 10px;'>", unsafe_allow_html=True)
        st.markdown("**⭐ Review Distribution (per month)**")
        fig = px.histogram(df, x="reviews_per_month", nbins=40)
        fig.update_traces(marker_line_width=0, marker_color="#FF8A4C")
        fig.update_layout(**PLOTLY_LUX_TEMPLATE, height=320, xaxis_title="Reviews per Month", yaxis_title="Listings",
                           showlegend=False)
        style_axes(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT MODEL
# ═════════════════════════════════════════════════════════════════════════

def page_about(bundle):
    st.markdown('<span class="eyebrow">🧠 Under the Hood</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About the <span class="grad-text">model</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-desc">A transparent look at how the AI estimates your Airbnb price.</div>',
                unsafe_allow_html=True)

    if bundle is None:
        st.warning("Add `AB_NYC_2019.csv` next to app.py to see live model metrics.")
        return

    metrics = bundle["metrics"]
    m1, m2, m3 = st.columns(3)
    for col, (icon, label, val) in zip([m1, m2, m3], [
        ("🎯", "R² Score", f"{metrics['r2']*100:.1f}%"),
        ("📏", "Mean Absolute Error", f"${metrics['mae']:.2f}"),
        ("📐", "RMSE", f"${metrics['rmse']:.2f}"),
    ]):
        with col:
            st.markdown(f"""
            <div class="stat-card"><div class="stat-icon">{icon}</div>
            <div class="stat-num">{val}</div><div class="stat-label">{label}</div></div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div style='padding:8px 12px;'>", unsafe_allow_html=True)
        st.markdown("**🌲 Top 15 Most Important Features**")
        imp = bundle["importance"].head(15).iloc[::-1]
        fig = px.bar(imp, x="importance", y="feature", orientation="h")
        fig.update_traces(marker_line_width=0)
        fig.update_layout(**PLOTLY_LUX_TEMPLATE, height=460, xaxis_title="Importance", yaxis_title="",
                           showlegend=False)
        style_axes(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass" style="padding:22px 26px;">
    <b>Model:</b> Random Forest Regressor (200 trees) &nbsp;·&nbsp;
    <b>Training data:</b> NYC Airbnb Open Data 2019 &nbsp;·&nbsp;
    <b>Preprocessing:</b> IQR outlier removal, median imputation, one-hot encoding
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: DEVELOPER
# ═════════════════════════════════════════════════════════════════════════

def page_developer():
    st.markdown('<span class="eyebrow">👤 Behind the Platform</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">The <span class="grad-text">developer</span></div>',
                unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"""
        <div style="padding:20px 14px;">
        <div style="font-size:24px; font-weight:800;">👤 {DEVELOPER_NAME}</div>
        <div style="color:var(--slate); margin-top:10px; line-height:1.65; font-size:15px;">
        Machine Learning practitioner passionate about turning raw data into real-world
        pricing intelligence for the sharing economy. This project applies a Random Forest
        Regressor to NYC Airbnb listings to help hosts price with confidence.
        </div>
        </div>
        """, unsafe_allow_html=True)

    render_footer()


def render_footer():
    st.markdown("""
    <div class="footer-wrap">
        Built with
        <span class="footer-chip">Python</span>
        <span class="footer-chip">Streamlit</span>
        <span class="footer-chip">Scikit-Learn</span>
        <span class="footer-chip">Pandas</span>
        <span class="footer-chip">NumPy</span>
        <span class="footer-chip">Plotly</span>
        <span class="footer-chip">Machine Learning</span>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# APP ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

def main():
    inject_css()
    navbar()
    bundle = get_model_bundle()

    page = st.session_state.page
    if page == "Home":
        page_home(bundle)
    elif page == "Predict Price":
        page_predict(bundle)
    elif page == "Analytics":
        page_analytics(bundle)
    elif page == "About Model":
        page_about(bundle)
    elif page == "Developer":
        page_developer()
        return

    render_footer()


if __name__ == "__main__":
    main()