# -*- coding: utf-8 -*-
"""
styles.py — Design System "Enterprise Slate" pour L3amel Lfahem.

Injecte un CSS complet qui masque les traces natives de Streamlit et applique
une identité visuelle de type SaaS d'entreprise (palette Slate + Royal Blue/Teal).
Gère un thème clair et un thème sombre, basculables via st.session_state["theme"].
"""

import streamlit as st

LIGHT_VARS = """
    --bg-canvas: #F8FAFC;
    --bg-card: #FFFFFF;
    --border-card: #E2E8F0;
    --text-primary: #0F172A;
    --text-muted: #475569;
    --text-disabled: #94A3B8;
    --sidebar-bg: #0F172A;
    --sidebar-text: #F1F5F9;
"""

DARK_VARS = """
    --bg-canvas: #0B1220;
    --bg-card: #111827;
    --border-card: #1F2937;
    --text-primary: #F1F5F9;
    --text-muted: #94A3B8;
    --text-disabled: #475569;
    --sidebar-bg: #05070D;
    --sidebar-text: #E2E8F0;
"""

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    __THEME_VARS__
    --accent-blue: #2563EB;
    --accent-teal: #0D9488;
    --success-bg: #DCFCE7; --success-text: #15803D;
    --warning-bg: #FEF3C7; --warning-text: #B45309;
    --danger-bg: #FEE2E2; --danger-text: #B91C1C;
    --info-bg: #DBEAFE; --info-text: #1D4ED8;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-canvas) !important;
    color: var(--text-primary) !important;
}

#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.stDeployButton { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
button[title="View fullscreen"] { visibility: hidden !important; }

section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid #1E293B !important;
}
section[data-testid="stSidebar"] * {
    color: var(--sidebar-text) !important;
}
section[data-testid="stSidebar"] .stButton>button {
    background: #1E293B; border: 1px solid #334155; border-radius: 8px;
    color: #F1F5F9 !important;
}

/* Correctif critique : la règle générale ci-dessus met tout le texte de la
   sidebar en clair, y compris à l'intérieur des champs de saisie dont le
   fond reste blanc -> texte invisible. On force explicitement un contraste
   correct sur tous les widgets de saisie de la sidebar. */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] div[data-baseweb="select"] div,
section[data-testid="stSidebar"] div[data-baseweb="base-input"] {
    color: #0F172A !important;
    background-color: #FFFFFF !important;
}
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
    color: #64748B !important;
    opacity: 1;
}

/* Champs de saisie de la zone principale : contraste explicite, cohérent
   avec le thème clair/sombre actif (var(--bg-card) / var(--text-primary)). */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
div[data-baseweb="base-input"] input,
textarea[data-testid="stChatInputTextArea"] {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-card) !important;
}
div[data-testid="stChatInput"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-card) !important;
}

.block-container { padding-top: 1.6rem; max-width: 1400px; }

.custom-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    transition: all 0.2s ease-in-out;
}
.custom-card:hover {
    border-color: #CBD5E1;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
}

.app-title { font-weight: 700; font-size: 1.15rem; color: var(--text-primary); }
.app-subtitle { color: var(--text-muted); font-size: 0.85rem; }

.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 6px; background: #22C55E; box-shadow: 0 0 6px #22C55E;
}
.status-row { font-size: 0.82rem; margin-bottom: 6px; display: flex; align-items: center; }

.badge {
    display: inline-flex; align-items: center; padding: 2px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600; margin-right: 6px;
}
.badge-success { background: var(--success-bg); color: var(--success-text); }
.badge-warning { background: var(--warning-bg); color: var(--warning-text); }
.badge-danger  { background: var(--danger-bg);  color: var(--danger-text); }
.badge-info    { background: var(--info-bg);    color: var(--info-text); }

.citation-badge {
    display: inline-flex; align-items: center; background-color: var(--info-bg);
    border: 1px solid #BFDBFE; color: var(--info-text);
    padding: 2px 8px; border-radius: 6px; font-size: 12px; font-weight: 600;
    cursor: pointer; margin-right: 4px;
}
.citation-badge:hover { background-color: #BFDBFE; }

.chat-bubble-user {
    background: var(--accent-blue); color: white; padding: 10px 14px;
    border-radius: 14px 14px 2px 14px; margin: 6px 0; max-width: 85%;
    margin-left: auto; font-size: 0.92rem;
}
.chat-bubble-bot {
    background: var(--bg-card); border: 1px solid var(--border-card); color: var(--text-primary);
    padding: 10px 14px; border-radius: 14px 14px 14px 2px; margin: 6px 0; max-width: 90%;
    font-size: 0.92rem;
}
.confidence-bar-bg { background: var(--border-card); border-radius: 6px; height: 6px; width: 100%; }
.confidence-bar-fill { height: 6px; border-radius: 6px; }

.finops-metric { font-size: 0.78rem; color: var(--text-muted); display:flex; justify-content:space-between; margin-bottom:4px;}
.finops-value { font-weight: 700; color: var(--text-primary); }

.diff-add { background: #DCFCE7; color: #14532D; padding: 1px 3px; border-radius: 3px; }
.diff-del { background: #FEE2E2; color: #7F1D1D; padding: 1px 3px; text-decoration: line-through; border-radius: 3px; }
"""


def _theme_css(theme: str) -> str:
    variables = DARK_VARS if theme == "dark" else LIGHT_VARS
    return BASE_CSS.replace("__THEME_VARS__", variables)


def inject_css():
    """À appeler une seule fois en tête de app.py."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    st.markdown(f"<style>{_theme_css(st.session_state['theme'])}</style>", unsafe_allow_html=True)


def theme_toggle_button():
    """Widget sidebar pour basculer clair/sombre, à placer dans app.py."""
    label = "🌙 Mode sombre" if st.session_state.get("theme", "light") == "light" else "☀️ Mode clair"
    if st.button(label, key="theme_toggle_btn", use_container_width=True):
        st.session_state["theme"] = "dark" if st.session_state.get("theme", "light") == "light" else "light"
        st.rerun()


def status_dot(label: str, ok: bool = True):
    color = "#22C55E" if ok else "#EF4444"
    st.markdown(
        f'<div class="status-row"><span class="status-dot" style="background:{color};'
        f'box-shadow:0 0 6px {color};"></span>{label}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "info"):
    return f'<span class="badge badge-{kind}">{text}</span>'


LOGO_SVG = """
<svg width="34" height="34" viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg">
  <rect width="34" height="34" rx="9" fill="#2563EB"/>
  <path d="M9 23V11h3.2v9.1h6.4V23H9z" fill="#FFFFFF"/>
  <circle cx="24.5" cy="10.5" r="2.6" fill="#0D9488"/>
</svg>
"""
