# -*- coding: utf-8 -*-
"""
app.py — L3amel Lfahem (العامل الفاهم)
Enterprise Multilingual RAG & Document Intelligence Platform — point d'entrée Streamlit.

v2 : démarrage autonome (clé API via secrets, corpus par défaut préchargé et
mis en cache), correctifs de contraste, workspace Data Visualisation enrichi
de l'observatoire socio-économique, et module FinOps développé en profondeur.
"""

import os
import json
import tempfile

import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

from styles import inject_css, theme_toggle_button, status_dot, badge, LOGO_SVG
from rag_engine import (
    FinOpsGateway, extract_document, build_chunks, SearchIndex,
    mask_pii, detect_prompt_injection, compute_confidence, build_system_prompt,
    log_audit, get_audit_log, log_feedback, get_feedback_log, pdf_diff,
    register_tables_duckdb, text_to_sql, export_docx, export_pdf,
    compute_finops_insights, MODEL_FAST, MODEL_PRO,
)
from agents import (
    cascade_route, call_llm_with_fallback, multi_agent_deliberation,
    analyze_sentiment, translate_citation, regulatory_compliance_check,
)
from data_loader import load_default_legal_room, load_default_stats_db, DEFAULT_ROOM_NAME

st.set_page_config(page_title="L3amel Lfahem", page_icon="⚖️", layout="wide")
inject_css()

# ------------------------------------------------------------------------------
# ÉTAT DE SESSION
# ------------------------------------------------------------------------------
defaults = {
    "data_rooms": {}, "current_room": None, "chat_history": [],
    "role": "Lecteur", "target_lang": "Auto", "gateway": None,
    "last_answer": None, "last_sources": [], "budget_usd": 5.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _get_default_api_key() -> str:
    """Récupère la clé API depuis les Secrets Streamlit (déploiement autonome),
    sinon la variable d'environnement — sans jamais bloquer l'utilisateur final."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


_default_key = _get_default_api_key()
if _default_key and st.session_state["gateway"] is None:
    st.session_state["gateway"] = FinOpsGateway(api_key=_default_key)

# ------------------------------------------------------------------------------
# CHARGEMENT DU CORPUS PAR DÉFAUT (une seule fois, serveur entier, via cache_resource)
# ------------------------------------------------------------------------------
if DEFAULT_ROOM_NAME not in st.session_state["data_rooms"]:
    default_room = load_default_legal_room()
    if default_room:
        st.session_state["data_rooms"][DEFAULT_ROOM_NAME] = default_room
        st.session_state["current_room"] = DEFAULT_ROOM_NAME

if "stats_con" not in st.session_state:
    st.session_state["stats_con"], st.session_state["stats_tables"] = load_default_stats_db()

ROLE_ACCESS = {"Lecteur": ["public"], "RH": ["public", "RH"], "Finance": ["public", "Finance"], "Admin": None}


def visible_rooms():
    role = st.session_state["role"]
    allowed = ROLE_ACCESS[role]
    out = []
    for name, room in st.session_state["data_rooms"].items():
        if allowed is None or room.get("access", "public") in allowed:
            out.append(name)
    return out


# ==============================================================================
# SIDEBAR — COCKPIT D'ADMINISTRATION
# ==============================================================================
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">{LOGO_SVG}'
        f'<div><div class="app-title">L3amel Lfahem</div>'
        f'<div class="app-subtitle">العامل الفاهم — Droit du travail & Ingénieurs</div></div></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Status Hub**")
    status_dot("LLM Gateway Ready", ok=st.session_state["gateway"] is not None)
    status_dot("VectorDB Connected", ok=len(st.session_state["data_rooms"]) > 0)
    status_dot("Observatoire socio-économique", ok=len(st.session_state.get("stats_tables", [])) > 0)
    status_dot("Local Engine (fallback)", ok=True)

    if st.session_state["gateway"] is None:
        st.caption(
            "⚠️ Service IA non configuré côté serveur (`GROQ_API_KEY` manquant dans les "
            "Secrets). La navigation et la consultation des données restent disponibles ; "
            "contactez l'administrateur pour activer le chatbot."
        )
    st.divider()

    st.markdown("**Data Rooms**")
    rooms = visible_rooms()
    if rooms:
        idx = rooms.index(st.session_state["current_room"]) if st.session_state["current_room"] in rooms else 0
        st.session_state["current_room"] = st.selectbox("Data room active", rooms, index=idx)
    st.caption(f"« {DEFAULT_ROOM_NAME} » est chargé par défaut (Code du travail + statistiques marché ingénieurs).")

    with st.expander("➕ Ajouter un data room (optionnel)"):
        room_name = st.text_input("Nom du nouveau data room", value="")
        access_level = st.selectbox("Confidentialité", ["public", "RH", "Finance"])
        uploaded = st.file_uploader("Déposer un ou plusieurs PDF", type=["pdf"], accept_multiple_files=True)

        if st.button("⚙️ Traiter les documents", use_container_width=True,
                      disabled=not (uploaded and room_name)):
            try:
                progress = st.progress(0, text="Extraction en cours...")
                all_pages, all_tables, all_anomalies, paths = [], [], [], []
                for i, f in enumerate(uploaded):
                    tmp_path = os.path.join(tempfile.gettempdir(), f.name)
                    with open(tmp_path, "wb") as out:
                        out.write(f.getbuffer())
                    doc = extract_document(tmp_path)
                    all_pages.extend(doc["pages"])
                    all_tables.extend(doc["tables"])
                    all_anomalies.extend(doc["anomalies"])
                    paths.append(tmp_path)
                    progress.progress(int((i + 1) / len(uploaded) * 55),
                                       text=f"Extraction : {f.name} ({i+1}/{len(uploaded)})")

                progress.progress(65, text="Découpage sémantique (chunking)...")
                chunks = build_chunks(all_pages, data_room=room_name)

                progress.progress(75, text="Chargement du modèle d'embedding (une seule fois par serveur)...")
                index = SearchIndex(chunks)
                progress.progress(100, text="Terminé ✅")

                st.session_state["data_rooms"][room_name] = {
                    "pages": all_pages, "tables": all_tables, "chunks": chunks, "index": index,
                    "access": access_level, "paths": paths, "anomalies": all_anomalies,
                }
                st.session_state["current_room"] = room_name
                st.success(f"Data room « {room_name} » prêt : {len(chunks)} chunks, {len(all_tables)} tables.")
            except Exception as e:
                st.error(f"Échec du traitement : {e}")
    st.divider()

    st.markdown("**Switcher linguistique**")
    st.session_state["target_lang"] = st.selectbox(
        "Langue de réponse forcée", ["Auto", "Français", "Darija", "Arabe classique", "Anglais"]
    )
    multi_doc = st.checkbox("Recherche multi-documents (tous les data rooms visibles)", value=False)
    use_multi_agent = st.checkbox("Mode Multi-Agents (Juridique + Financier + Synthèse)", value=False)

    st.divider()
    st.session_state["role"] = st.selectbox("Rôle utilisateur (RBAC)", ["Lecteur", "RH", "Finance", "Admin"])

    st.divider()
    theme_toggle_button()

    st.divider()
    st.markdown("**FinOps — aperçu rapide**")
    if st.session_state["gateway"]:
        snap = st.session_state["gateway"].snapshot()
        st.markdown(
            f'<div class="finops-metric"><span>Tokens in/out</span>'
            f'<span class="finops-value">{snap["input_tokens"]}/{snap["output_tokens"]}</span></div>'
            f'<div class="finops-metric"><span>Coût session</span>'
            f'<span class="finops-value">${snap["cost_usd"]}</span></div>'
            f'<div class="finops-metric"><span>Latence dernier appel</span>'
            f'<span class="finops-value">{snap["last_latency_ms"]} ms</span></div>',
            unsafe_allow_html=True)
        st.caption("Détail complet, tendances et économies : onglet 💰 FinOps.")
    else:
        st.caption("Suivi FinOps indisponible : service IA non configuré côté serveur.")


# ==============================================================================
# HELPERS
# ==============================================================================

def gather_context(question: str, k: int = 5):
    gateway = st.session_state["gateway"]
    rooms = visible_rooms() if multi_doc else [st.session_state["current_room"]]
    candidats = []
    for r in rooms:
        if not r:
            continue
        idx: SearchIndex = st.session_state["data_rooms"][r]["index"]
        candidats.extend(idx.hybrid_search(question, gateway, k=k))
    candidats.sort(key=lambda c: -c.get("_rerank_score", 0))
    top = candidats[:k]
    contexte = "\n".join(
        f"[Source {i+1} — {c['source_file']} | Page {c['page']} | {c['article']}]\n{c['texte_parent']}"
        for i, c in enumerate(top)
    )
    return top, contexte


def render_citations(sources):
    for i, s in enumerate(sources, 1):
        st.markdown(f'<span class="citation-badge">📄 {i}. Page {s["page"]} · {s["article"]}</span>',
                    unsafe_allow_html=True)
    with st.expander("📚 Voir les extraits sources (réponse détaillée)"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**{i}. {s['source_file']} — Page {s['page']} — {s['article']}**")
            st.caption(s["texte_parent"][:600] + ("..." if len(s["texte_parent"]) > 600 else ""))


# ==============================================================================
# WORKSPACES
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chatbot & Lecteur", "📊 Data Visualisation",
    "🔍 Analyse différentielle & Conformité", "🗂️ Audit & Exports", "💰 FinOps",
])

# --- WORKSPACE 1 : CHATBOT & READER MODE ------------------------------------
with tab1:
    st.caption("Assistant spécialisé : Code du travail marocain et marché de l'emploi des ingénieurs. "
               "Le corpus par défaut est déjà chargé — posez directement votre question.")
    col_chat, col_reader = st.columns([1, 1])

    with col_chat:
        st.markdown("#### Assistant L3amel Lfahem")

        audio = st.audio_input("🎙️ Poser une question à l'oral")
        question_vocale = None
        if audio and st.session_state["gateway"]:
            tmp_audio = os.path.join(tempfile.gettempdir(), "voice_query.wav")
            with open(tmp_audio, "wb") as f:
                f.write(audio.getbuffer())
            question_vocale = st.session_state["gateway"].transcribe_audio(tmp_audio)
            st.caption(f"🗣️ Transcrit : {question_vocale}")

        question = st.chat_input("Écrivez votre question (FR / Darija / العربية / English)...")
        question = question_vocale or question

        for msg in st.session_state["chat_history"]:
            cls = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
            st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg["role"] == "assistant" and msg.get("sources"):
                render_citations(msg["sources"])
                conf = msg.get("confidence", {})
                kind_color = {"success": "#15803D", "warning": "#B45309", "danger": "#B91C1C"}.get(conf.get("kind"), "#B91C1C")
                st.markdown(
                    f'<div class="confidence-bar-bg"><div class="confidence-bar-fill" '
                    f'style="width:{conf.get("score",0)}%;background:{kind_color};"></div></div>'
                    f'<span style="font-size:11px;">{conf.get("label","")} ({conf.get("score",0)}%)</span>',
                    unsafe_allow_html=True,
                )

        if question:
            if not st.session_state["current_room"]:
                st.warning("Aucun data room disponible pour le moment.")
            elif not st.session_state["gateway"]:
                st.warning("Le service IA n'est pas disponible pour le moment. Réessayez plus tard ou "
                           "contactez l'administrateur de la plateforme.")
            else:
                is_injection, reason = detect_prompt_injection(question)
                masked_question, redactions = mask_pii(question)
                st.session_state["chat_history"].append({"role": "user", "content": question})

                if is_injection:
                    reponse_text = ("⚠️ Cette requête contient un motif potentiellement "
                                     "malveillant et a été bloquée par le module anti-injection.")
                    sources, confiance = [], {"score": 0, "label": "Bloqué", "kind": "danger"}
                else:
                    sources, contexte = gather_context(masked_question)
                    langue_cible = st.session_state["target_lang"]
                    lang_hint = "" if langue_cible == "Auto" else f"\nRéponds obligatoirement en {langue_cible}."

                    if use_multi_agent:
                        result = multi_agent_deliberation(masked_question, contexte, langue_cible,
                                                            st.session_state["gateway"])
                        reponse_text = result["reponse_finale"]
                        model_used, tok_in, tok_out, cost = "multi-agent", 0, 0, result.get("cout_total", 0)
                        latency = 0
                    else:
                        model = cascade_route(masked_question)
                        sys_prompt = build_system_prompt(lang_hint)
                        user_prompt = f"CONTEXTE :\n{contexte}\n\nQUESTION : {masked_question}"
                        result = call_llm_with_fallback(
                            st.session_state["gateway"],
                            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                            model=model,
                        )
                        reponse_text = result["text"]
                        model_used = result.get("model", model)
                        tok_in, tok_out = result.get("input_tokens", 0), result.get("output_tokens", 0)
                        cost, latency = result.get("cost_usd", 0), result.get("latency_ms", 0)

                    confiance = compute_confidence(sources)
                    log_audit(
                        user=st.session_state["role"], question=question, model=model_used,
                        tokens_in=tok_in, tokens_out=tok_out, cost_usd=cost, latency_ms=latency,
                        doc_source=st.session_state["current_room"], confidence=confiance["score"],
                        redactions=len(redactions),
                    )

                st.session_state["chat_history"].append({
                    "role": "assistant", "content": reponse_text,
                    "sources": sources, "confidence": confiance,
                })
                st.session_state["last_answer"] = reponse_text
                st.session_state["last_sources"] = sources
                st.rerun()

        if st.session_state["chat_history"] and st.session_state["chat_history"][-1]["role"] == "assistant":
            c1, c2, c3 = st.columns(3)
            if c1.button("👍 Utile"):
                log_feedback(st.session_state["chat_history"][-2]["content"],
                             st.session_state["chat_history"][-1]["content"], 1)
                st.toast("Merci pour votre retour !")
            if c2.button("👎 Pas utile"):
                log_feedback(st.session_state["chat_history"][-2]["content"],
                             st.session_state["chat_history"][-1]["content"], -1)
                st.toast("Merci, nous en tiendrons compte.")
            if c3.button("🔊 Écouter la réponse"):
                try:
                    from gtts import gTTS
                    tts_path = os.path.join(tempfile.gettempdir(), "reponse.mp3")
                    gTTS(text=st.session_state["last_answer"][:800], lang="fr").save(tts_path)
                    st.audio(tts_path)
                except Exception as e:
                    st.error(f"Synthèse vocale indisponible : {e}")

    with col_reader:
        st.markdown("#### Lecteur PDF — Citation Deep-Link")
        if st.session_state["current_room"]:
            room = st.session_state["data_rooms"][st.session_state["current_room"]]
            default_page = st.session_state["last_sources"][0]["page"] if st.session_state["last_sources"] else 1
            default_page = max(1, int(default_page) or 1)
            page_choisie = st.number_input("Page à afficher", min_value=1,
                                            max_value=max(1, len(room["pages"])), value=default_page)
            try:
                from pdf2image import convert_from_path
                img = convert_from_path(room["paths"][0], first_page=page_choisie, last_page=page_choisie)
                if img:
                    st.image(img[0], use_container_width=True)
            except Exception:
                page_data = next((p for p in room["pages"] if p["page"] == page_choisie), None)
                st.info("Aperçu image indisponible (poppler non installé sur ce poste) — texte extrait :")
                st.write(page_data["texte"] if page_data else "")
        else:
            st.caption("Aucun document chargé.")

# --- WORKSPACE 2 : DATA VISUALISATION ---------------------------------------
with tab2:
    st.caption("Tableaux de bord et requêtes SQL sur l'observatoire socio-économique marocain préchargé "
               "(loyers par région, CNOPS, RCAR, chômage HCP, comptes nationaux) et sur les tables extraites "
               "de vos propres documents.")

    st.markdown("#### 🗃️ Observatoire socio-économique (préchargé par défaut)")
    registered = st.session_state.get("stats_tables", [])
    if registered:
        labels = [f"{r['source_file']} — {r['sheet']} ({r['rows']} lignes)" for r in registered]
        choix_idx = st.selectbox("Jeu de données", range(len(labels)), format_func=lambda i: labels[i])
        table_meta = registered[choix_idx]
        df_preview = st.session_state["stats_con"].execute(f"SELECT * FROM {table_meta['table']} LIMIT 200").df()
        st.dataframe(df_preview, use_container_width=True)

        num_cols = df_preview.select_dtypes(include="number").columns.tolist()
        if num_cols:
            y_col = st.selectbox("Colonne numérique à visualiser", num_cols, key="stats_ycol")
            fig = px.bar(df_preview.reset_index(), x="index", y=y_col, template="simple_white",
                         title=f"{table_meta['source_file']} — {table_meta['sheet']}")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### 🧮 Text-to-SQL sur l'observatoire (DuckDB)")
        sql_question = st.text_input("Question analytique (ex : \"moyenne des loyers à Casablanca en 2022\")",
                                      key="stats_sql_q")
        if st.button("Exécuter en SQL", key="stats_sql_btn") and sql_question and st.session_state["gateway"]:
            table_names = [r["table"] for r in registered]
            sql, df_res, err = text_to_sql(sql_question, st.session_state["stats_con"], table_names,
                                            st.session_state["gateway"])
            st.code(sql, language="sql")
            if err:
                st.error(err)
            else:
                st.dataframe(df_res, use_container_width=True)
    else:
        st.info("Observatoire non disponible (dossier `data/` absent de ce déploiement).")

    st.divider()
    st.markdown("#### 📈 Marché des ingénieurs (aperçu rapide)")
    default_room = st.session_state["data_rooms"].get(DEFAULT_ROOM_NAME, {})
    market_tables = default_room.get("market_tables", {})
    if "market_ingenieurs" in market_tables:
        df_m = market_tables["market_ingenieurs"]
        colA, colB = st.columns(2)
        with colA:
            fig1 = px.bar(df_m.groupby("Secteur_Activite")["Salaire_Net_Mensuel_DH"].mean().reset_index(),
                          x="Secteur_Activite", y="Salaire_Net_Mensuel_DH", template="simple_white",
                          title="Salaire net mensuel moyen par secteur")
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)
        with colB:
            fig2 = px.bar(df_m.groupby("Ville")["Salaire_Net_Mensuel_DH"].mean().reset_index(),
                          x="Ville", y="Salaire_Net_Mensuel_DH", template="simple_white",
                          title="Salaire net mensuel moyen par ville")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("#### 🧠 Carte mentale & 📝 Quiz — document actif")
    room = st.session_state["data_rooms"].get(st.session_state["current_room"], {})
    colC, colD = st.columns(2)
    with colC:
        if st.button("🧠 Générer une carte mentale (Mermaid)") and st.session_state["gateway"]:
            echantillon = " ".join(p["texte"] for p in room.get("pages", [])[:5])[:2500]
            prompt = ("Génère un diagramme Mermaid (syntaxe 'graph TD') représentant la structure "
                      "hiérarchique des thèmes principaux de ce texte. Réponds uniquement avec le "
                      f"code Mermaid brut, sans balises markdown.\n\n{echantillon}")
            res = st.session_state["gateway"].chat([{"role": "user", "content": prompt}], model=MODEL_FAST)
            mermaid_code = res["text"].replace("```mermaid", "").replace("```", "").strip()
            components.html(f"""
                <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
                <div class="mermaid">{mermaid_code}</div>
                <script>mermaid.initialize({{startOnLoad:true}});</script>
            """, height=420, scrolling=True)

    with colD:
        if st.button("📝 Générer un QCM (5 questions)") and st.session_state["gateway"]:
            echantillon = " ".join(p["texte"] for p in room.get("pages", [])[:6])[:3000]
            prompt = ('Génère un QCM de 5 questions au format JSON strict : '
                      '[{"question":"...","options":["A","B","C","D"],"reponse_correcte":0}] '
                      f"à partir de ce texte :\n\n{echantillon}")
            res = st.session_state["gateway"].chat([{"role": "user", "content": prompt}], model=MODEL_PRO)
            try:
                st.session_state["quiz"] = json.loads(res["text"])
            except Exception:
                st.error("Le modèle n'a pas renvoyé un JSON valide — réessayez.")

    if "quiz" in st.session_state:
        score = 0
        for i, q in enumerate(st.session_state["quiz"]):
            choix = st.radio(q["question"], q["options"], key=f"quiz_{i}")
            if q["options"].index(choix) == q["reponse_correcte"]:
                score += 1
        st.markdown(f"**Score : {score} / {len(st.session_state['quiz'])}**")

# --- WORKSPACE 3 : DIFFERENTIAL ANALYSIS & COMPLIANCE -----------------------
with tab3:
    st.markdown("#### Analyse différentielle de PDF")
    c1, c2 = st.columns(2)
    file_a = c1.file_uploader("Version A", type=["pdf"], key="diff_a")
    file_b = c2.file_uploader("Version B", type=["pdf"], key="diff_b")

    if file_a and file_b and st.button("Comparer les versions"):
        path_a = os.path.join(tempfile.gettempdir(), "v_a_" + file_a.name)
        path_b = os.path.join(tempfile.gettempdir(), "v_b_" + file_b.name)
        with open(path_a, "wb") as f:
            f.write(file_a.getbuffer())
        with open(path_b, "wb") as f:
            f.write(file_b.getbuffer())
        doc_a, doc_b = extract_document(path_a), extract_document(path_b)
        diff_result = pdf_diff(doc_a["pages"], doc_b["pages"])
        any_diff = False
        for d in diff_result:
            if d["modifie"]:
                any_diff = True
                st.markdown(f"**Page {d['page']}** — modifications détectées")
                st.markdown(f'<div class="custom-card">{d["html"][:3000]}</div>', unsafe_allow_html=True)
        if not any_diff:
            st.success("Aucune différence textuelle détectée entre les deux versions.")

    st.divider()
    st.markdown("#### Vérification de conformité réglementaire")
    contrat_file = st.file_uploader("Document à vérifier (contrat, règlement interne...)", type=["pdf"], key="compliance")
    if contrat_file and st.button("Lancer l'analyse de conformité") and st.session_state["current_room"]:
        path_c = os.path.join(tempfile.gettempdir(), "compliance_" + contrat_file.name)
        with open(path_c, "wb") as f:
            f.write(contrat_file.getbuffer())
        doc_c = extract_document(path_c)
        texte_complet = " ".join(p["texte"] for p in doc_c["pages"])
        ref_chunks, _ = gather_context("obligations et clauses générales du contrat de travail", k=6)
        rapport = regulatory_compliance_check(texte_complet, ref_chunks, st.session_state["gateway"])
        st.markdown(f'<div class="custom-card">{rapport}</div>', unsafe_allow_html=True)

# --- WORKSPACE 4 : AUDIT LOG & EXECUTIVE EXPORTS ----------------------------
with tab4:
    st.markdown("#### Registre d'audit (conformité SOC2-ready)")
    st.dataframe(get_audit_log(), use_container_width=True)

    st.markdown("#### Boucle RLHF — Feedback utilisateurs")
    fb = get_feedback_log()
    if not fb.empty:
        st.metric("Note moyenne", round(fb["note"].mean(), 2))
        st.dataframe(fb, use_container_width=True)
    else:
        st.caption("Aucun feedback enregistré pour le moment.")

    st.markdown("#### Export exécutif")
    if st.session_state["last_answer"]:
        col1, col2 = st.columns(2)
        last_q = st.session_state["chat_history"][-2]["content"] if len(st.session_state["chat_history"]) >= 2 else ""
        docx_bytes = export_docx("Rapport L3amel Lfahem", last_q, st.session_state["last_answer"], st.session_state["last_sources"])
        pdf_bytes = export_pdf("Rapport L3amel Lfahem", last_q, st.session_state["last_answer"], st.session_state["last_sources"])
        col1.download_button("⬇️ Exporter en Word", docx_bytes, file_name="rapport_l3amel_lfahem.docx")
        col2.download_button("⬇️ Exporter en PDF", pdf_bytes, file_name="rapport_l3amel_lfahem.pdf")
    else:
        st.caption("Posez une question dans le Workspace 1 pour activer l'export.")

# --- WORKSPACE 5 : FINOPS COMMAND CENTER ------------------------------------
with tab5:
    st.caption("Pilotage financier de l'usage IA : coûts en temps réel, économies du cascade routing, "
               "projection budgétaire et empreinte énergétique indicative.")

    st.session_state["budget_usd"] = st.number_input(
        "🎯 Budget de la session (USD)", min_value=0.5, value=float(st.session_state["budget_usd"]), step=0.5)

    df_audit = get_audit_log()
    insights = compute_finops_insights(df_audit, budget_usd=st.session_state["budget_usd"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coût total", f"${insights['total_cost']}")
    c2.metric("Requêtes tracées", insights["total_calls"])
    c3.metric("Coût moyen / requête", f"${insights['avg_cost_per_call']}")
    c4.metric("Tokens moyens / requête", insights["avg_tokens_per_call"])

    st.markdown("##### 🎯 Consommation du budget")
    st.progress(min(1.0, insights["budget_pct"] / 100))
    st.caption(f"{insights['budget_pct']}% du budget de ${st.session_state['budget_usd']} consommé.")
    if insights["budget_pct"] >= 90:
        st.error("⚠️ Budget quasiment atteint — envisagez de relever le plafond ou de limiter le mode Multi-Agents.")
    elif insights["budget_pct"] >= 70:
        st.warning("Le budget de session est consommé à plus de 70%.")

    colE, colF = st.columns(2)
    with colE:
        st.markdown("##### 🚀 Économies du Cascade Routing")
        st.metric("Économisé vs tout-en-modèle-Pro",
                   f"${insights['cascade_savings_usd']}", f"{insights['cascade_savings_pct']}%")
        st.caption("Estimation : coût réel comparé à un scénario où chaque requête aurait été "
                   f"traitée par {MODEL_PRO} au lieu du routage intelligent rapide/raisonné.")
    with colF:
        st.markdown("##### 🌱 Empreinte énergétique (indicative)")
        st.metric("Énergie estimée", f"{insights['estimated_kwh']} kWh")
        st.caption("Ordre de grandeur basé sur une heuristique publique (~0,3 Wh / 1000 tokens) — "
                   "**non mesuré**, à titre de sensibilisation FinOps/Green AI uniquement.")

    st.markdown("##### 📈 Tendance des coûts")
    if not insights["trend"].empty:
        fig_trend = px.line(insights["trend"], x="horodatage", y="cout_usd", template="simple_white",
                             title="Coût quotidien (USD)")
        fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.caption("Pas encore assez de données pour tracer une tendance — posez quelques questions au chatbot.")

    st.markdown("##### 🧮 Répartition par modèle")
    if not insights["by_model"].empty:
        fig_model = px.bar(insights["by_model"], x="modele", y="cout_usd", template="simple_white",
                            title="Coût cumulé par modèle")
        fig_model.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_model, use_container_width=True)
        st.dataframe(insights["by_model"], use_container_width=True)

    st.markdown("##### 🔮 Projection mensuelle")
    st.metric("Coût mensuel projeté (au rythme actuel)", f"${insights['projected_monthly_usd']}")
    st.caption("Extrapolation linéaire à partir du rythme de dépense observé sur la période couverte "
               "par le registre d'audit — indicatif, pas un engagement contractuel.")
