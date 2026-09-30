# -*- coding: utf-8 -*-
"""
agents.py — Orchestration des agents pour L3amel Lfahem.

Contient :
  - Cascade Routing (modèle rapide vs modèle raisonné selon la complexité)
  - Fallback LLM local (Ollama) si le réseau / l'API Groq échoue
  - Mode Multi-Agents (Agent Juridique, Agent Financier, Agent Synthèse)
  - Analyse de sentiment / ton du document
  - Traduction à la volée des extraits cités
  - Vérification de cohérence réglementaire
"""

import json
import requests
from typing import List, Dict, Optional

from rag_engine import FinOpsGateway, MODEL_FAST, MODEL_PRO, build_system_prompt

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # modèle local à adapter selon ce qui est disponible via `ollama pull`


# ==============================================================================
# DOMAINE 6.5 — DYNAMIC CASCADE ROUTING
# ==============================================================================

COMPLEX_KEYWORDS = [
    "comparer", "compare", "différence", "calcule", "calcul", "plusieurs",
    "et aussi", "en plus de", "شنو الفرق", "قارن",
]


def cascade_route(question: str) -> str:
    """Heuristique de routage : question longue / multi-parties / mots-clés analytiques
    → modèle raisonné (Pro) ; sinon → modèle rapide (Fast)."""
    q = question.lower()
    long_question = len(question.split()) > 25
    multi_question = question.count("?") > 1 or " et " in q
    complexe = any(kw in q for kw in COMPLEX_KEYWORDS)
    if long_question or multi_question or complexe:
        return MODEL_PRO
    return MODEL_FAST


# ==============================================================================
# DOMAINE 7.4 — LOCAL LLM FALLBACK (mode déconnecté / souverain)
# ==============================================================================

def call_llm_with_fallback(gateway: FinOpsGateway, messages: List[dict], model: str,
                            temperature: float = 0.2) -> dict:
    try:
        return gateway.chat(messages, model=model, temperature=temperature)
    except Exception as groq_error:
        try:
            prompt = "\n".join(m["content"] for m in messages)
            resp = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "text": data.get("response", ""),
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "latency_ms": 0.0, "model": f"local::{OLLAMA_MODEL}",
                "fallback": True,
            }
        except Exception as ollama_error:
            return {
                "text": (
                    "⚠️ Le service est momentanément indisponible (API Groq inaccessible et "
                    "aucun modèle local Ollama détecté sur ce poste). "
                    f"Détail : {groq_error}"
                ),
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "latency_ms": 0.0, "model": "aucun", "fallback": True, "error": True,
            }


# ==============================================================================
# DOMAINE 7.1 — MODE MULTI-AGENTS SPÉCIALISÉS (AGENTIC RAG)
# ==============================================================================

AGENT_JURIDIQUE_PROMPT = """Tu es l'Agent Juridique de L3amel Lfahem, spécialisé dans
l'interprétation stricte des articles du Code du travail marocain. Analyse le contexte
fourni sous l'angle des droits, obligations, délais légaux et procédures. Cite les
articles précis. Sois concis (5 lignes maximum)."""

AGENT_FINANCIER_PROMPT = """Tu es l'Agent Financier de L3amel Lfahem, spécialisé dans les
implications chiffrées : indemnités, salaires, primes, cotisations, pénalités. Analyse le
contexte fourni sous cet angle uniquement. Si aucune donnée chiffrée n'est pertinente,
dis-le explicitement. Sois concis (5 lignes maximum)."""

AGENT_SYNTHESE_PROMPT = """Tu es l'Agent de Synthèse de L3amel Lfahem. Tu reçois l'avis de
l'Agent Juridique et de l'Agent Financier sur la même question. Fusionne-les en une réponse
unique, claire, structurée, dans la langue de la question d'origine, en conservant les
citations (Page X, Article Y) et en respectant les consignes de non-conseil-juridique-engageant."""


def multi_agent_deliberation(question: str, contexte: str, langue_detectee: str,
                              gateway: FinOpsGateway) -> dict:
    """Fait débattre 3 agents spécialisés puis produit une réponse unifiée."""
    base_context = f"CONTEXTE DOCUMENTAIRE :\n{contexte}\n\nQUESTION : {question}"

    avis_juridique = call_llm_with_fallback(
        gateway,
        [{"role": "system", "content": AGENT_JURIDIQUE_PROMPT}, {"role": "user", "content": base_context}],
        model=MODEL_FAST,
    )
    avis_financier = call_llm_with_fallback(
        gateway,
        [{"role": "system", "content": AGENT_FINANCIER_PROMPT}, {"role": "user", "content": base_context}],
        model=MODEL_FAST,
    )

    synth_prompt = (
        f"Langue de réponse attendue : {langue_detectee}\n\n"
        f"Avis de l'Agent Juridique :\n{avis_juridique['text']}\n\n"
        f"Avis de l'Agent Financier :\n{avis_financier['text']}\n\n"
        f"Question originale : {question}"
    )
    avis_synthese = call_llm_with_fallback(
        gateway,
        [{"role": "system", "content": AGENT_SYNTHESE_PROMPT + "\n" + build_system_prompt()},
         {"role": "user", "content": synth_prompt}],
        model=MODEL_PRO,
    )

    total_tokens = sum(a.get("input_tokens", 0) + a.get("output_tokens", 0)
                        for a in [avis_juridique, avis_financier, avis_synthese])
    total_cost = sum(a.get("cost_usd", 0) for a in [avis_juridique, avis_financier, avis_synthese])

    return {
        "reponse_finale": avis_synthese["text"],
        "avis_juridique": avis_juridique["text"],
        "avis_financier": avis_financier["text"],
        "tokens_total": total_tokens,
        "cout_total": total_cost,
    }


# ==============================================================================
# DOMAINE 2.3 — ANALYSE DE SENTIMENT & TON DU DOCUMENT
# ==============================================================================

def analyze_sentiment(texte_echantillon: str, gateway: FinOpsGateway) -> dict:
    prompt = (
        "Classe le ton dominant de cet extrait de document en UNE des catégories suivantes : "
        "Neutre/Administratif, Réclamation/Plainte, Note critique, Rapport positif, Procédure formelle. "
        'Réponds en JSON strict: {"ton": "...", "justification": "..."}\n\n'
        f"Extrait : {texte_echantillon[:1500]}"
    )
    result = gateway.chat([{"role": "user", "content": prompt}], model=MODEL_FAST, temperature=0.0)
    try:
        return json.loads(result["text"])
    except Exception:
        return {"ton": "Indéterminé", "justification": result["text"][:200]}


# ==============================================================================
# DOMAINE 2.4 — TRADUCTION À LA VOLÉE DES EXTRAITS CITÉS
# ==============================================================================

def translate_citation(texte: str, langue_cible: str, gateway: FinOpsGateway) -> str:
    prompt = f"Traduis fidèlement ce texte juridique vers le {langue_cible}, sans commentaire ajouté :\n\n{texte}"
    result = gateway.chat([{"role": "user", "content": prompt}], model=MODEL_FAST, temperature=0.0)
    return result["text"]


# ==============================================================================
# DOMAINE 7.3 — ANALYSE DE COHÉRENCE RÉGLEMENTAIRE
# ==============================================================================

def regulatory_compliance_check(document_text: str, reference_chunks: List[dict],
                                 gateway: FinOpsGateway) -> str:
    reference = "\n---\n".join(
        f"({c['article']}, Page {c['page']}) {c['texte_enfant']}" for c in reference_chunks
    )
    prompt = (
        "Tu es l'Agent de Conformité Réglementaire. Compare le document ci-dessous aux extraits "
        "de référence du Code du travail marocain fournis. Identifie les clauses potentiellement "
        "non conformes ou à risque, avec la référence exacte de l'article concerné. Si tout semble "
        "conforme au regard des extraits fournis, dis-le explicitement. Structure ta réponse en "
        "une liste à puces.\n\n"
        f"RÉFÉRENTIEL :\n{reference}\n\nDOCUMENT À ANALYSER :\n{document_text[:4000]}"
    )
    result = call_llm_with_fallback(
        gateway, [{"role": "system", "content": build_system_prompt()}, {"role": "user", "content": prompt}],
        model=MODEL_PRO,
    )
    return result["text"]
