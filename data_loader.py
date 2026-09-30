# -*- coding: utf-8 -*-
"""
data_loader.py — Chargement de la base documentaire ET statistique PAR DÉFAUT
de L3amel Lfahem, afin que l'application soit utilisable immédiatement, sans
qu'il soit nécessaire de téléverser quoi que ce soit au premier lancement.

Deux corpus distincts, chacun mis en cache serveur (st.cache_resource) pour
n'être calculé qu'une seule fois, quel que soit le nombre de sessions :

1. Corpus RAG "Code du travail & Marché Ingénieurs" :
   - Texte intégral du Code du travail marocain (chunké, indexé FAISS+BM25).
   - Statistiques agrégées du marché de l'emploi des ingénieurs (offres,
     salaires) et de l'insertion professionnelle (stages ANAPEC), transformées
     en chunks narratifs pour être citables comme n'importe quelle source RAG.

2. Corpus Analytics "Observatoire socio-économique" :
   - Toutes les feuilles Excel officielles fournies (loyers par région, CNOPS,
     RCAR, chômage HCP, comptes nationaux...) chargées automatiquement en
     tables DuckDB grâce à un détecteur d'en-tête générique, car ces fichiers
     utilisent des mises en page statistiques non standard (en-têtes fusionnés,
     bilingues FR/AR, lignes de titre).
"""

import os
import re
from typing import List, Tuple

import pandas as pd
import duckdb
import streamlit as st

from rag_engine import extract_document, build_chunks, SearchIndex

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CODE_TRAVAIL_PDF = os.path.join(DATA_DIR, "Code_du_travail.pdf")
MARKET_CSV = os.path.join(DATA_DIR, "market_ingenieurs_maroc_10k_advanced.csv")
ANAPEC_CSV = os.path.join(DATA_DIR, "anapec_stages_insertion_maroc_10k.csv")

DEFAULT_ROOM_NAME = "Code du travail & Marché Ingénieurs"


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", str(text)).strip("_").lower()
    return re.sub(r"_+", "_", s) or "table"


# ==============================================================================
# 1. STATISTIQUES OFFICIELLES (XLSX/XLS) — détecteur d'en-tête générique
# ==============================================================================

def _smart_read_sheet(path: str, sheet_name: str) -> pd.DataFrame:
    """Les tableaux HCP/CNOPS/RCAR placent l'en-tête réel quelques lignes plus
    bas, précédé de titres/métadonnées. On repère la ligne la plus dense en
    cellules non vides pour s'en servir d'en-tête, sans jamais planter."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    if raw.empty:
        return pd.DataFrame()

    best_row, best_score = 0, -1
    for i in range(min(15, len(raw))):
        row = raw.iloc[i]
        score = row.notna().sum() + sum(isinstance(v, str) and len(v.strip()) > 0 for v in row)
        if score > best_score:
            best_score, best_row = score, i

    header_vals = raw.iloc[best_row].fillna("").astype(str).str.strip().tolist()
    seen, header = {}, []
    for i, h in enumerate(header_vals):
        h = h if h else f"col_{i}"
        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"
        else:
            seen[h] = 0
        header.append(h)

    df = raw.iloc[best_row + 1:].reset_index(drop=True)
    df.columns = header
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df


@st.cache_resource(show_spinner="📊 Indexation de l'observatoire socio-économique (chargement unique, mis en cache)...")
def load_default_stats_db() -> Tuple[duckdb.DuckDBPyConnection, List[dict]]:
    con = duckdb.connect(":memory:")
    registered = []
    if not os.path.isdir(DATA_DIR):
        return con, registered

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.lower().endswith((".xlsx", ".xls")):
            continue
        path = os.path.join(DATA_DIR, fname)
        try:
            xl = pd.ExcelFile(path)
        except Exception:
            continue
        for sheet in xl.sheet_names:
            try:
                df = _smart_read_sheet(path, sheet)
                if df.empty or df.shape[1] < 2 or df.shape[0] < 1:
                    continue
                table_name = _slug(f"{os.path.splitext(fname)[0]}_{sheet}")[:63]
                con.register(table_name, df)
                registered.append({
                    "table": table_name, "source_file": fname, "sheet": sheet,
                    "rows": len(df), "columns": list(df.columns),
                })
            except Exception:
                continue
    return con, registered


# ==============================================================================
# 2. MARCHÉ DES INGÉNIEURS — agrégats narratifs injectés dans le RAG
# ==============================================================================

def _chunk(cid: int, source_file: str, article: str, texte: str) -> dict:
    """Formate un agrégat statistique comme un chunk compatible SearchIndex."""
    return {
        "id": cid, "parent_id": f"stat_{cid}", "data_room": DEFAULT_ROOM_NAME,
        "source_file": source_file, "page": 0, "article": article,
        "texte_enfant": texte, "texte_parent": texte,
    }


def _summarize_engineers_market(df: pd.DataFrame, start_id: int) -> List[dict]:
    chunks, cid = [], start_id
    src = "market_ingenieurs_maroc_10k_advanced.csv"

    par_secteur = df.groupby("Secteur_Activite")["Salaire_Net_Mensuel_DH"].agg(["mean", "count"]).round(0)
    lignes = [f"{sec} : salaire net mensuel moyen {int(r['mean'])} DH ({int(r['count'])} offres)"
              for sec, r in par_secteur.iterrows()]
    chunks.append(_chunk(cid, src, "Statistique — Salaires par secteur",
                          "Marché de l'emploi des ingénieurs au Maroc — salaire net mensuel moyen par secteur "
                          f"d'activité (échantillon de {len(df)} offres) : " + " ; ".join(lignes) + "."))
    cid += 1

    par_filiere = df.groupby("Filiere")["Salaire_Net_Mensuel_DH"].agg(["mean", "count"]).round(0)
    lignes = [f"{f} : {int(r['mean'])} DH/mois en moyenne ({int(r['count'])} offres)"
              for f, r in par_filiere.iterrows()]
    chunks.append(_chunk(cid, src, "Statistique — Salaires par filière d'ingénierie",
                          "Salaire net mensuel moyen des ingénieurs par filière de formation : "
                          + " ; ".join(lignes) + "."))
    cid += 1

    par_exp = df.groupby("Experience_Requise")["Salaire_Net_Mensuel_DH"].agg(["mean", "count"]).round(0)
    lignes = [f"{e} : {int(r['mean'])} DH/mois ({int(r['count'])} offres)" for e, r in par_exp.iterrows()]
    chunks.append(_chunk(cid, src, "Statistique — Salaires par niveau d'expérience",
                          "Salaire net mensuel moyen des ingénieurs selon le niveau d'expérience requis : "
                          + " ; ".join(lignes) + "."))
    cid += 1

    par_ville = df.groupby("Ville")["Salaire_Net_Mensuel_DH"].agg(["mean", "count"]).round(0)
    lignes = [f"{v} : {int(r['mean'])} DH/mois ({int(r['count'])} offres)" for v, r in par_ville.iterrows()]
    chunks.append(_chunk(cid, src, "Statistique — Salaires par ville",
                          "Salaire net mensuel moyen des ingénieurs par ville au Maroc : "
                          + " ; ".join(lignes) + "."))
    cid += 1

    mode_travail = df["Mode_Travail"].value_counts(normalize=True).mul(100).round(1)
    lignes = [f"{m} : {p}% des offres" for m, p in mode_travail.items()]
    chunks.append(_chunk(cid, src, "Statistique — Mode de travail",
                          "Répartition des offres d'emploi d'ingénieurs selon le mode de travail : "
                          + " ; ".join(lignes) + "."))
    cid += 1

    contrat = df["Type_Contrat"].value_counts(normalize=True).mul(100).round(1)
    lignes = [f"{c} : {p}%" for c, p in contrat.items()]
    chunks.append(_chunk(cid, src, "Statistique — Types de contrat",
                          "Répartition des offres d'emploi d'ingénieurs par type de contrat (échantillon "
                          f"de {len(df)} offres) : " + " ; ".join(lignes) + "."))
    cid += 1

    return chunks


def _summarize_anapec_insertion(df: pd.DataFrame, start_id: int) -> List[dict]:
    chunks, cid = [], start_id
    src = "anapec_stages_insertion_maroc_10k.csv"

    par_dispositif = df.groupby("Type_Dispositif")["Gratification_Mensuelle_DH"].agg(["mean", "count"]).round(0)
    lignes = [f"{d} : gratification moyenne {int(r['mean'])} DH/mois ({int(r['count'])} dossiers)"
              for d, r in par_dispositif.iterrows()]
    chunks.append(_chunk(cid, src, "Statistique — Dispositifs ANAPEC pour ingénieurs",
                          "Insertion professionnelle des jeunes ingénieurs via l'ANAPEC — gratification "
                          f"mensuelle moyenne par type de dispositif (échantillon de {len(df)} dossiers) : "
                          + " ; ".join(lignes) + ". Ces dispositifs bénéficient généralement d'une "
                          "exonération d'IR et d'une prise en charge CNSS par l'État, conformément au "
                          "cadre légal marocain d'incitation à l'emploi des jeunes diplômés."))
    cid += 1

    par_filiere = df.groupby("Filiere")["Duree_Mois"].mean().round(1)
    lignes = [f"{f} : durée moyenne {d} mois" for f, d in par_filiere.items()]
    chunks.append(_chunk(cid, src, "Statistique — Durée des stages/contrats d'insertion par filière",
                          "Durée moyenne des stages et contrats d'insertion ANAPEC pour ingénieurs, par "
                          "filière de formation : " + " ; ".join(lignes) + "."))
    cid += 1

    preembauche = df["Option_Preembauche"].value_counts(normalize=True).mul(100).round(1)
    lignes = [f"{o} : {p}%" for o, p in preembauche.items()]
    chunks.append(_chunk(cid, src, "Statistique — Taux de pré-embauche",
                          "Répartition des dossiers ANAPEC selon l'option de pré-embauche déclarée : "
                          + " ; ".join(lignes) + "."))
    cid += 1

    statut = df["Statut_Dossier"].value_counts(normalize=True).mul(100).round(1)
    lignes = [f"{s} : {p}%" for s, p in statut.items()]
    chunks.append(_chunk(cid, src, "Statistique — Statut des dossiers d'insertion",
                          f"Sur un échantillon de {len(df)} dossiers d'insertion ANAPEC pour ingénieurs, "
                          "répartition par statut : " + " ; ".join(lignes) + "."))
    cid += 1

    return chunks


@st.cache_resource(show_spinner="⚖️ Indexation du Code du travail et du marché de l'emploi des ingénieurs "
                                  "(chargement unique, mis en cache)...")
def load_default_legal_room() -> dict:
    if not os.path.exists(CODE_TRAVAIL_PDF):
        return None

    doc = extract_document(CODE_TRAVAIL_PDF)
    chunks = build_chunks(doc["pages"], data_room=DEFAULT_ROOM_NAME)
    next_id = (max((c["id"] for c in chunks), default=-1)) + 1

    market_tables = {}
    if os.path.exists(MARKET_CSV):
        df_market = pd.read_csv(MARKET_CSV)
        market_tables["market_ingenieurs"] = df_market
        chunks.extend(_summarize_engineers_market(df_market, next_id))
        next_id = max(c["id"] for c in chunks) + 1
    if os.path.exists(ANAPEC_CSV):
        df_anapec = pd.read_csv(ANAPEC_CSV)
        market_tables["anapec_stages"] = df_anapec
        chunks.extend(_summarize_anapec_insertion(df_anapec, next_id))

    index = SearchIndex(chunks)

    return {
        "pages": doc["pages"], "tables": doc["tables"], "chunks": chunks, "index": index,
        "access": "public", "paths": [CODE_TRAVAIL_PDF], "anomalies": doc["anomalies"],
        "market_tables": market_tables, "is_default": True,
    }
