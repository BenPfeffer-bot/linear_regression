"""
Streamlit Dashboard — Hedge Fund Clone Analysis
Aberdeen Orbita Capital Return Strategy Ltd A (GBP)

On essaie de cloner un hedge fund market-neutral avec des ETFs publics.
Spoiler : le fund garde bien ses secrets (c'est le jeu du market-neutral).
"""

from __future__ import annotations
import warnings
import time
import itertools
import io
import sys

# On fait taire les warnings sklearn/statsmodels -- y'en a beaucoup trop
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yfinance as yf

from importlib.machinery import SourceFileLoader

# Le tiret dans "mini-project.py" empêche un import classique,
# du coup on triche avec SourceFileLoader (merci Python)
_mp = SourceFileLoader("mini_project", "mini-project.py").load_module()

# On récupère tout ce qu'il nous faut du moteur principal
ABERDEEN_MONTHLY = _mp.ABERDEEN_MONTHLY  # rendements mensuels du fund (hardcodés)
UNIVERSE = _mp.UNIVERSE  # dict {catégorie: [tickers]} -- ~88 ETFs
ALL_TICKERS = _mp.ALL_TICKERS  # liste plate de tous les tickers
START_DATE = _mp.START_DATE
END_DATE = _mp.END_DATE
TARGET_R2 = _mp.TARGET_R2  # objectif R² ajusté -- ambitieux vu que c'est du market-neutral
MAX_FACTORS = _mp.MAX_FACTORS  # nb max de facteurs qu'on s'autorise
P_THRESHOLD = _mp.P_THRESHOLD  # seuil p-value pour le stepwise

# ── Helpers ──────────────────────────────────────────────────────────────────
# Métriques de perf classiques, annualisées depuis du mensuel


def sharpe(r):
    """Sharpe annualisé -- on suppose rf=0, c'est une approximation qui passe"""
    return r.mean() / r.std() * np.sqrt(12)


def ann_ret(r):
    """Rendement annualisé par composition géométrique"""
    return (1 + r.mean()) ** 12 - 1


def ann_vol(r):
    """Volatilité annualisée -- la stat qu'on regarde en premier"""
    return r.std() * np.sqrt(12)


def max_dd(r):
    """Max drawdown -- personne aime voir ce chiffre en rouge"""
    cum = (1 + r).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()


# Thème sombre Plotly -- fond transparent pour s'intégrer au dark mode Streamlit
DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,25,0.9)",
)

# ── Page config ──────────────────────────────────────────────────────────────
# Layout wide pour profiter de l'écran, sidebar fermée
# (le dashboard est non-interactif, tout se lance au chargement)
st.set_page_config(
    page_title="HF Clone — Aberdeen Orbita",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS custom : terminal vert style Matrix + cartes KPI façon GitHub Dark
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');
    .main .block-container { padding-top: 1.5rem; max-width: 1500px; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    .terminal {
        background: #0c0c0c;
        color: #00ff41;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 20px 24px;
        border-radius: 12px;
        border: 1px solid #1a3a1a;
        max-height: 520px;
        overflow-y: auto;
        line-height: 1.6;
        white-space: pre-wrap;
        box-shadow: 0 0 30px rgba(0,255,65,0.05);
    }
    .terminal .dim { color: #4a7a4a; }
    .terminal .cyan { color: #4fc3f7; }
    .terminal .yellow { color: #ffb74d; }
    .terminal .red { color: #e57373; }
    .terminal .bold { font-weight: 700; }
    .terminal .white { color: #e0e0e0; }
    .kpi {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .kpi .val {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        line-height: 1.1;
    }
    .kpi .lbl {
        font-size: 0.75rem;
        opacity: 0.5;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Run analysis (cached) ───────────────────────────────────────────────────
# Le @st.cache_data évite de re-télécharger 88 tickers à chaque refresh
# (sinon c'est ~30s d'attente à chaque fois, pas ouf)
@st.cache_data(show_spinner=False)
def run_analysis():
    """Pipeline complet : download -> feature engineering -> stepwise OLS.
    Retourne un dict avec le modèle, résidus, historique, etc.
    + un log qu'on affichera dans le terminal façon hacker."""
    log = []

    def p(msg):
        """Raccourci pour logger -- chaque ligne finit dans le terminal vert"""
        log.append(msg)

    # En-tête du log, ambiance CLI old-school
    p("=" * 72)
    p("  HEDGE FUND CLONE ENGINE")
    p("  Aberdeen Orbita Capital Return Strategy Ltd A (GBP)")
    p("=" * 72)
    p(f"  Period        : {START_DATE} -> {END_DATE}")
    p(f"  Target R2adj  : {TARGET_R2 * 100:.0f}%")
    p(f"  Max factors   : {MAX_FACTORS}")
    p(f"  P-threshold   : {P_THRESHOLD}")
    p(
        f"  Universe      : {len(ALL_TICKERS)} tickers across {len(UNIVERSE)} asset classes"
    )
    p("")

    # ── Download ─────────────────────────────────────────────────────────
    # On télécharge par batch de 30 pour pas se faire rate-limit par Yahoo
    p("-" * 72)
    p(f"  [1/6] Downloading {len(ALL_TICKERS)} tickers from Yahoo Finance...")
    p("-" * 72)

    all_prices = []
    failed = []
    batch_size = 30
    for i in range(0, len(ALL_TICKERS), batch_size):
        batch = ALL_TICKERS[i : i + batch_size]
        batch_num = i // batch_size + 1
        p(f"    Batch {batch_num}: {len(batch)} tickers...")
        try:
            raw = yf.download(
                batch,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            # yfinance renvoie un MultiIndex quand on demande plusieurs tickers d'un coup
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]] if "Close" in raw.columns else raw
            all_prices.append(close)
        except Exception as e:
            p(f"    WARNING: Batch failed - {e}")
            failed.extend(batch)
        # Petite pause pour pas énerver l'API
        time.sleep(0.3)

    # Assemblage : on colle tout et on passe en rendements mensuels
    prices_daily = pd.concat(all_prices, axis=1)
    prices_daily = prices_daily.loc[:, ~prices_daily.columns.duplicated()]
    prices_monthly = prices_daily.resample("ME").last()
    returns_monthly = prices_monthly.pct_change().dropna(how="all")

    # On vire les ETFs avec trop de trous (délistés, lancés trop tard, etc.)
    threshold = int(0.80 * len(returns_monthly))
    returns_monthly = returns_monthly.dropna(axis=1, thresh=threshold)

    p(
        f"    OK: {returns_monthly.shape[1]} assets loaded | {len(failed)} failed | {returns_monthly.shape[0]} months"
    )
    p("")

    # ── Align ────────────────────────────────────────────────────────────
    # Les rendements Aberdeen sont en dur dans mini-project.py
    # Faut les caler sur les dates du marché
    p("-" * 72)
    p("  [2/6] Aligning Aberdeen returns with market data...")
    p("-" * 72)

    target_dates = returns_monthly.index
    n_align = min(len(ABERDEEN_MONTHLY), len(target_dates))
    y = pd.Series(
        ABERDEEN_MONTHLY[-n_align:], index=target_dates[-n_align:], name="Aberdeen"
    )
    X_raw = returns_monthly.loc[y.index].copy()
    # On garde que les colonnes avec au moins 90% de données valides
    X_raw = X_raw.dropna(axis=1, thresh=int(0.90 * len(y)))
    # Les NaN restants -> médiane (pas parfait mais ça fait le job)
    X_raw = X_raw.fillna(X_raw.median())

    p(f"    Aberdeen : {len(y)} obs ({y.index[0].date()} -> {y.index[-1].date()})")
    p(f"    Factors  : {X_raw.shape[1]} ETF/indices")
    p(
        f"    Ann. Return : {ann_ret(y) * 100:.2f}%  |  Sharpe : {sharpe(y):.2f}  |  MaxDD : {max_dd(y) * 100:.2f}%"
    )
    p("")

    # ── Correlations ─────────────────────────────────────────────────────
    # Premier coup d'oeil : quels ETFs bougent comme le fund ?
    corr_raw = X_raw.corrwith(y).abs().sort_values(ascending=False)
    p("-" * 72)
    p("  [3/6] Top 10 raw correlations with fund:")
    p("-" * 72)
    for ticker, corr_val in corr_raw.head(10).items():
        bar = "#" * int(corr_val * 80)
        p(f"    {ticker:<8} {corr_val:.4f}  {bar}")
    p("")

    # ── Feature engineering ──────────────────────────────────────────────
    # C'est ici que ça devient intéressant : on crée des features dérivées
    # pour capturer les effets retardés, les tendances, et la non-linéarité
    p("-" * 72)
    p("  [4/6] Feature engineering...")
    p("-" * 72)

    X_enrich = X_raw.copy()

    # Lags t-1 et t-2 : le fund réagit peut-être avec du retard
    for col in X_raw.columns:
        X_enrich[f"{col}_lag1"] = X_raw[col].shift(1)
        X_enrich[f"{col}_lag2"] = X_raw[col].shift(2)

    # Moyennes mobiles 3m et 6m sur les 25 plus corrélés -- lisse le bruit
    for col in corr_raw.head(25).index:
        X_enrich[f"{col}_ma3"] = X_raw[col].rolling(3).mean()
        X_enrich[f"{col}_ma6"] = X_raw[col].rolling(6).mean()

    # Volatilité réalisée 3m -- les HF sont sensibles au régime de vol
    for col in corr_raw.head(15).index:
        X_enrich[f"{col}_vol3"] = X_raw[col].rolling(3).std()

    # Termes quadratiques -- pour capter la convexité (effet gamma)
    for col in corr_raw.head(25).index:
        X_enrich[f"{col}_sq"] = X_raw[col] ** 2

    # Interactions croisées top-12 : genre XLE * GDX pour l'effet commodities
    top_interact = corr_raw.head(12).index.tolist()
    for a, b in itertools.combinations(top_interact, 2):
        X_enrich[f"{a}x{b}"] = X_raw[a] * X_raw[b]

    # On perd quelques lignes à cause des lags/rolling -- c'est le prix à payer
    X_enrich = X_enrich.dropna()
    y_enrich = y.loc[X_enrich.index]
    n_before = X_enrich.shape[1]

    # Déduplication : on vire les features corrélées >85% entre elles
    # pour éviter la multicolinéarité (sinon les VIF explosent)
    # Entre deux features redondantes, on garde celle qui corrèle le plus avec y
    corr_matrix = X_enrich.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        highly_corr = upper.index[upper[col] > 0.85].tolist()
        if highly_corr:
            candidates = [col] + highly_corr
            corrs_y = {
                c: abs(X_enrich[c].corr(y_enrich))
                for c in candidates
                if c not in to_drop
            }
            if corrs_y:
                keep = max(corrs_y, key=corrs_y.get)
                to_drop.update(c for c in candidates if c != keep)
    X_enrich = X_enrich.drop(columns=list(to_drop))

    # PCA : composantes orthogonales pour capter des patterns
    # que les features brutes ratent -- en gros, des "macro factors" synthétiques
    scaler_pca = StandardScaler()
    X_sc = scaler_pca.fit_transform(X_enrich)
    n_comp = min(30, X_enrich.shape[1], X_enrich.shape[0] - 1)
    pca_obj = PCA(n_components=n_comp)
    pca_scores = pca_obj.fit_transform(X_sc)
    for i in range(n_comp):
        X_enrich[f"PC{i + 1}"] = pca_scores[:, i]

    p(f"    Raw features     : {n_before}")
    p(f"    After dedup      : {X_enrich.shape[1] - n_comp}")
    p(f"    + PCA components : {n_comp}")
    p(f"    Total candidates : {X_enrich.shape[1]}")
    p(f"    Observations     : {len(y_enrich)}")
    p("")

    # ── Forward stepwise ─────────────────────────────────────────────────
    # La boucle principale : on ajoute un facteur à la fois, celui qui
    # améliore le plus le R² ajusté tout en restant significatif.
    # C'est greedy (pas optimal), mais en pratique ça marche bien
    p("-" * 72)
    p(f"  [5/6] Forward stepwise (target R2adj >= {TARGET_R2 * 100:.0f}%)")
    p("-" * 72)
    p(f"  {'Step':<6} {'Factor':<28} {'R2adj':>8}  {'p-value':>10}")
    p(
        f"  {'----':<6} {'----------------------------':<28} {'--------':>8}  {'----------':>10}"
    )

    remaining = list(X_enrich.columns)
    selected = []
    best_r2 = 0.0
    history = []

    for step in range(MAX_FACTORS):
        best_new = None
        best_new_r2 = best_r2
        best_p = 1.0
        # On teste chaque candidat restant -- oui c'est O(n*k), c'est pas grave
        for cand in remaining:
            try:
                res = sm.OLS(
                    y_enrich, sm.add_constant(X_enrich[selected + [cand]])
                ).fit()
                pval = float(res.pvalues.get(cand, 1.0))
                # Double condition : améliore R²adj ET significatif
                if res.rsquared_adj > best_new_r2 and pval < P_THRESHOLD:
                    best_new_r2 = res.rsquared_adj
                    best_new = cand
                    best_p = pval
            except Exception:
                continue
        if best_new is None:
            p("  -> Stopped: no significant factor found")
            break
        selected.append(best_new)
        remaining.remove(best_new)
        best_r2 = best_new_r2
        hit = " TARGET HIT" if best_r2 >= TARGET_R2 else ""
        p(f"  {step + 1:<6} {best_new:<28} {best_r2:>8.4f}  {best_p:>10.6f}{hit}")
        history.append(
            {
                "step": step + 1,
                "factor": best_new,
                "r2_adj": best_new_r2,
                "p_value": best_p,
            }
        )
        if best_r2 >= TARGET_R2:
            break

    p("")

    # ── Final model ──────────────────────────────────────────────────────
    # On refit le modèle final avec les facteurs sélectionnés
    # et on calcule tous les diagnostics OLS qu'un prof voudrait voir
    p("-" * 72)
    p("  [6/6] Final OLS model & diagnostics")
    p("-" * 72)

    X_final = sm.add_constant(X_enrich[selected])
    model = sm.OLS(y_enrich, X_final).fit()
    y_hat = model.fittedvalues.rename("Clone")
    resid = model.resid

    # Durbin-Watson : ~2 = pas d'autocorrélation des résidus (on veut entre 1.5 et 2.5)
    dw = durbin_watson(resid)
    # Breusch-Pagan : p>0.05 = variance des résidus constante (homoscédasticité)
    _, bp_p, _, _ = het_breuschpagan(resid, X_final)
    # Jarque-Bera : p>0.05 = résidus normalement distribués
    jb_stat, jb_p = stats.jarque_bera(resid)

    # VIF (Variance Inflation Factor) : si >5 on a de la multicolinéarité
    # qui gonfle artificiellement les erreurs standard des coefficients
    vif_vals = [
        variance_inflation_factor(X_final.values, i) for i in range(X_final.shape[1])
    ]
    vif_df = pd.DataFrame({"Variable": X_final.columns, "VIF": vif_vals})
    vif_df = vif_df[vif_df["Variable"] != "const"]

    # Poids du portefeuille clone, normalisés pour que la somme des |poids| = 100%
    params = model.params.drop("const")
    weights = params / params.abs().sum() * 100

    # L'alpha = la constante OLS = ce que le fund génère "au-delà" des facteurs
    # C'est le vrai talent du gérant (en théorie)
    alpha_m = model.params["const"]

    p(f"    R2           : {model.rsquared * 100:.2f}%")
    p(f"    R2 adjusted  : {model.rsquared_adj * 100:.2f}%")
    p(f"    F-stat       : {model.fvalue:.4f}  (p = {model.f_pvalue:.2e})")
    p(f"    RMSE         : {np.sqrt(np.mean(resid**2)):.6f}")
    p(f"    Factors      : {len(selected)}")
    p(f"    Alpha (ann.) : {((1 + alpha_m) ** 12 - 1) * 100:.2f}%")
    p("")
    p(f"    Durbin-Watson   : {dw:.4f}  {'OK' if 1.5 < dw < 2.5 else 'WARNING'}")
    p(f"    Breusch-Pagan p : {bp_p:.4f}  {'OK' if bp_p > 0.05 else 'WARNING'}")
    p(f"    Jarque-Bera p   : {jb_p:.4f}  {'OK' if jb_p > 0.05 else 'WARNING'}")
    p("")

    n_ok = (vif_df["VIF"] < 5).sum()
    p(f"    VIF: {n_ok}/{len(vif_df)} factors < 5 (no multicollinearity)")
    p("")
    p("=" * 72)
    p(f"  COMPLETE | R2adj = {model.rsquared_adj * 100:.2f}% | {len(selected)} factors")
    p("=" * 72)

    return {
        "y": y_enrich,
        "y_hat": y_hat,
        "model": model,
        "selected": selected,
        "history": pd.DataFrame(history),
        "corr_raw": corr_raw,
        "params": params,
        "weights": weights,
        "resid": resid,
        "dw": dw,
        "bp_p": bp_p,
        "jb_p": jb_p,
        "vif_df": vif_df,
        "log": log,
    }


# ── Header ───────────────────────────────────────────────────────────────────
# Titre centré, sobre -- pas besoin d'en faire des tonnes
st.markdown(
    '<h1 style="text-align:center; font-weight:700; letter-spacing:-1px; margin-bottom:0;">'
    "Hedge Fund Clone Engine</h1>"
    '<p style="text-align:center; opacity:0.5; margin-top:0; font-size:1rem;">'
    f"Aberdeen Orbita Capital Return Strategy Ltd A (GBP) &mdash; {START_DATE} to {END_DATE}</p>",
    unsafe_allow_html=True,
)

# ── Run ──────────────────────────────────────────────────────────────────────
# Lancement auto -- pas de bouton, pas de sidebar, ça tourne direct
# Le cache fait que ça s'exécute qu'une fois (sauf clear cache)
with st.spinner("Running clone analysis..."):
    R = run_analysis()

# On dépack les résultats pour la lisibilité dans la suite
y, y_hat, model = R["y"], R["y_hat"], R["model"]
selected, hist_df = R["selected"], R["history"]
params, weights = R["params"], R["weights"]
resid, corr_raw = R["resid"], R["corr_raw"]

# ── Terminal output ──────────────────────────────────────────────────────────
# Le log complet de l'algo dans un faux terminal vert -- fait bien en présentation
# et montre qu'il y a de la substance derrière le dashboard
with st.expander("Algorithm Log", expanded=True):
    terminal_html = "\n".join(R["log"])
    st.markdown(f'<div class="terminal">{terminal_html}</div>', unsafe_allow_html=True)

# ── KPI row ──────────────────────────────────────────────────────────────────
# 6 métriques clés sur une ligne -- l'essentiel d'un coup d'oeil
st.markdown("---")


def kpi(col, val, lbl, color):
    """Carte KPI avec gradient sombre + couleur d'accent pour la valeur"""
    col.markdown(
        f'<div class="kpi"><div class="val" style="color:{color}">{val}</div>'
        f'<div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )


k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi(k1, f"{model.rsquared_adj * 100:.1f}%", "R\u00b2 Adjusté", "#4fc3f7")
kpi(k2, f"{model.rsquared * 100:.1f}%", "R\u00b2", "#81c784")
kpi(k3, f"{len(selected)}", "Factors", "#ffb74d")
kpi(k4, f"{sharpe(y):.2f}", "HF Sharpe", "#ce93d8")
kpi(k5, f"{sharpe(y_hat):.2f}", "Cloné Sharpe", "#ff8a65")
# Alpha annualisé : on compose l'alpha mensuel sur 12 mois
alpha_ann = ((1 + model.params["const"]) ** 12 - 1) * 100
kpi(k6, f"{alpha_ann:.1f}%", "Ann. Alpha", "#e57373")

st.markdown("")

# ── Charts ───────────────────────────────────────────────────────────────────
# Tous en dark mode avec le même thème -- cohérence visuelle

# Row 1 : les deux graphes les plus parlants
col1, col2 = st.columns(2)

with col1:
    # Performance cumulée : est-ce que le clone suit le fund ?
    # La zone ombrée entre les courbes = le tracking error visuel
    cum_hf = (1 + y).cumprod() - 1
    cum_cl = (1 + y_hat).cumprod() - 1

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cum_hf.index,
            y=cum_hf.values,
            name="Aberdeen HF",
            line=dict(color="#4fc3f7", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cum_cl.index,
            y=cum_cl.values,
            name="OLS Clone",
            line=dict(color="#ff8a65", width=2, dash="dash"),
        )
    )
    # Fill entre les deux courbes pour mieux voir l'écart
    fig.add_trace(
        go.Scatter(
            x=list(cum_hf.index) + list(cum_cl.index[::-1]),
            y=list(cum_hf.values) + list(cum_cl.values[::-1]),
            fill="toself",
            fillcolor="rgba(79,195,247,0.06)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="Performance Cumulé: Aberdeen vs Clone",
        yaxis_tickformat=".0%",
        height=400,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=50, r=20, t=45, b=35),
        **DARK,
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    # Scatter rendements mensuels : chaque point = un mois
    # Couleur = rendement du fund (rouge=mauvais, bleu=bon)
    # Clone parfait = tous les points sur la diagonale
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_hat.values,
            y=y.values,
            mode="markers",
            marker=dict(
                color=y.values,
                colorscale="RdYlBu",
                size=7,
                line=dict(width=0.5, color="white"),
            ),
            text=[d.strftime("%Y-%m") for d in y.index],
            hovertemplate="Clone: %{x:.2%}<br>Aberdeen: %{y:.2%}<br>%{text}<extra></extra>",
        )
    )
    # Droite de régression pour voir la tendance générale
    z = np.polyfit(y_hat.values, y.values, 1)
    xl = np.linspace(y_hat.min(), y_hat.max(), 100)
    fig.add_trace(
        go.Scatter(
            x=xl,
            y=np.poly1d(z)(xl),
            mode="lines",
            line=dict(color="#ff8a65", width=2, dash="dash"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"HF vs Clone monthly returns (R\u00b2 = {model.rsquared:.3f})",
        xaxis_title="Clone Return",
        yaxis_title="HF Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        height=400,
        margin=dict(l=50, r=20, t=45, b=45),
        **DARK,
    )
    st.plotly_chart(fig, width="stretch")


# Row 2 : progression du R² + composition du portefeuille
col3, col4 = st.columns(2)

with col3:
    # Double axe Y : R²adj (ligne bleue) + p-values (barres oranges)
    # On voit bien le gain marginal décroissant à chaque nouveau facteur
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=hist_df["step"],
            y=hist_df["r2_adj"],
            mode="lines+markers",
            name="R\u00b2 adj",
            line=dict(color="#4fc3f7", width=3),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=hist_df["step"],
            y=hist_df["p_value"],
            name="p-value",
            opacity=0.3,
            marker_color="#ff8a65",
        ),
        secondary_y=True,
    )
    # Ligne verte = l'objectif R² (on l'atteint ou pas, la vie est ainsi faite)
    fig.add_hline(
        y=TARGET_R2,
        line_dash="dot",
        line_color="#81c784",
        annotation_text=f"Target {TARGET_R2 * 100:.0f}%",
        secondary_y=False,
    )
    fig.update_layout(
        title="Stepwise Selection: R\u00b2 Progression",
        xaxis_title="Step",
        height=400,
        margin=dict(l=50, r=50, t=45, b=35),
        **DARK,
    )
    fig.update_yaxes(title_text="R\u00b2 adj", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="p-value", secondary_y=True)
    st.plotly_chart(fig, width="stretch")

with col4:
    # Barres horizontales : poids de chaque facteur dans le portefeuille clone
    # Bleu = position longue, rouge = short
    ws = weights.sort_values()
    colors_w = ["#4fc3f7" if v > 0 else "#e57373" for v in ws.values]
    fig = go.Figure(
        go.Bar(
            y=ws.index,
            x=ws.values,
            orientation="h",
            marker_color=colors_w,
            opacity=0.9,
            text=[f"{v:+.1f}%" for v in ws.values],
            textposition="outside",
            textfont=dict(size=10),
        )
    )
    fig.add_vline(x=0, line_color="white", line_width=0.8)
    fig.update_layout(
        title="Distribution des poids de chaque actifs (portefeuille cloné)",
        xaxis_title="Weight (%)",
        height=400,
        margin=dict(l=110, r=70, t=45, b=35),
        **DARK,
    )
    st.plotly_chart(fig, width="stretch")


# Row 3 : QQ plot (le graphe des résidus est commenté, on le garde au cas où)
col5, col6 = st.columns(2)

with col6:
    # QQ plot : si les points collent à la diagonale, résidus normaux
    # Des queues épaisses = le modèle galère sur les gros mouvements
    # (normal pour un HF market-neutral, c'est justement son alpha)
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            marker=dict(color="#4fc3f7", size=6, line=dict(width=0.5, color="white")),
            name="Residuals",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[osm.min(), osm.max()],
            y=[slope * osm.min() + intercept, slope * osm.max() + intercept],
            mode="lines",
            line=dict(color="#ff8a65", dash="dash"),
            name="Normal",
        )
    )
    fig.update_layout(
        title=f"Q-Q Plot  |  Jarque-Bera p={R['jb_p']:.3f}",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=350,
        margin=dict(l=50, r=20, t=45, b=45),
        **DARK,
    )
    st.plotly_chart(fig, width="stretch")

# ── Performance table ────────────────────────────────────────────────────────
# Le récap final : les chiffres qui comptent pour comparer fund vs clone
st.markdown("---")
tc1, tc2 = st.columns(2)

with tc1:
    st.markdown("### HF vs Clone")
    perf = pd.DataFrame(
        {
            "": ["Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Max Drawdown"],
            "Aberdeen HF": [
                f"{ann_ret(y) * 100:.2f}%",
                f"{ann_vol(y) * 100:.2f}%",
                f"{sharpe(y):.2f}",
                f"{max_dd(y) * 100:.2f}%",
            ],
            "OLS Clone": [
                f"{ann_ret(y_hat) * 100:.2f}%",
                f"{ann_vol(y_hat) * 100:.2f}%",
                f"{sharpe(y_hat):.2f}",
                f"{max_dd(y_hat) * 100:.2f}%",
            ],
        }
    ).set_index("")
    st.dataframe(perf, width="stretch")

with tc2:
    # Tests statistiques : PASS/FAIL pour valider la robustesse du modèle
    st.markdown("### OLS Diagnostics")
    dw_ok = 1.5 < R["dw"] < 2.5
    bp_ok = R["bp_p"] > 0.05
    jb_ok = R["jb_p"] > 0.05
    diag = pd.DataFrame(
        {
            "Test": ["Durbin-Watson", "Breusch-Pagan", "Jarque-Bera"],
            "Value": [f"{R['dw']:.4f}", f"{R['bp_p']:.4f}", f"{R['jb_p']:.4f}"],
            "Threshold": ["1.5 - 2.5", "> 0.05", "> 0.05"],
            "Result": [
                "PASS" if dw_ok else "FAIL",
                "PASS" if bp_ok else "FAIL",
                "PASS" if jb_ok else "FAIL",
            ],
        }
    ).set_index("Test")
    st.dataframe(diag, width="stretch")
