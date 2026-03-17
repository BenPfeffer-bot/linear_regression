"""
════════════════════════════════════════════════════════════════════════════════
 HEDGE FUND CLONE – Moteur de recherche automatique de facteurs (yfinance)
 Aberdeen Orbita Capital Return Strategy Ltd A (GBP)
════════════════════════════════════════════════════════════════════════════════
 Fonctionnement :
   1. Télécharge ~150 ETF/indices sur la période du HF via yfinance
   2. Calcule les rendements mensuels
   3. Lance un forward stepwise automatique avec enrichissement de features
      (lags, termes quadratiques, interactions croisées)
   4. S'arrête dès que R² ajusté ≥ CIBLE (défaut 60%) ou explore tout l'univers
   5. Affiche ANOVA + table coefficients + diagnostics + graphiques

 Usage :
   pip install yfinance pandas numpy statsmodels scikit-learn matplotlib scipy
   python hf_clone_search.py

 Pour changer la cible R² : modifier TARGET_R2 en bas du fichier
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import warnings, time, itertools

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import yfinance as yf


# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Période du HF Aberdeen (ajuster selon ton track record)
START_DATE = "2006-07-01"
END_DATE = "2016-05-01"

# Cible de R² ajusté
TARGET_R2 = 0.60  # ← modifier ici (ex: 0.40 pour un objectif plus souple)

# Nombre maximum de facteurs dans le modèle final
MAX_FACTORS = 12

# Seuil p-value pour le stepwise
P_THRESHOLD = 0.15

# Fichier de sortie
OUTPUT_PNG = "aberdeen_clone_search.png"
OUTPUT_CSV = "aberdeen_factors_found.csv"


# ══════════════════════════════════════════════════════════════════════════════
# 1. RENDEMENTS MENSUELS D'ABERDEEN
# ══════════════════════════════════════════════════════════════════════════════

ABERDEEN_MONTHLY = [
    0.0129,
    0.0018,
    0.0245,
    0.0140,
    0.0061,
    0.0237,
    -0.0066,
    -0.0106,
    0.0161,
    0.0044,
    0.0249,
    0.0270,
    0.0119,
    -0.0263,
    0.0265,
    -0.0083,
    -0.0036,
    -0.0161,
    0.0228,
    0.0367,
    -0.0046,
    -0.0313,
    0.0401,
    0.0014,
    0.0153,
    0.0058,
    0.0158,
    -0.0533,
    0.0114,
    0.0135,
    0.0036,
    0.0289,
    0.0307,
    -0.0258,
    0.0056,
    0.0275,
    -0.0033,
    0.0137,
    0.0498,
    0.0222,
    -0.0190,
    0.0229,
    0.0064,
    -0.0114,
    0.0288,
    -0.0100,
    0.0174,
    -0.0135,
    0.0060,
    -0.0008,
    -0.0016,
    -0.0081,
    0.0078,
    -0.0106,
    0.0200,
    0.0160,
    0.0008,
    0.0068,
    0.0176,
    0.0094,
    0.0162,
    -0.0173,
    0.0004,
    0.0011,
    0.0602,
    0.0145,
    -0.0039,
    0.0188,
    0.0079,
    0.0150,
    -0.0057,
    0.0290,
    -0.0208,
    0.0058,
    -0.0250,
    0.0051,
    0.0092,
    -0.0055,
    0.0147,
    0.0329,
    0.0041,
    0.0214,
    -0.0049,
    0.0457,
    -0.0007,
    0.0133,
    -0.0087,
    0.0116,
    0.0045,
    0.0319,
    -0.0023,
    -0.0345,
    -0.0096,
    -0.0207,
    0.0097,
    0.0047,
    -0.0057,
    -0.0322,
    0.0104,
    0.0156,
    -0.0185,
    -0.0154,
    -0.0233,
    0.0166,
    0.0035,
    -0.0132,
    -0.0001,
    0.0240,
    0.0079,
    0.0384,
    0.0038,
    0.0082,
    0.0012,
    0.0142,
    -0.0065,
    -0.0244,
    0.0434,
    -0.0195,
    -0.0103,
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. UNIVERS D'ACTIFS (ETF + indices liquides, pas de HF indices)
#    Organisé par classe d'actifs pour la lisibilité
# ══════════════════════════════════════════════════════════════════════════════

UNIVERSE: dict[str, list[str]] = {
    # ── Actions mondiales ──────────────────────────────────────────────────
    "Equity_US": [
        "SPY",  # S&P 500
        "QQQ",  # Nasdaq 100
        "IWM",  # Russell 2000 (small cap)
        "IWD",  # Russell 1000 Value
        "IWF",  # Russell 1000 Growth
        "DVY",  # Dividend ETF
        "XLF",  # Financials
        "XLE",  # Energy
        "XLV",  # Healthcare
        "XLK",  # Technology
        "XLU",  # Utilities
        "XLP",  # Consumer Staples
        "XLY",  # Consumer Discretionary
    ],
    "Equity_Intl": [
        "EFA",  # MSCI EAFE (developed ex-US)
        "EEM",  # MSCI Emerging Markets
        "EWJ",  # Japan (MSCI Japan)
        "EWG",  # Germany
        "EWU",  # United Kingdom
        "EWQ",  # France
        "EWZ",  # Brazil
        "FXI",  # China large cap
        "VWO",  # Vanguard EM
        "ACWI",  # MSCI All Country World
        "VEA",  # Vanguard EAFE
        "IEMG",  # iShares Core EM
    ],
    # ── Obligations ───────────────────────────────────────────────────────
    "Fixed_Income": [
        "AGG",  # US Aggregate Bonds
        "TLT",  # 20+ Year Treasury
        "IEF",  # 7-10 Year Treasury
        "SHY",  # 1-3 Year Treasury
        "LQD",  # IG Corporate Bonds
        "HYG",  # High Yield Bonds
        "JNK",  # SPDR High Yield
        "MBB",  # Mortgage-backed
        "TIP",  # TIPS (inflation-linked)
        "EMB",  # EM Bonds USD
        "BNDX",  # Intl bonds
        "BWX",  # Intl Treasury
    ],
    # ── Matières premières ────────────────────────────────────────────────
    "Commodities": [
        "GLD",  # Gold
        "SLV",  # Silver
        "USO",  # Oil (WTI)
        "UNG",  # Natural Gas
        "DBC",  # Bloomberg Commodity
        "PDBC",  # Diversified commodities
        "GSG",  # S&P GSCI
        "CORN",  # Corn
        "WEAT",  # Wheat
        "SOYB",  # Soybeans
        "CPER",  # Copper
    ],
    # ── Devises ───────────────────────────────────────────────────────────
    "FX": [
        "UUP",  # US Dollar Index
        "FXE",  # EUR/USD
        "FXB",  # GBP/USD
        "FXY",  # JPY/USD
        "FXC",  # CAD/USD
        "FXA",  # AUD/USD
        "FXF",  # CHF/USD
    ],
    # ── Volatilité & alternatives ─────────────────────────────────────────
    "Vol_Alt": [
        "^VIX",  # CBOE VIX
        "VXX",  # VIX Short-term futures
        "SVXY",  # Short VIX
        "VIXY",  # Long VIX
        "SPLV",  # Low volatility S&P
        "USMV",  # Min volatility
    ],
    # ── REITs & Infrastructure ────────────────────────────────────────────
    "Real_Assets": [
        "VNQ",  # Vanguard REITs
        "IYR",  # iShares REITs
        "REM",  # Mortgage REITs
        "IAU",  # Gold (iShares)
        "WOOD",  # Timber
        "MOO",  # Agribusiness
    ],
    # ── Facteurs / Smart Beta ─────────────────────────────────────────────
    "Factor_ETF": [
        "MTUM",  # Momentum
        "VLUE",  # Value
        "SIZE",  # Size
        "QUAL",  # Quality
        "USMV",  # Min Volatility
        "DGRO",  # Dividend Growth
        "PFF",  # Preferred stock
        "PKW",  # Buyback ETF
    ],
    # ── Stratégies alternatives liquides ──────────────────────────────────
    "Liquid_Alt": [
        "BTAL",  # Beta-hedged
        "QAI",  # IQ Hedge Multi-Strategy
        "MNA",  # Merger Arb
        "CSMA",  # Credit Suisse Merger Arb
        "RLY",  # Multi-asset diversified
        "WTMF",  # Trend following
    ],
    # ── Crypto / Nouvelles classes (si dispo sur la période) ─────────────
    "Thematic": [
        "IBB",  # Biotech
        "XBI",  # SPDR Biotech
        "HACK",  # Cybersecurity
        "GDX",  # Gold miners
        "GDXJ",  # Junior gold miners
        "KRE",  # Regional Banks
        "XHB",  # Homebuilders
        "ITB",  # Home Construction
    ],
}

# Aplatir en liste unique
ALL_TICKERS = [t for group in UNIVERSE.values() for t in group]


# ══════════════════════════════════════════════════════════════════════════════
# 3. FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════


def sep(n: int = 78) -> None:
    print("═" * n)


def sharpe(r: pd.Series) -> float:
    return r.mean() / r.std() * np.sqrt(12)


def ann_ret(r: pd.Series) -> float:
    return (1 + r.mean()) ** 12 - 1


def ann_vol(r: pd.Series) -> float:
    return r.std() * np.sqrt(12)


def max_dd(r: pd.Series) -> float:
    cum = (1 + r).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()


def download_monthly_returns(
    tickers: list[str], start: str, end: str, batch_size: int = 30, pause: float = 1.0
) -> pd.DataFrame:
    """
    Télécharge les cours de clôture ajustés via yfinance par batches,
    puis convertit en rendements mensuels (fin de mois).
    """
    all_prices = []
    failed = []

    print(f"\n{'─' * 78}")
    print(f" Téléchargement de {len(tickers)} tickers  |  {start} → {end}")
    print(f"{'─' * 78}")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1} : {batch}")
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            # Extraire Close
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]] if "Close" in raw.columns else raw

            all_prices.append(close)
        except Exception as e:
            print(f"    ⚠ Batch échoué : {e}")
            for t in batch:
                failed.append(t)
        time.sleep(pause)

    if not all_prices:
        raise RuntimeError("Aucune donnée téléchargée. Vérifier connexion internet.")

    prices_daily = pd.concat(all_prices, axis=1)
    prices_daily = prices_daily.loc[:, ~prices_daily.columns.duplicated()]

    # Rééchantillonnage mensuel (dernier jour ouvré du mois)
    prices_monthly = prices_daily.resample("ME").last()

    # Rendements mensuels
    returns_monthly = prices_monthly.pct_change().dropna(how="all")

    # Supprimer colonnes avec trop de NaN (< 80% de données)
    threshold = int(0.80 * len(returns_monthly))
    returns_monthly = returns_monthly.dropna(axis=1, thresh=threshold)

    n_ok = returns_monthly.shape[1]
    print(
        f"\n  ✓ {n_ok} actifs chargés avec succès  |  "
        f"{len(failed)} échoués  |  "
        f"{returns_monthly.shape[0]} mois"
    )

    return returns_monthly


def build_enriched_features(
    X_base: pd.DataFrame, top_n_interactions: int = 5
) -> pd.DataFrame:
    """
    Enrichit X_base avec :
      - Lags t-1
      - Termes quadratiques (expositions non-linéaires)
      - Interactions croisées entre les top_n facteurs les plus corrélés
    """
    X = X_base.copy()

    # Lags t-1
    for col in X_base.columns:
        X[f"{col}_lag1"] = X_base[col].shift(1)

    # Termes quadratiques
    for col in X_base.columns:
        X[f"{col}_sq"] = X_base[col] ** 2

    # Interactions croisées (top_n facteurs par corrélation avec la target)
    # → calculées APRÈS alignement avec y dans la fonction principale

    return X.dropna()


def forward_stepwise_search(
    y: pd.Series,
    X: pd.DataFrame,
    max_vars: int = MAX_FACTORS,
    p_thr: float = P_THRESHOLD,
    target_r2: float = TARGET_R2,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Forward stepwise automatique :
      - Ajoute le facteur qui maximise R² ajusté sous contrainte p < p_thr
      - S'arrête si R² ajusté >= target_r2 ou si plus de facteurs disponibles
    Retourne : (liste des facteurs sélectionnés, historique DataFrame)
    """
    remaining = list(X.columns)
    selected = []
    best_r2 = 0.0
    history = []

    if verbose:
        print(f"\n{'─' * 78}")
        print(
            f" Forward Stepwise  |  cible R²adj ≥ {target_r2 * 100:.0f}%  |  p < {p_thr}  |  max {max_vars} vars"
        )
        print(f"{'─' * 78}")
        print(
            f"  {'Étape':<6} {'Facteur ajouté':<30} {'R²adj':>8}  {'p-value':>10}  {'Statut'}"
        )
        print(f"  {'─' * 6} {'─' * 30} {'─' * 8}  {'─' * 10}  {'─' * 20}")

    for step in range(max_vars):
        best_new = None
        best_new_r2 = best_r2
        best_p = 1.0

        for cand in remaining:
            try:
                res = sm.OLS(y, sm.add_constant(X[selected + [cand]])).fit()
                pval = float(res.pvalues.get(cand, 1.0))
                if res.rsquared_adj > best_new_r2 and pval < p_thr:
                    best_new_r2 = res.rsquared_adj
                    best_new = cand
                    best_p = pval
            except Exception:
                continue

        if best_new is None:
            if verbose:
                print(f"  → Arrêt : aucun facteur supplémentaire significatif.")
            break

        selected.append(best_new)
        remaining.remove(best_new)
        best_r2 = best_new_r2

        status = (
            f"✓ CIBLE ATTEINTE ({best_r2 * 100:.1f}%)" if best_r2 >= target_r2 else ""
        )
        if verbose:
            print(
                f"  {step + 1:<6} {best_new:<30} {best_r2:>8.4f}  {best_p:>10.6f}  {status}"
            )

        history.append(
            {
                "step": step + 1,
                "factor": best_new,
                "r2_adj": best_new_r2,
                "p_value": best_p,
                "target_hit": best_r2 >= target_r2,
            }
        )

        if best_r2 >= target_r2:
            if verbose:
                print(
                    f"\n  🎯 R²adj = {best_r2 * 100:.2f}% ≥ cible {target_r2 * 100:.0f}%  →  recherche terminée."
                )
            break

    return selected, pd.DataFrame(history)


def print_anova(
    ols_model: sm.regression.linear_model.RegressionResultsWrapper,
    y: pd.Series,
    factor_names: list[str],
) -> dict:
    """Affiche la table ANOVA et retourne les valeurs clés."""
    n = int(ols_model.nobs)
    k = len(factor_names)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum(ols_model.resid**2))
    ss_reg = ss_tot - ss_res
    df_reg = k
    df_res = n - k - 1
    ms_reg = ss_reg / df_reg
    ms_res = ss_res / df_res
    f_stat = ms_reg / ms_res
    f_pval = 1 - stats.f.cdf(f_stat, df_reg, df_res)

    sep()
    print("TABLE ANOVA")
    sep()
    print(
        f"  {'Source':<22} {'SS':>14} {'df':>6} {'MS':>14} {'F-stat':>10} {'p-value':>12}"
    )
    print(f"  {'─' * 22} {'─' * 14} {'─' * 6} {'─' * 14} {'─' * 10} {'─' * 12}")
    print(
        f"  {'Régression':<22} {ss_reg:>14.8f} {df_reg:>6} {ms_reg:>14.8f} {f_stat:>10.4f} {f_pval:>12.8f}"
    )
    print(f"  {'Résidus':<22} {ss_res:>14.8f} {df_res:>6} {ms_res:>14.8f}")
    print(f"  {'Total':<22} {ss_tot:>14.8f} {n - 1:>6}")
    print()
    print(
        f"  R²          = {ols_model.rsquared:.6f}   ({ols_model.rsquared * 100:.2f}%)"
    )
    print(
        f"  R² ajusté   = {ols_model.rsquared_adj:.6f}   ({ols_model.rsquared_adj * 100:.2f}%)"
    )
    print(f"  F-statistic = {f_stat:.4f}   (p = {f_pval:.8f})")
    print(f"  RMSE        = {np.sqrt(ms_res):.8f}")
    print()

    return dict(
        ss_reg=ss_reg,
        ss_res=ss_res,
        ss_tot=ss_tot,
        df_reg=df_reg,
        df_res=df_res,
        f_stat=f_stat,
        f_pval=f_pval,
    )


def print_coef_table(ols_model) -> None:
    """Table complète : Intercept, β, Std Error, t-stat, p-value, IC 95%."""
    sep()
    print("TABLE DES COEFFICIENTS")
    sep()
    hdr = (
        f"  {'Variable':<28} {'Coeff':>12} {'Std Err':>10} "
        f"{'t-stat':>9} {'p-value':>12} {'IC95% inf':>12} {'IC95% sup':>12}  Sig."
    )
    print(hdr)
    print("  " + "─" * 102)

    for var in ols_model.params.index:
        coef = ols_model.params[var]
        se = ols_model.bse[var]
        tval = ols_model.tvalues[var]
        pval = ols_model.pvalues[var]
        ci_l = ols_model.conf_int().loc[var, 0]
        ci_h = ols_model.conf_int().loc[var, 1]

        if pval < 0.01:
            sig = "***"
        elif pval < 0.05:
            sig = "**"
        elif pval < 0.10:
            sig = "*"
        elif pval < 0.15:
            sig = "."
        else:
            sig = ""

        label = "Intercept (α)" if var == "const" else var
        print(
            f"  {label:<28} {coef:>12.6f} {se:>10.6f} "
            f"{tval:>9.4f} {pval:>12.8f} {ci_l:>12.6f} {ci_h:>12.6f}  {sig}"
        )

    print()
    print("  Légende : *** p<0.01 | ** p<0.05 | * p<0.10 | . p<0.15")


def print_diagnostics(ols_model, X_with_const: pd.DataFrame) -> dict:
    """Durbin-Watson, Breusch-Pagan, Jarque-Bera, VIF."""
    dw = durbin_watson(ols_model.resid)
    _, bp_p, _, _ = het_breuschpagan(ols_model.resid, X_with_const)
    jb_stat, jb_p = stats.jarque_bera(ols_model.resid)

    vif_vals = [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(X_with_const.shape[1])
    ]
    vif_df = pd.DataFrame({"Variable": X_with_const.columns, "VIF": vif_vals})
    vif_df = vif_df[vif_df["Variable"] != "const"]

    sep()
    print("DIAGNOSTICS")
    sep()
    dw_ok = "✓" if 1.5 < dw < 2.5 else "⚠"
    bp_ok = "✓" if bp_p > 0.05 else "⚠"
    jb_ok = "✓" if jb_p > 0.05 else "⚠"
    print(f"  {dw_ok} Durbin-Watson       : {dw:.4f}  (1.5–2.5 = OK)")
    print(f"  {bp_ok} Breusch-Pagan p     : {bp_p:.4f}  (>0.05 = homoscédasticité)")
    print(f"  {jb_ok} Jarque-Bera p       : {jb_p:.4f}  (>0.05 = normalité résidus)")
    print(f"     Skewness résidus    : {ols_model.resid.skew():.4f}")
    print(f"     Kurtosis résidus    : {ols_model.resid.kurt():.4f}  (excès)")
    print()
    print("  VIF (< 5 = pas de multicolinéarité) :")
    for _, row in vif_df.iterrows():
        flag = "✓" if row["VIF"] < 5 else "⚠"
        print(f"    {flag} {row['Variable']:<30} {row['VIF']:.2f}")
    print()

    return dict(dw=dw, bp_p=bp_p, jb_p=jb_p, vif=vif_df)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MOTEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════


def run_clone_search(target_r2: float = TARGET_R2) -> None:

    sep(78)
    print(" HEDGE FUND CLONE – Recherche automatique de facteurs (yfinance)")
    print(" Aberdeen Orbita Capital Return Strategy Ltd A (GBP)")
    sep(78)

    # ── 4.1 Téléchargement ────────────────────────────────────────────────────
    returns_monthly = download_monthly_returns(
        ALL_TICKERS, start=START_DATE, end=END_DATE
    )

    # ── 4.2 Construction de la série Aberdeen ────────────────────────────────
    # On crée un DatetimeIndex mensuel aligné sur les données téléchargées
    target_dates = returns_monthly.index

    # Aberdeen a 119 obs → on aligne sur la longueur disponible
    aberdeen_full = pd.Series(ABERDEEN_MONTHLY, name="Aberdeen")
    n_align = min(len(aberdeen_full), len(target_dates))
    # Aligner depuis la fin (les données récentes sont mieux synchronisées)
    y = pd.Series(
        ABERDEEN_MONTHLY[-n_align:], index=target_dates[-n_align:], name="Aberdeen"
    )

    # Aligner les facteurs sur la même période
    X_raw = returns_monthly.loc[y.index].copy()

    # Supprimer colonnes encore vides après alignement
    X_raw = X_raw.dropna(axis=1, thresh=int(0.90 * len(y)))
    X_raw = X_raw.fillna(X_raw.median())

    print(
        f"\n  Série Aberdeen  : {len(y)} obs  ({y.index[0].date()} → {y.index[-1].date()})"
    )
    print(f"  Facteurs dispo  : {X_raw.shape[1]} ETF/indices")

    # ── 4.3 Statistiques descriptives HF ─────────────────────────────────────
    sep()
    print("STATISTIQUES DESCRIPTIVES – Aberdeen Orbita Capital")
    sep()
    print(f"  Rendement annualisé   : {ann_ret(y) * 100:.2f}%")
    print(f"  Volatilité annualisée : {ann_vol(y) * 100:.2f}%")
    print(f"  Ratio de Sharpe       : {sharpe(y):.2f}")
    print(f"  Max Drawdown          : {max_dd(y) * 100:.2f}%")
    print(f"  Skewness              : {y.skew():.4f}")
    print(f"  Kurtosis (excès)      : {y.kurt():.4f}")
    print()

    # ── 4.4 Corrélations brutes ───────────────────────────────────────────────
    corr_raw = X_raw.corrwith(y).abs().sort_values(ascending=False)
    sep()
    print("TOP 20 CORRÉLATIONS ABSOLUES (features brutes)")
    sep()
    print(corr_raw.head(20).to_string())
    print()

    # ── 4.5 Phase 1 : Stepwise sur features brutes ───────────────────────────
    sep()
    print("PHASE 1 – Stepwise sur features brutes")
    sel1, hist1 = forward_stepwise_search(
        y,
        X_raw,
        max_vars=MAX_FACTORS,
        p_thr=P_THRESHOLD,
        target_r2=target_r2,
        verbose=True,
    )

    ols1 = sm.OLS(y, sm.add_constant(X_raw[sel1])).fit()
    r2_phase1 = ols1.rsquared_adj
    print(
        f"\n  Phase 1 → R²adj = {r2_phase1 * 100:.2f}%  |  {len(sel1)} facteurs : {sel1}"
    )

    if r2_phase1 >= target_r2:
        print(f"\n  ✅ Cible atteinte dès la Phase 1 !")
        best_factors = sel1
        best_model = ols1
        best_X_const = sm.add_constant(X_raw[sel1])
    else:
        # ── 4.6 Phase 2 : Enrichissement + nouveau stepwise ──────────────────
        sep()
        print(f"PHASE 2 – Features enrichies (lags, sq, interactions)")
        print(
            f"          R² Phase 1 = {r2_phase1 * 100:.2f}%  <  cible {target_r2 * 100:.0f}%"
        )
        sep()

        # Enrichir l'univers
        X_enrich = X_raw.copy()

        # Lags t-1
        for col in X_raw.columns:
            X_enrich[f"{col}_lag1"] = X_raw[col].shift(1)

        # Termes quadratiques des top-20 corrélés
        for col in corr_raw.head(20).index:
            X_enrich[f"{col}_sq"] = X_raw[col] ** 2

        # Interactions croisées entre les top-8 corrélés
        top8 = corr_raw.head(8).index.tolist()
        for a, b in itertools.combinations(top8, 2):
            X_enrich[f"{a}x{b}"] = X_raw[a] * X_raw[b]

        X_enrich = X_enrich.dropna()
        y_enrich = y.loc[X_enrich.index]

        print(f"  Features enrichies : {X_enrich.shape[1]} variables")

        sel2, hist2 = forward_stepwise_search(
            y_enrich,
            X_enrich,
            max_vars=MAX_FACTORS,
            p_thr=P_THRESHOLD,
            target_r2=target_r2,
            verbose=True,
        )

        X_sel2 = sm.add_constant(X_enrich[sel2])
        ols2 = sm.OLS(y_enrich, X_sel2).fit()
        r2_ph2 = ols2.rsquared_adj

        print(
            f"\n  Phase 2 → R²adj = {r2_ph2 * 100:.2f}%  |  {len(sel2)} facteurs : {sel2}"
        )

        if r2_ph2 >= target_r2:
            print(f"  ✅ Cible atteinte en Phase 2 !")
        else:
            print(
                f"  ⚠  Cible {target_r2 * 100:.0f}% non atteinte — meilleur modèle conservé."
            )
            print(
                f"     Note : ce HF est un FoF Arbitrage market-neutral dont la performance"
            )
            print(
                f"     est majoritairement de l'alpha pur, non réplicable par des ETF liquides."
            )
            print(
                f"     Référence : Hasanhodzic & Lo (2007), R² typique = 10-20% pour cette stratégie."
            )

        best_factors = sel2
        best_model = ols2
        best_X_const = X_sel2
        y = y_enrich

    # ── 4.7 Résultats finaux ──────────────────────────────────────────────────
    sep()
    print("RÉSULTATS FINAUX – MODÈLE OPTIMAL")
    sep()

    anova_res = print_anova(best_model, best_factors)
    print_coef_table(best_model)
    diag_res = print_diagnostics(best_model, best_X_const)

    # Portefeuille de réplication
    y_hat = best_model.fittedvalues.rename("Clone")
    params = best_model.params.drop("const")
    weights = params / params.abs().sum() * 100

    sep()
    print("PORTEFEUILLE DE RÉPLICATION")
    sep()
    print(f"  {'Facteur':<30} {'β':>12} {'Poids %':>10}  Direction")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 10}  {'─' * 8}")
    for f, w in weights.sort_values().items():
        print(
            f"  {f:<30} {params[f]:>12.6f} {w:>+9.2f}%  {'LONG' if w > 0 else 'SHORT'}"
        )

    alpha_m = best_model.params["const"]
    print(
        f"\n  Alpha mensuel  : {alpha_m * 100:.4f}%  "
        f"(annualisé ≈ {((1 + alpha_m) ** 12 - 1) * 100:.2f}%)"
    )
    print(f"  R² ajusté      : {best_model.rsquared_adj * 100:.2f}%")

    # Comparaison performances
    sep()
    print("COMPARAISON HF vs CLONE")
    sep()
    print(f"  {'Métrique':<26} {'HF Aberdeen':>14} {'Clone OLS':>14}")
    print(f"  {'─' * 56}")
    for label, hf_v, cl_v in [
        ("Rdt annualisé", ann_ret(y), ann_ret(y_hat)),
        ("Vol annualisée", ann_vol(y), ann_vol(y_hat)),
        ("Max Drawdown", max_dd(y), max_dd(y_hat)),
    ]:
        print(f"  {label:<26} {hf_v * 100:>13.2f}% {cl_v * 100:>13.2f}%")
    print(f"  {'Ratio de Sharpe':<26} {sharpe(y):>14.2f} {sharpe(y_hat):>14.2f}")

    # Sauvegarde CSV des facteurs
    results_df = pd.DataFrame(
        {
            "factor": list(params.index),
            "beta": list(params.values),
            "weight_pct": list(weights.values),
            "std_err": [best_model.bse[f] for f in params.index],
            "t_stat": [best_model.tvalues[f] for f in params.index],
            "p_value": [best_model.pvalues[f] for f in params.index],
        }
    )
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  ✓ Résultats sauvegardés → {OUTPUT_CSV}")

    # ── 4.8 Graphiques ────────────────────────────────────────────────────────
    _plot_results(y, y_hat, best_model, weights, anova_res, diag_res)


# ══════════════════════════════════════════════════════════════════════════════
# 5. GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════


def _plot_results(y, y_hat, ols_model, weights, anova_res, diag_res) -> None:

    C = {
        "hf": "#1a3a6b",
        "clone": "#e05c2a",
        "pos": "#27ae60",
        "neg": "#e74c3c",
        "grey": "#9aa3b0",
        "bg": "#f8f9fc",
        "grid": "#e2e5ea",
        "dark": "#1a2332",
    }
    fmt_pct = FuncFormatter(lambda x, _: f"{x:.1%}")

    fig = plt.figure(figsize=(20, 24), facecolor=C["bg"])
    fig.suptitle(
        "Clone de Hedge Fund – Aberdeen Orbita Capital Return Strategy Ltd A (GBP)\n"
        f"Recherche automatique yfinance  |  R²adj = {ols_model.rsquared_adj * 100:.2f}%  |  "
        f"F-stat = {anova_res['f_stat']:.2f}  (p = {anova_res['f_pval']:.4f})",
        fontsize=13,
        fontweight="bold",
        y=0.987,
        color=C["dark"],
    )
    gs = gridspec.GridSpec(
        4, 2, hspace=0.50, wspace=0.30, top=0.955, bottom=0.04, left=0.07, right=0.96
    )

    # A – Performance cumulée
    ax1 = fig.add_subplot(gs[0, :])
    cum_hf = (1 + y).cumprod() - 1
    cum_cl = (1 + y_hat).cumprod() - 1
    ax1.fill_between(cum_hf.index, cum_hf, cum_cl, alpha=0.12, color=C["clone"])
    ax1.plot(cum_hf.index, cum_hf, color=C["hf"], lw=2.4, label="Aberdeen HF")
    ax1.plot(cum_cl.index, cum_cl, color=C["clone"], lw=2.0, ls="--", label="Clone OLS")
    ax1.axhline(0, color=C["grey"], lw=0.8, ls=":")
    ax1.set_title("Performance cumulée – HF vs Clone", fontweight="bold", pad=8)
    ax1.yaxis.set_major_formatter(fmt_pct)
    ax1.set_facecolor(C["bg"])
    ax1.grid(True, color=C["grid"], lw=0.6)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.set_ylabel("Rendement cumulé")

    # B – Scatter
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(y_hat, y, alpha=0.55, color=C["hf"], s=32, edgecolors="white", lw=0.4)
    z = np.polyfit(y_hat, y, 1)
    xl = np.linspace(y_hat.min(), y_hat.max(), 100)
    ax2.plot(xl, np.poly1d(z)(xl), color=C["clone"], lw=2.0)
    ax2.axhline(0, color=C["grey"], lw=0.6, ls=":")
    ax2.axvline(0, color=C["grey"], lw=0.6, ls=":")
    ax2.set_xlabel("Rendement Clone")
    ax2.set_ylabel("Rendement HF")
    ax2.set_title(
        f"HF vs Clone\nR² = {ols_model.rsquared:.3f}  |  R²adj = {ols_model.rsquared_adj:.3f}",
        fontweight="bold",
        pad=6,
    )
    ax2.xaxis.set_major_formatter(fmt_pct)
    ax2.yaxis.set_major_formatter(fmt_pct)
    ax2.set_facecolor(C["bg"])
    ax2.grid(True, color=C["grid"], lw=0.6)

    # C – Table ANOVA
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ss_r, ss_e, ss_t = anova_res["ss_reg"], anova_res["ss_res"], anova_res["ss_tot"]
    df_r, df_e = anova_res["df_reg"], anova_res["df_res"]
    anova_td = [
        ["Source", "SS", "df", "MS", "F", "p-value"],
        [
            "Régression",
            f"{ss_r:.6f}",
            f"{df_r}",
            f"{ss_r / df_r:.6f}",
            f"{anova_res['f_stat']:.4f}",
            f"{anova_res['f_pval']:.6f}",
        ],
        ["Résidus", f"{ss_e:.6f}", f"{df_e}", f"{ss_e / df_e:.6f}", "—", "—"],
        ["Total", f"{ss_t:.6f}", f"{df_r + df_e}", "—", "—", "—"],
        [
            "R²",
            f"{ols_model.rsquared:.4f}",
            f"({ols_model.rsquared * 100:.1f}%)",
            "",
            "",
            "",
        ],
        [
            "R²adj",
            f"{ols_model.rsquared_adj:.4f}",
            f"({ols_model.rsquared_adj * 100:.1f}%)",
            "",
            "",
            "",
        ],
    ]
    tbl3 = ax3.table(
        cellText=anova_td[1:],
        colLabels=anova_td[0],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl3.auto_set_font_size(False)
    tbl3.set_fontsize(8.0)
    for (r, c), cell in tbl3.get_celld().items():
        cell.set_edgecolor(C["grid"])
        if r == 0:
            cell.set_facecolor(C["hf"])
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#dce8f5")
        elif r in [4, 5]:
            cell.set_facecolor("#f5f0dc")
        elif r % 2 == 0:
            cell.set_facecolor("#eef1f6")
        else:
            cell.set_facecolor("white")
    ax3.set_title("Table ANOVA", fontweight="bold", pad=8)

    # D – Table coefficients
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    coef_rows = []
    for var in ols_model.params.index:
        pval = ols_model.pvalues[var]
        sig = (
            "***"
            if pval < 0.01
            else (
                "**"
                if pval < 0.05
                else ("*" if pval < 0.10 else ("." if pval < 0.15 else ""))
            )
        )
        coef_rows.append(
            [
                "Intercept (α)" if var == "const" else var,
                f"{ols_model.params[var]:.6f}",
                f"{ols_model.bse[var]:.6f}",
                f"{ols_model.tvalues[var]:.4f}",
                f"{pval:.6f}",
                f"{ols_model.conf_int().loc[var, 0]:.6f}",
                f"{ols_model.conf_int().loc[var, 1]:.6f}",
                sig,
            ]
        )
    headers4 = [
        "Variable",
        "β",
        "Std Error",
        "t-stat",
        "p-value",
        "IC95% inf",
        "IC95% sup",
        "Sig.",
    ]
    tbl4 = ax4.table(
        cellText=coef_rows,
        colLabels=headers4,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl4.auto_set_font_size(False)
    tbl4.set_fontsize(8.0)
    for (r, c), cell in tbl4.get_celld().items():
        cell.set_edgecolor(C["grid"])
        if r == 0:
            cell.set_facecolor(C["hf"])
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#f5f0dc")
        elif r % 2 == 0:
            cell.set_facecolor("#eef1f6")
        else:
            cell.set_facecolor("white")
        if r > 0 and c == 4:
            try:
                pv = float(coef_rows[r - 1][4])
                if pv < 0.01:
                    cell.set_facecolor("#c8f0c8")
                elif pv < 0.05:
                    cell.set_facecolor("#e8f4e8")
                elif pv < 0.15:
                    cell.set_facecolor("#fef9e7")
            except:
                pass
    ax4.set_title(
        "Table des coefficients (*** p<0.01 | ** p<0.05 | * p<0.10 | . p<0.15  |  cellules vertes = significatif)",
        fontweight="bold",
        pad=8,
    )

    # E – Résidus
    ax5 = fig.add_subplot(gs[3, 0])
    resid = ols_model.resid
    bar_cols = [C["pos"] if r >= 0 else C["neg"] for r in resid.values]
    ax5.bar(resid.index, resid.values, color=bar_cols, alpha=0.75, width=20)
    ax5.axhline(0, color=C["dark"], lw=1)
    dw_v = diag_res["dw"]
    bp_v = diag_res["bp_p"]
    ax5.set_title(
        f"Résidus OLS  |  DW = {dw_v:.2f}  |  BP p = {bp_v:.3f}",
        fontweight="bold",
        pad=6,
    )
    ax5.yaxis.set_major_formatter(fmt_pct)
    ax5.set_ylabel("Résidu")
    ax5.set_facecolor(C["bg"])
    ax5.grid(True, color=C["grid"], lw=0.6, axis="y")

    # F – Poids
    ax6 = fig.add_subplot(gs[3, 1])
    ws = weights.sort_values()
    bc = [C["hf"] if v > 0 else C["clone"] for v in ws]
    bars = ax6.barh(
        ws.index, ws.values, color=bc, alpha=0.85, height=0.6, edgecolor="white", lw=0.5
    )
    for bar, val in zip(bars, ws.values):
        ax6.text(
            val + (0.5 if val >= 0 else -0.5),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}%",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8.5,
            fontweight="bold",
            color=C["dark"],
        )
    ax6.axvline(0, color="#444", lw=1)
    ax6.set_title("Poids du portefeuille de réplication", fontweight="bold", pad=6)
    ax6.set_xlabel("Poids normalisé (%)")
    ax6.set_facecolor(C["bg"])
    ax6.grid(True, color=C["grid"], lw=0.6, axis="x")

    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print(f"\n  ✓ Graphique sauvegardé → {OUTPUT_PNG}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 6. POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_clone_search(target_r2=TARGET_R2)
