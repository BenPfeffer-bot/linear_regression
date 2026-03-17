"""
Streamlit Dashboard — Hedge Fund Clone Analysis
Aberdeen Orbita Capital Return Strategy Ltd A (GBP)
"""

from __future__ import annotations
import warnings
import time
import itertools

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yfinance as yf

# ── Import data & config from mini-project ──────────────────────────────────
from importlib.machinery import SourceFileLoader

_mp = SourceFileLoader("mini_project", "mini-project.py").load_module()

ABERDEEN_MONTHLY = _mp.ABERDEEN_MONTHLY
UNIVERSE = _mp.UNIVERSE
ALL_TICKERS = _mp.ALL_TICKERS
START_DATE = _mp.START_DATE
END_DATE = _mp.END_DATE

# ── Utility functions ────────────────────────────────────────────────────────

def sharpe(r):
    return r.mean() / r.std() * np.sqrt(12)

def ann_ret(r):
    return (1 + r.mean()) ** 12 - 1

def ann_vol(r):
    return r.std() * np.sqrt(12)

def max_dd(r):
    cum = (1 + r).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HF Clone — Aberdeen Orbita",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .metric-card .value {
        font-size: 2.4rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        line-height: 1.1;
    }

    .metric-card .label {
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .phase-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .phase-1 { background: #0d47a1; color: white; }
    .phase-2 { background: #e65100; color: white; }
    .phase-3 { background: #1b5e20; color: white; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
    }

    div[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }

    .highlight-box {
        background: linear-gradient(135deg, #0a1628 0%, #1a2744 100%);
        border-left: 4px solid #4fc3f7;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        margin: 12px 0;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    target_r2 = st.slider(
        "Target R\u00b2 adjusted", 0.10, 0.70, 0.45, 0.05,
        help="Minimum R\u00b2 adjusted for the stepwise to stop"
    )
    max_factors = st.slider("Max factors", 5, 30, 25)
    p_threshold = st.slider("P-value threshold", 0.05, 0.30, 0.25, 0.05)
    dedup_corr = st.slider("Dedup correlation", 0.70, 0.95, 0.85, 0.05,
                           help="Remove features with pairwise correlation above this")

    st.markdown("---")
    st.markdown("### Asset Universe")
    selected_classes = st.multiselect(
        "Asset classes",
        list(UNIVERSE.keys()),
        default=list(UNIVERSE.keys()),
    )

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; opacity:0.5; font-size:0.75rem;">'
        'Hedge Fund Clone Engine<br>Built with Streamlit & Plotly'
        '</div>',
        unsafe_allow_html=True,
    )

run_button = st.sidebar.button("Run Clone Analysis", type="primary", use_container_width=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="text-align:center; font-family: Inter, sans-serif; '
    'font-weight:700; letter-spacing:-1px;">'
    'Hedge Fund Clone Engine'
    '</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align:center; opacity:0.6; margin-top:-10px; font-size:1.1rem;">'
    'Aberdeen Orbita Capital Return Strategy Ltd A (GBP) &mdash; '
    f'{START_DATE} to {END_DATE}'
    '</p>',
    unsafe_allow_html=True,
)


# ── Main logic ───────────────────────────────────────────────────────────────
if run_button or "results" in st.session_state:

    if run_button:
        # Build ticker list from selected classes
        tickers = [t for cls in selected_classes for t in UNIVERSE.get(cls, [])]

        with st.status("Running clone analysis...", expanded=True) as status:

            # ── Download ─────────────────────────────────────────────────────
            st.write("Downloading ETF data from Yahoo Finance...")
            all_prices = []
            failed = []
            batch_size = 30

            for i in range(0, len(tickers), batch_size):
                batch = tickers[i : i + batch_size]
                try:
                    raw = yf.download(
                        batch, start=START_DATE, end=END_DATE,
                        auto_adjust=True, progress=False, threads=True,
                    )
                    if isinstance(raw.columns, pd.MultiIndex):
                        close = raw["Close"]
                    else:
                        close = raw[["Close"]] if "Close" in raw.columns else raw
                    all_prices.append(close)
                except Exception:
                    failed.extend(batch)
                time.sleep(0.5)

            prices_daily = pd.concat(all_prices, axis=1)
            prices_daily = prices_daily.loc[:, ~prices_daily.columns.duplicated()]
            prices_monthly = prices_daily.resample("ME").last()
            returns_monthly = prices_monthly.pct_change().dropna(how="all")
            threshold = int(0.80 * len(returns_monthly))
            returns_monthly = returns_monthly.dropna(axis=1, thresh=threshold)

            st.write(f"Loaded {returns_monthly.shape[1]} assets over {returns_monthly.shape[0]} months")

            # ── Align Aberdeen ───────────────────────────────────────────────
            target_dates = returns_monthly.index
            n_align = min(len(ABERDEEN_MONTHLY), len(target_dates))
            y = pd.Series(
                ABERDEEN_MONTHLY[-n_align:],
                index=target_dates[-n_align:],
                name="Aberdeen",
            )
            X_raw = returns_monthly.loc[y.index].copy()
            X_raw = X_raw.dropna(axis=1, thresh=int(0.90 * len(y)))
            X_raw = X_raw.fillna(X_raw.median())

            # ── Correlations ─────────────────────────────────────────────────
            corr_raw = X_raw.corrwith(y).abs().sort_values(ascending=False)

            # ── Feature engineering ──────────────────────────────────────────
            st.write("Engineering features (lags, rolling, interactions)...")
            X_enrich = X_raw.copy()
            for col in X_raw.columns:
                X_enrich[f"{col}_lag1"] = X_raw[col].shift(1)
                X_enrich[f"{col}_lag2"] = X_raw[col].shift(2)
            for col in corr_raw.head(25).index:
                X_enrich[f"{col}_ma3"] = X_raw[col].rolling(3).mean()
                X_enrich[f"{col}_ma6"] = X_raw[col].rolling(6).mean()
            for col in corr_raw.head(15).index:
                X_enrich[f"{col}_vol3"] = X_raw[col].rolling(3).std()
            for col in corr_raw.head(25).index:
                X_enrich[f"{col}_sq"] = X_raw[col] ** 2
            top_interact = corr_raw.head(12).index.tolist()
            for a, b in itertools.combinations(top_interact, 2):
                X_enrich[f"{a}x{b}"] = X_raw[a] * X_raw[b]

            X_enrich = X_enrich.dropna()
            y_enrich = y.loc[X_enrich.index]
            n_raw = X_enrich.shape[1]

            # Dedup
            corr_matrix = X_enrich.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = set()
            for col in upper.columns:
                highly_corr = upper.index[upper[col] > dedup_corr].tolist()
                if highly_corr:
                    candidates = [col] + highly_corr
                    corrs_y = {c: abs(X_enrich[c].corr(y_enrich)) for c in candidates if c not in to_drop}
                    if corrs_y:
                        keep = max(corrs_y, key=corrs_y.get)
                        to_drop.update(c for c in candidates if c != keep)
            X_enrich = X_enrich.drop(columns=list(to_drop))

            st.write(f"{n_raw} features -> {X_enrich.shape[1]} after deduplication")

            # ── PCA ──────────────────────────────────────────────────────────
            scaler_pca = StandardScaler()
            X_sc = scaler_pca.fit_transform(X_enrich)
            n_comp = min(30, X_enrich.shape[1], X_enrich.shape[0] - 1)
            pca = PCA(n_components=n_comp)
            pca_scores = pca.fit_transform(X_sc)
            for i in range(n_comp):
                X_enrich[f"PC{i+1}"] = pca_scores[:, i]

            # ── Forward stepwise ─────────────────────────────────────────────
            st.write("Running forward stepwise selection...")
            remaining = list(X_enrich.columns)
            selected = []
            best_r2 = 0.0
            history = []

            for step in range(max_factors):
                best_new = None
                best_new_r2 = best_r2
                best_p = 1.0
                for cand in remaining:
                    try:
                        res = sm.OLS(y_enrich, sm.add_constant(X_enrich[selected + [cand]])).fit()
                        pval = float(res.pvalues.get(cand, 1.0))
                        if res.rsquared_adj > best_new_r2 and pval < p_threshold:
                            best_new_r2 = res.rsquared_adj
                            best_new = cand
                            best_p = pval
                    except Exception:
                        continue
                if best_new is None:
                    break
                selected.append(best_new)
                remaining.remove(best_new)
                best_r2 = best_new_r2
                history.append({
                    "step": step + 1, "factor": best_new,
                    "r2_adj": best_new_r2, "p_value": best_p,
                })
                if best_r2 >= target_r2:
                    break

            # ── Final OLS ────────────────────────────────────────────────────
            st.write("Fitting final OLS model...")
            X_final = sm.add_constant(X_enrich[selected])
            model = sm.OLS(y_enrich, X_final).fit()
            y_hat = model.fittedvalues.rename("Clone")
            resid = model.resid

            # Diagnostics
            dw = durbin_watson(resid)
            _, bp_p, _, _ = het_breuschpagan(resid, X_final)
            jb_stat, jb_p = stats.jarque_bera(resid)
            vif_vals = []
            for i in range(X_final.shape[1]):
                vif_vals.append(variance_inflation_factor(X_final.values, i))
            vif_df = pd.DataFrame({"Variable": X_final.columns, "VIF": vif_vals})
            vif_df = vif_df[vif_df["Variable"] != "const"]

            params = model.params.drop("const")
            weights = params / params.abs().sum() * 100

            status.update(label="Analysis complete!", state="complete")

        # Store in session
        st.session_state["results"] = {
            "y": y_enrich, "y_hat": y_hat, "model": model,
            "selected": selected, "history": pd.DataFrame(history),
            "corr_raw": corr_raw, "params": params, "weights": weights,
            "resid": resid, "dw": dw, "bp_p": bp_p, "jb_p": jb_p,
            "vif_df": vif_df, "X_raw": X_raw,
        }

    # ── Retrieve results ─────────────────────────────────────────────────
    R = st.session_state["results"]
    y = R["y"]
    y_hat = R["y_hat"]
    model = R["model"]
    selected = R["selected"]
    hist_df = R["history"]
    corr_raw = R["corr_raw"]
    params = R["params"]
    weights = R["weights"]
    resid = R["resid"]
    dw = R["dw"]
    bp_p = R["bp_p"]
    jb_p = R["jb_p"]
    vif_df = R["vif_df"]

    # ── KPI Cards ────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)

    def metric_card(col, value, label, color="#4fc3f7"):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="value" style="color:{color}">{value}</div>'
            f'<div class="label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    metric_card(c1, f"{model.rsquared_adj*100:.1f}%", "R\u00b2 Adjusted", "#4fc3f7")
    metric_card(c2, f"{model.rsquared*100:.1f}%", "R\u00b2 Raw", "#81c784")
    metric_card(c3, f"{len(selected)}", "Factors", "#ffb74d")
    metric_card(c4, f"{model.f_pvalue:.2e}", "F-test p-value", "#e57373")
    metric_card(
        c5,
        f"{((1 + model.params['const'])**12 - 1)*100:.1f}%",
        "Annualized Alpha",
        "#ce93d8",
    )

    st.markdown("")

    # ── Tab layout ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Performance", "Factor Analysis", "Diagnostics",
        "Portfolio", "Stepwise History",
    ])

    # ────────────────────────────────────────────────────────────────────
    # TAB 1: PERFORMANCE
    # ────────────────────────────────────────────────────────────────────
    with tab1:
        col_l, col_r = st.columns([2, 1])

        with col_l:
            # Cumulative performance
            cum_hf = (1 + y).cumprod() - 1
            cum_cl = (1 + y_hat).cumprod() - 1

            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(
                x=cum_hf.index, y=cum_hf.values,
                name="Aberdeen HF", line=dict(color="#4fc3f7", width=2.5),
                fill="tonexty" if False else None,
            ))
            fig_perf.add_trace(go.Scatter(
                x=cum_cl.index, y=cum_cl.values,
                name="OLS Clone", line=dict(color="#ff8a65", width=2, dash="dash"),
            ))
            # Fill between
            fig_perf.add_trace(go.Scatter(
                x=list(cum_hf.index) + list(cum_cl.index[::-1]),
                y=list(cum_hf.values) + list(cum_cl.values[::-1]),
                fill="toself", fillcolor="rgba(255,138,101,0.08)",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig_perf.update_layout(
                title="Cumulative Performance",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.8)",
                yaxis_tickformat=".0%",
                height=420,
                legend=dict(x=0.02, y=0.98),
                margin=dict(l=60, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        with col_r:
            st.markdown("### Performance Comparison")
            perf_data = {
                "Metric": [
                    "Annualized Return", "Annualized Volatility",
                    "Sharpe Ratio", "Max Drawdown", "Skewness", "Kurtosis",
                ],
                "Aberdeen HF": [
                    f"{ann_ret(y)*100:.2f}%", f"{ann_vol(y)*100:.2f}%",
                    f"{sharpe(y):.2f}", f"{max_dd(y)*100:.2f}%",
                    f"{y.skew():.3f}", f"{y.kurt():.3f}",
                ],
                "OLS Clone": [
                    f"{ann_ret(y_hat)*100:.2f}%", f"{ann_vol(y_hat)*100:.2f}%",
                    f"{sharpe(y_hat):.2f}", f"{max_dd(y_hat)*100:.2f}%",
                    f"{y_hat.skew():.3f}", f"{y_hat.kurt():.3f}",
                ],
            }
            st.dataframe(
                pd.DataFrame(perf_data).set_index("Metric"),
                use_container_width=True,
            )

            # Monthly returns distribution
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=y.values, name="Aberdeen", opacity=0.7,
                marker_color="#4fc3f7", nbinsx=20,
            ))
            fig_dist.add_trace(go.Histogram(
                x=y_hat.values, name="Clone", opacity=0.7,
                marker_color="#ff8a65", nbinsx=20,
            ))
            fig_dist.update_layout(
                title="Monthly Returns Distribution",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.8)",
                barmode="overlay",
                xaxis_tickformat=".1%",
                height=250,
                margin=dict(l=40, r=20, t=40, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # Scatter plot
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_hat.values, y=y.values, mode="markers",
                marker=dict(
                    color=y.values, colorscale="RdYlBu", size=8,
                    line=dict(width=0.5, color="white"),
                    colorbar=dict(title="HF Return"),
                ),
                text=[d.strftime("%Y-%m") for d in y.index],
                hovertemplate="Clone: %{x:.2%}<br>HF: %{y:.2%}<br>%{text}<extra></extra>",
            ))
            # Regression line
            z = np.polyfit(y_hat.values, y.values, 1)
            xl = np.linspace(y_hat.min(), y_hat.max(), 100)
            fig_scatter.add_trace(go.Scatter(
                x=xl, y=np.poly1d(z)(xl),
                mode="lines", line=dict(color="#ff8a65", width=2, dash="dash"),
                showlegend=False,
            ))
            fig_scatter.update_layout(
                title=f"HF vs Clone Returns (R\u00b2 = {model.rsquared:.3f})",
                xaxis_title="Clone Return", yaxis_title="HF Return",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.8)",
                xaxis_tickformat=".1%", yaxis_tickformat=".1%",
                height=380,
                margin=dict(l=60, r=20, t=50, b=50),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_s2:
            # Rolling 12m correlation
            rolling_corr = y.rolling(12).corr(y_hat)
            fig_rcorr = go.Figure()
            fig_rcorr.add_trace(go.Scatter(
                x=rolling_corr.index, y=rolling_corr.values,
                fill="tozeroy", fillcolor="rgba(79,195,247,0.15)",
                line=dict(color="#4fc3f7", width=2),
            ))
            fig_rcorr.add_hline(y=rolling_corr.mean(), line_dash="dash",
                                line_color="#ff8a65", annotation_text=f"Mean: {rolling_corr.mean():.2f}")
            fig_rcorr.update_layout(
                title="Rolling 12-Month Correlation (HF vs Clone)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.8)",
                height=380, yaxis_range=[-0.5, 1],
                margin=dict(l=60, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_rcorr, use_container_width=True)


    # ────────────────────────────────────────────────────────────────────
    # TAB 2: FACTOR ANALYSIS
    # ────────────────────────────────────────────────────────────────────
    with tab2:
        col_coef, col_corr = st.columns([3, 2])

        with col_coef:
            # Coefficient table
            st.markdown("### OLS Coefficients")
            ci = model.conf_int()
            coef_data = []
            for var in model.params.index:
                pval = model.pvalues[var]
                sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
                coef_data.append({
                    "Variable": "Intercept" if var == "const" else var,
                    "Coefficient": model.params[var],
                    "Std Error": model.bse[var],
                    "t-stat": model.tvalues[var],
                    "p-value": pval,
                    "Sig.": sig,
                })
            coef_df = pd.DataFrame(coef_data)
            st.dataframe(
                coef_df.style.format({
                    "Coefficient": "{:.6f}", "Std Error": "{:.6f}",
                    "t-stat": "{:.3f}", "p-value": "{:.6f}",
                }).background_gradient(subset=["p-value"], cmap="RdYlGn", vmin=0, vmax=0.15),
                use_container_width=True, height=500,
            )

        with col_corr:
            # Top correlations
            st.markdown("### Top Raw Correlations with Fund")
            top_corr = corr_raw.head(20)
            fig_corr = go.Figure(go.Bar(
                x=top_corr.values, y=top_corr.index,
                orientation="h",
                marker=dict(
                    color=top_corr.values,
                    colorscale=[[0, "#1a237e"], [0.5, "#4fc3f7"], [1, "#ff8a65"]],
                ),
            ))
            fig_corr.update_layout(
                title="Absolute Correlation with Aberdeen",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.8)",
                height=500, yaxis=dict(autorange="reversed"),
                margin=dict(l=80, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # ANOVA summary
        st.markdown("### ANOVA Table")
        n = int(model.nobs)
        k = len(selected)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        ss_res = float(np.sum(resid ** 2))
        ss_reg = ss_tot - ss_res
        df_reg, df_res = k, n - k - 1

        anova_df = pd.DataFrame({
            "Source": ["Regression", "Residuals", "Total"],
            "SS": [ss_reg, ss_res, ss_tot],
            "df": [df_reg, df_res, n - 1],
            "MS": [ss_reg / df_reg, ss_res / df_res, np.nan],
            "F-stat": [model.fvalue, np.nan, np.nan],
            "p-value": [model.f_pvalue, np.nan, np.nan],
        })
        st.dataframe(
            anova_df.style.format({
                "SS": "{:.8f}", "MS": "{:.8f}",
                "F-stat": "{:.4f}", "p-value": "{:.2e}",
            }, na_rep=""),
            use_container_width=True,
        )


    # ────────────────────────────────────────────────────────────────────
    # TAB 3: DIAGNOSTICS
    # ────────────────────────────────────────────────────────────────────
    with tab3:
        # Diagnostic badges
        d1, d2, d3 = st.columns(3)
        with d1:
            ok = 1.5 < dw < 2.5
            st.metric("Durbin-Watson", f"{dw:.4f}", "PASS" if ok else "FAIL",
                       delta_color="normal" if ok else "inverse")
        with d2:
            ok = bp_p > 0.05
            st.metric("Breusch-Pagan p", f"{bp_p:.4f}", "PASS" if ok else "FAIL",
                       delta_color="normal" if ok else "inverse")
        with d3:
            ok = jb_p > 0.05
            st.metric("Jarque-Bera p", f"{jb_p:.4f}", "PASS" if ok else "FAIL",
                       delta_color="normal" if ok else "inverse")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            # Residual bar chart
            colors = ["#81c784" if r >= 0 else "#e57373" for r in resid.values]
            fig_resid = go.Figure(go.Bar(
                x=resid.index, y=resid.values,
                marker_color=colors, opacity=0.8,
            ))
            fig_resid.add_hline(y=0, line_color="white", line_width=1)
            fig_resid.update_layout(
                title="OLS Residuals Over Time",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.8)",
                yaxis_tickformat=".1%",
                height=350,
                margin=dict(l=60, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_resid, use_container_width=True)

        with col_r2:
            # QQ plot
            (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=osm, y=osr, mode="markers",
                marker=dict(color="#4fc3f7", size=6, line=dict(width=0.5, color="white")),
                name="Residuals",
            ))
            fig_qq.add_trace(go.Scatter(
                x=[osm.min(), osm.max()],
                y=[slope * osm.min() + intercept, slope * osm.max() + intercept],
                mode="lines", line=dict(color="#ff8a65", dash="dash"),
                name="Normal line",
            ))
            fig_qq.update_layout(
                title="Q-Q Plot (Normality of Residuals)",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.8)",
                height=350,
                margin=dict(l=60, r=20, t=50, b=50),
            )
            st.plotly_chart(fig_qq, use_container_width=True)

        # VIF table
        st.markdown("### Variance Inflation Factors")
        vif_styled = vif_df.copy()
        vif_styled["Status"] = vif_styled["VIF"].apply(lambda v: "OK" if v < 5 else "HIGH")
        st.dataframe(
            vif_styled.style.format({"VIF": "{:.2f}"}).applymap(
                lambda v: "color: #81c784" if v == "OK" else "color: #e57373",
                subset=["Status"],
            ),
            use_container_width=True,
        )


    # ────────────────────────────────────────────────────────────────────
    # TAB 4: PORTFOLIO
    # ────────────────────────────────────────────────────────────────────
    with tab4:
        col_w, col_t = st.columns([3, 2])

        with col_w:
            ws = weights.sort_values()
            colors_w = ["#1a237e" if v > 0 else "#e65100" for v in ws.values]
            fig_w = go.Figure(go.Bar(
                y=ws.index, x=ws.values, orientation="h",
                marker_color=colors_w, opacity=0.9,
                text=[f"{v:+.1f}%" for v in ws.values],
                textposition="outside", textfont=dict(size=11),
            ))
            fig_w.add_vline(x=0, line_color="white", line_width=1)
            fig_w.update_layout(
                title="Replication Portfolio Weights",
                xaxis_title="Normalized Weight (%)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.8)",
                height=max(400, len(ws) * 28),
                margin=dict(l=120, r=80, t=50, b=40),
            )
            st.plotly_chart(fig_w, use_container_width=True)

        with col_t:
            st.markdown("### Portfolio Breakdown")
            port_df = pd.DataFrame({
                "Factor": params.index,
                "Beta": params.values,
                "Weight %": weights.values,
                "Direction": ["LONG" if w > 0 else "SHORT" for w in weights.values],
            }).sort_values("Weight %", ascending=False)
            st.dataframe(
                port_df.style.format({"Beta": "{:.6f}", "Weight %": "{:+.2f}%"}).applymap(
                    lambda v: "color: #81c784" if v == "LONG" else "color: #e57373",
                    subset=["Direction"],
                ),
                use_container_width=True, height=500,
            )

            alpha_m = model.params["const"]
            st.markdown(
                f'<div class="highlight-box">'
                f'<b>Monthly Alpha:</b> {alpha_m*100:.4f}%<br>'
                f'<b>Annualized Alpha:</b> {((1+alpha_m)**12 - 1)*100:.2f}%<br>'
                f'<b>R\u00b2 Adjusted:</b> {model.rsquared_adj*100:.2f}%'
                f'</div>',
                unsafe_allow_html=True,
            )


    # ────────────────────────────────────────────────────────────────────
    # TAB 5: STEPWISE HISTORY
    # ────────────────────────────────────────────────────────────────────
    with tab5:
        col_h1, col_h2 = st.columns([3, 2])

        with col_h1:
            fig_step = make_subplots(specs=[[{"secondary_y": True}]])
            fig_step.add_trace(go.Scatter(
                x=hist_df["step"], y=hist_df["r2_adj"],
                mode="lines+markers",
                name="R\u00b2 Adjusted",
                line=dict(color="#4fc3f7", width=3),
                marker=dict(size=10, symbol="circle"),
            ), secondary_y=False)
            fig_step.add_trace(go.Bar(
                x=hist_df["step"], y=hist_df["p_value"],
                name="p-value", opacity=0.4,
                marker_color="#ff8a65",
            ), secondary_y=True)
            fig_step.add_hline(
                y=target_r2, line_dash="dot", line_color="#81c784",
                annotation_text=f"Target: {target_r2*100:.0f}%",
                secondary_y=False,
            )
            fig_step.update_layout(
                title="Stepwise Factor Selection Progress",
                xaxis_title="Step",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.8)",
                height=450,
                margin=dict(l=60, r=60, t=50, b=40),
            )
            fig_step.update_yaxes(title_text="R\u00b2 Adjusted", tickformat=".0%", secondary_y=False)
            fig_step.update_yaxes(title_text="p-value", secondary_y=True)
            st.plotly_chart(fig_step, use_container_width=True)

        with col_h2:
            st.markdown("### Selection Log")
            log_df = hist_df.copy()
            log_df["r2_adj"] = log_df["r2_adj"].map(lambda x: f"{x*100:.2f}%")
            log_df["p_value"] = log_df["p_value"].map(lambda x: f"{x:.6f}")
            st.dataframe(log_df, use_container_width=True, height=450)

        # Marginal R2 contribution
        if len(hist_df) > 1:
            marginal = hist_df["r2_adj"].diff().fillna(hist_df["r2_adj"].iloc[0])
            fig_marg = go.Figure(go.Bar(
                x=hist_df["factor"], y=marginal * 100,
                marker_color=px.colors.sample_colorscale(
                    "Viridis", np.linspace(0.2, 0.9, len(marginal))
                ),
                text=[f"+{v*100:.2f}%" for v in marginal],
                textposition="outside",
            ))
            fig_marg.update_layout(
                title="Marginal R\u00b2 Contribution per Factor",
                yaxis_title="Marginal R\u00b2 adj (%)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,35,0.8)",
                height=350,
                xaxis_tickangle=-45,
                margin=dict(l=60, r=20, t=50, b=100),
            )
            st.plotly_chart(fig_marg, use_container_width=True)

else:
    # Landing state
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="value" style="color:#4fc3f7">88</div>'
            '<div class="label">ETFs in Universe</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="value" style="color:#81c784">10</div>'
            '<div class="label">Asset Classes</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="metric-card">'
            '<div class="value" style="color:#ff8a65">119</div>'
            '<div class="label">Monthly Observations</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        '<div class="highlight-box">'
        '<b>How it works:</b> This engine downloads ~90 ETF/indices via Yahoo Finance, '
        'engineers 300+ features (lags, rolling averages, quadratics, cross-interactions), '
        'deduplicates correlated features, then runs a forward stepwise OLS regression '
        'to find the best factor model that replicates the hedge fund\'s returns.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(
        '<p style="text-align:center; opacity:0.5; font-size:1.1rem;">'
        'Configure parameters in the sidebar, then click <b>Run Clone Analysis</b> to start.'
        '</p>',
        unsafe_allow_html=True,
    )
