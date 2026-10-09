import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
from scipy.stats import linregress

st.set_page_config(page_title="PhytoRelease Lab", page_icon="🧪", layout="wide")

# ----------------------------- Model functions -----------------------------
def first_order_release(t, m0, k):
    """Cumulative release for a first-order empirical model."""
    t = np.asarray(t, dtype=float)
    if m0 < 0 or k < 0:
        raise ValueError("Loading and release constant must be non-negative.")
    # -expm1(-x) is numerically stable for small x.
    return m0 * (-np.expm1(-k * t))

def time_to_fraction(fraction, k):
    """Time required to reach a release fraction under the first-order model."""
    if not 0 < fraction < 1 or k <= 0:
        return np.nan
    return -math.log(1 - fraction) / k

def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return np.nan
    return 1 - ss_res / ss_tot

def validate_csv(df):
    required = {"time", "measured_cumulative_release"}
    if not required.issubset(set(df.columns)):
        return None, "CSV must contain columns named time and measured_cumulative_release."
    data = df[["time", "measured_cumulative_release"]].copy()
    data["time"] = pd.to_numeric(data["time"], errors="coerce")
    data["measured_cumulative_release"] = pd.to_numeric(data["measured_cumulative_release"], errors="coerce")
    data = data.dropna()
    if len(data) < 3:
        return None, "Provide at least three rows with valid numeric values."
    if (data["time"] < 0).any() or (data["measured_cumulative_release"] < 0).any():
        return None, "Time and measured cumulative release must be non-negative."
    if data["time"].nunique() < 3:
        return None, "At least three distinct time values are required for fitting."
    data = data.sort_values("time").reset_index(drop=True)
    return data, None

def fit_first_order(data):
    t = data["time"].to_numpy(dtype=float)
    y = data["measured_cumulative_release"].to_numpy(dtype=float)
    if np.max(y) <= 0:
        raise ValueError("All measured release values are zero; k cannot be meaningfully fitted.")
    # Fit M0 and k; non-negative bounds keep parameters physically interpretable.
    p0 = [max(float(np.max(y)), 1e-6), 0.1]
    params, _ = curve_fit(first_order_release, t, y, p0=p0,
                          bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=20000)
    m0_fit, k_fit = [float(x) for x in params]
    predicted = first_order_release(t, m0_fit, k_fit)
    rmse = float(np.sqrt(np.mean((y - predicted) ** 2)))
    r2 = safe_r2(y, predicted)
    return m0_fit, k_fit, predicted, rmse, r2

def make_release_figure(t, curves, y_title, percent=False):
    fig = go.Figure()
    for label, values in curves:
        fig.add_trace(go.Scatter(x=t, y=values, mode="lines", name=label,
                                 hovertemplate="Time: %{x:.3g}<br>Value: %{y:.4g}<extra>%{fullData.name}</extra>"))
    fig.update_layout(template="plotly_white", height=390, margin=dict(l=20, r=20, t=30, b=20),
                      xaxis_title="Time (same unit used for k)", yaxis_title=y_title,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    if percent:
        fig.update_yaxes(range=[0, 100])
    return fig

# ----------------------------- Header -----------------------------
st.title("🧪 PhytoRelease Lab")
st.markdown("### Explore phytochemical release. Compare scaffold formulations. Generate testable research hypotheses.")
st.caption("Interactive computational research prototype • First-order empirical release model")
st.info("**Research-use disclaimer:** outputs are mathematical estimates based on entered parameters. They are not experimental measurements, evidence of therapeutic efficacy, proof of clinical safety, or validated predictions of real scaffold performance.")

# ----------------------------- Sidebar controls -----------------------------
with st.sidebar:
    st.header("Simulation controls")
    phytochemical = st.selectbox("Phytochemical", ["EGCG", "Punicalagin", "Green tea extract", "Custom phytochemical"])
    if phytochemical == "Custom phytochemical":
        phytochemical = st.text_input("Custom phytochemical name", "Custom compound")
    material = st.selectbox("Scaffold material", ["Gelatin", "GelMA", "Alginate", "Custom material"])
    if material == "Custom material":
        material = st.text_input("Custom material name", "Custom scaffold")
    st.markdown("---")
    st.subheader("Main formulation")
    loading = st.number_input("Initial phytochemical loading", min_value=0.0, value=10.0, step=0.5,
                               help="Total initially loaded amount. Use a consistent mass unit, e.g. mg.")
    unit = st.selectbox("Loading / release unit", ["mg", "µg", "g"])
    use_encapsulation = st.checkbox("Apply encapsulation efficiency")
    efficiency = st.slider("Encapsulation efficiency (%)", 0.0, 100.0, 80.0, 1.0, disabled=not use_encapsulation)
    porosity = st.slider("Scaffold porosity (%) — descriptive only", 1.0, 99.0, 60.0, 1.0)
    diffusion = st.number_input("Effective diffusion coefficient (m²/s) — descriptive only",
                                min_value=0.0, value=1e-12, format="%.2e",
                                help="Recorded for context only; not used to derive k in this model.")
    degradation = st.number_input("Degradation parameter (1/time) — descriptive only",
                                  min_value=0.0, value=0.01, format="%.4f",
                                  help="Recorded for context only; not used to derive k in this model.")
    st.markdown("---")
    st.subheader("Release model")
    k = st.number_input("Effective first-order release constant k (1/time)", min_value=0.0,
                         value=0.15, step=0.01, format="%.4f",
                         help="Enter or experimentally fit this parameter. Its time unit is the inverse of your time axis unit.")
    duration = st.number_input("Simulation duration", min_value=0.1, value=10.0, step=1.0)
    n_points = st.slider("Number of time points", min_value=20, max_value=500, value=120, step=10)
    time_unit = st.selectbox("Time unit", ["hours", "days", "minutes"])

    releasable = loading * (efficiency / 100.0 if use_encapsulation else 1.0)
    t = np.linspace(0, float(duration), int(n_points))
    released = first_order_release(t, releasable, float(k))
    remaining = np.maximum(releasable - released, 0.0)
    percent = (released / releasable * 100.0) if releasable > 0 else np.zeros_like(released)

# ----------------------------- KPI row -----------------------------
end_release = float(released[-1]) if len(released) else 0.0
end_remaining = float(remaining[-1]) if len(remaining) else 0.0
end_pct = float(percent[-1]) if len(percent) else 0.0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Releasable initial amount", f"{releasable:.3g} {unit}")
c2.metric("Released at end", f"{end_release:.3g} {unit}")
c3.metric("Released at end", f"{end_pct:.1f}%")
c4.metric("Estimated remaining", f"{end_remaining:.3g} {unit}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Release dashboard", "Formulation comparison", "Sensitivity & target", "Experimental data fit", "Model & exports"
])

with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Cumulative release")
        st.plotly_chart(make_release_figure(t, [(f"{phytochemical} • {material}", released)], f"Cumulative amount released ({unit})"),
                        use_container_width=True)
    with right:
        st.subheader("Estimated amount remaining")
        st.plotly_chart(make_release_figure(t, [(f"{phytochemical} • {material}", remaining)], f"Amount remaining ({unit})"),
                        use_container_width=True)
    st.subheader("Cumulative percentage released")
    st.plotly_chart(make_release_figure(t, [(f"{phytochemical} • {material}", percent)], "Cumulative release (%)", percent=True),
                    use_container_width=True)
    main_df = pd.DataFrame({
        f"Time ({time_unit})": t,
        f"Cumulative released ({unit})": released,
        f"Remaining ({unit})": remaining,
        "Cumulative released (%)": percent
    })
    st.dataframe(main_df.head(15), use_container_width=True, hide_index=True)
    st.caption(f"Context inputs (not used to calculate k): material = {material}; porosity = {porosity:.1f}%; diffusion coefficient = {diffusion:.2e} m²/s; degradation parameter = {degradation:g} 1/time.")

with tab2:
    st.subheader("Compare up to three hypothetical formulations")
    st.write("Each formulation has its own loading and independently entered release constant. Comparisons use the same first-order equation.")
    compare_duration = st.number_input("Comparison duration", min_value=0.1, value=float(duration), step=1.0, key="compare_duration")
    compare_time = st.number_input("Evaluate comparison at time", min_value=0.0, value=float(duration), step=0.5, key="compare_time")
    compare_time = min(compare_time, compare_duration)
    comp_specs = []
    cols = st.columns(3)
    defaults = [
        ("Formulation A", float(loading), float(k)),
        ("Formulation B", max(float(loading) * 1.2, 0.0), max(float(k) * 0.7, 0.0)),
        ("Formulation C", max(float(loading) * 0.8, 0.0), max(float(k) * 1.4, 0.0)),
    ]
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Formulation {chr(65+i)}**")
            name = st.text_input("Name", defaults[i][0], key=f"comp_name_{i}")
            m0 = st.number_input(f"Initial loading ({unit})", min_value=0.0, value=float(defaults[i][1]), step=0.5, key=f"comp_m0_{i}")
            ck = st.number_input("Release constant k (1/time)", min_value=0.0, value=float(defaults[i][2]), step=0.01, format="%.4f", key=f"comp_k_{i}")
            comp_specs.append((name or f"Formulation {chr(65+i)}", m0, ck))
    ct = np.linspace(0, float(compare_duration), int(n_points))
    comp_curves = []
    rows = []
    for name, m0, ck in comp_specs:
        vals = first_order_release(ct, m0, ck)
        comp_curves.append((name, vals))
        released_at = float(first_order_release([compare_time], m0, ck)[0])
        rem_at = max(m0 - released_at, 0.0)
        pct_at = (released_at / m0 * 100.0) if m0 > 0 else 0.0
        rows.append({
            "Formulation": name, f"Initial loading ({unit})": m0, "k (1/time)": ck,
            f"Released at {compare_time:g} {time_unit} ({unit})": released_at,
            "Released (%)": pct_at, f"Remaining ({unit})": rem_at,
            f"Time to 50% ({time_unit})": time_to_fraction(0.5, ck)
        })
    st.plotly_chart(make_release_figure(ct, comp_curves, f"Cumulative amount released ({unit})"), use_container_width=True)
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.download_button("Download formulation comparison CSV", comp_df.to_csv(index=False).encode("utf-8"),
                       file_name="phytorelease_formulation_comparison.csv", mime="text/csv")
    st.caption("The table describes mathematical behavior only. It does not identify a clinically or biologically optimal formulation.")

with tab3:
    st.subheader("Release-constant sensitivity analysis")
    st.write("Change k to see how the mathematical curve changes. These curves are not experimental evidence.")
    k_values = st.multiselect("Select k values (1/time)", [0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0],
                              default=sorted(set([0.0, float(k), 0.5])))
    if not k_values:
        st.warning("Select at least one k value to plot sensitivity curves.")
    else:
        sensitivity_curves = [(f"k = {kv:g}", first_order_release(t, releasable, float(kv))) for kv in k_values]
        st.plotly_chart(make_release_figure(t, sensitivity_curves, f"Cumulative amount released ({unit})"), use_container_width=True)
    st.markdown("---")
    st.subheader("Optional target-release objective")
    target_pct = st.slider("Hypothetical desired release by target time (%)", 1, 99, 50)
    target_time = st.number_input("Target time", min_value=0.1, value=min(3.0, float(duration)), step=0.5)
    target_k = -math.log(1 - target_pct / 100.0) / target_time
    predicted_target = float(first_order_release([target_time], releasable, target_k)[0])
    st.write(f"To mathematically reach **{target_pct}%** release by **{target_time:g} {time_unit}**, the first-order model requires approximately **k = {target_k:.4g} 1/{time_unit}**.")
    st.write(f"At that target time, predicted release is **{predicted_target:.3g} {unit}** out of {releasable:.3g} {unit}.")
    st.caption("This is a user-defined modelling target, not an established therapeutic or product requirement.")

with tab4:
    st.subheader("Upload and fit experimental observations")
    st.write("CSV format: `time,measured_cumulative_release`. Use measured data only for this workflow; the included sample file is synthetic and clearly labelled.")
    uploaded = st.file_uploader("Upload experimental CSV", type=["csv"])
    if uploaded is None:
        st.info("No file uploaded. The simulation remains fully usable.")
        st.code("time,measured_cumulative_release\n0,0\n1,1.4\n2,2.6\n3,3.8", language="csv")
    else:
        try:
            raw_df = pd.read_csv(uploaded)
            data, error = validate_csv(raw_df)
            if error:
                st.error(error)
            else:
                st.dataframe(data, use_container_width=True, hide_index=True)
                try:
                    fit_m0, fit_k, fit_pred, rmse, r2 = fit_first_order(data)
                    f1, f2, f3, f4 = st.columns(4)
                    f1.metric("Fitted releasable amount", f"{fit_m0:.4g}")
                    f2.metric("Fitted k", f"{fit_k:.4g} 1/time")
                    f3.metric("RMSE", f"{rmse:.4g}")
                    f4.metric("R²", "Not defined" if np.isnan(r2) else f"{r2:.4f}")
                    fit_fig = go.Figure()
                    fit_fig.add_trace(go.Scatter(x=data["time"], y=data["measured_cumulative_release"],
                                                 mode="markers", name="Uploaded observations",
                                                 marker=dict(size=9, symbol="circle")))
                    dense_t = np.linspace(float(data["time"].min()), float(data["time"].max()), 200)
                    fit_fig.add_trace(go.Scatter(x=dense_t, y=first_order_release(dense_t, fit_m0, fit_k),
                                                 mode="lines", name="Fitted first-order model"))
                    fit_fig.update_layout(template="plotly_white", height=400,
                                          xaxis_title=f"Time ({time_unit}, confirm uploaded units)",
                                          yaxis_title=f"Measured cumulative release ({unit}, confirm uploaded units)")
                    st.plotly_chart(fit_fig, use_container_width=True)
                    fitted_table = data.copy()
                    fitted_table["fitted_prediction"] = first_order_release(data["time"], fit_m0, fit_k)
                    fitted_table["residual"] = fitted_table["measured_cumulative_release"] - fitted_table["fitted_prediction"]
                    st.download_button("Download fitted observations and predictions",
                                       fitted_table.to_csv(index=False).encode("utf-8"),
                                       file_name="phytorelease_fitted_results.csv", mime="text/csv")
                    st.caption("Confirm that uploaded time and release units match your intended units. Curve fit quality alone does not validate the biological mechanism.")
                except Exception as exc:
                    st.error(f"Could not fit this dataset: {exc}")
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")

with tab5:
    st.subheader("Mathematical model")
    st.latex(r"M_t=M_0(1-e^{-kt})")
    st.latex(r"M_{remaining}=M_0-M_t")
    st.latex(r"R(t)=100\\frac{M_t}{M_0}")
    st.markdown("""
    **Definitions**
    - `M₀`: releasable initial amount (loading unit).
    - `Mₜ`: cumulative amount released at time `t` (loading unit).
    - `k`: effective first-order release constant (inverse time).
    - `t`: elapsed time (time unit consistent with `k`).

    **Assumptions and limitations**
    - The model is a simplified empirical approximation.
    - It does not independently resolve diffusion, swelling, erosion, scaffold degradation, binding, or changing environmental conditions.
    - Porosity, diffusion coefficient, and degradation inputs are displayed as contextual metadata only. They are **not** silently converted into `k`.
    - `k` should be entered from a justified source or estimated from suitable experimental observations.
    - Encapsulation efficiency, when enabled, scales the initial releasable amount; it does not change the release mechanism.
    """)
    st.subheader("Simulation data export")
    st.download_button("Download simulated release data CSV", main_df.to_csv(index=False).encode("utf-8"),
                       file_name="phytorelease_simulated_release.csv", mime="text/csv")
    report_text = f"""PHYTORELEASE LAB — SIMULATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Phytochemical: {phytochemical}
Scaffold material: {material}
Initial loading: {loading:g} {unit}
Encapsulation efficiency applied: {'Yes' if use_encapsulation else 'No'}
Encapsulation efficiency: {efficiency:g}%
Releasable initial amount: {releasable:g} {unit}
Porosity (context only): {porosity:g}%
Diffusion coefficient (context only): {diffusion:.6g} m^2/s
Degradation parameter (context only): {degradation:g} 1/time
First-order release constant k: {k:g} 1/time
Duration: {duration:g} {time_unit}
Predicted released at end: {end_release:g} {unit} ({end_pct:.3g}%)
Estimated remaining at end: {end_remaining:g} {unit}

MODEL
M_t = M_0 * (1 - exp(-k*t))
M_remaining = M_0 - M_t

ASSUMPTIONS
This is a simplified first-order empirical model. Porosity, diffusion coefficient, and degradation parameter are contextual inputs and are not used to derive k. Parameter values require experimental or literature justification.

DISCLAIMER
These are computational estimates, not experimental measurements. They do not establish therapeutic efficacy, clinical safety, or real scaffold performance.
"""
    st.download_button("Download concise simulation report (TXT)", report_text.encode("utf-8"),
                       file_name="phytorelease_simulation_report.txt", mime="text/plain")
    st.markdown("---")
    st.success("Basic model checks: non-negative loading and k are enforced; release is bounded by releasable loading; remaining amount is non-negative; k = 0 produces no release; zero loading is handled safely.")
    st.caption("Version 1.0 • For research exploration and prototype demonstration only")
