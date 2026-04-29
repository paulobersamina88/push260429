
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Online MDOF RSA–Pushover PRO", layout="wide")

st.title("Online MDOF RSA–Pushover Reconciliation PRO")
st.caption("Streamlit online version: STAAD mass/stiffness → modal RSA → plastic moment capacity → nonlinear MDOF pushover → ADRS reconciliation")

def assemble_shear_K(k):
    n = len(k)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i == 0:
            K[i, i] += k[i]
        else:
            K[i, i] += k[i]
            K[i-1, i-1] += k[i]
            K[i, i-1] -= k[i]
            K[i-1, i] -= k[i]
    return K

def modal_analysis(W, k):
    g = 9.80665
    m = W / g
    M = np.diag(m)
    K = assemble_shear_K(k)
    A = np.linalg.inv(M) @ K
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argsort(np.real(eigvals))
    eigvals = np.real(eigvals[idx])
    eigvecs = np.real(eigvecs[:, idx])
    omega = np.sqrt(np.maximum(eigvals, 1e-12))
    T = 2*np.pi/omega

    phi = eigvecs.copy()
    for j in range(phi.shape[1]):
        if abs(phi[-1,j]) > 1e-12:
            phi[:,j] = phi[:,j] / phi[-1,j]
        if np.sum(phi[:,j]) < 0:
            phi[:,j] *= -1

    rows = []
    total_m = np.sum(m)
    one = np.ones(len(W))
    for j in range(len(W)):
        ph = phi[:,j]
        modal_mass = ph @ M @ ph
        gamma = (ph @ M @ one) / modal_mass
        eff_mass = gamma**2 * modal_mass
        rows.append({
            "Mode": j+1,
            "Omega_rad/s": omega[j],
            "Period_s": T[j],
            "Gamma": gamma,
            "Modal_Mass": modal_mass,
            "Effective_Modal_Mass": eff_mass,
            "Eff_Mass_Ratio_%": eff_mass/total_m*100,
        })
    df = pd.DataFrame(rows)
    df["Cumulative_%"] = df["Eff_Mass_Ratio_%"].cumsum()
    return df, phi, K, M

def ubc97_sa(T, Ca, Cv):
    T = np.asarray(T, dtype=float)
    T0 = 0.2 * Cv / (2.5*Ca) if Ca > 0 else 0.0
    Ts = Cv / (2.5*Ca) if Ca > 0 else 0.0
    plateau = 2.5 * Ca
    Sa = np.zeros_like(T)
    for i, t in enumerate(T):
        if t <= T0 and T0 > 0:
            Sa[i] = Ca + (plateau - Ca) * (t / T0)
        elif t <= Ts:
            Sa[i] = plateau
        else:
            Sa[i] = Cv / max(t, 1e-9)
    return Sa

def cqc_combine(values, periods, zeta=0.05):
    values = np.asarray(values, dtype=float)
    w = 2*np.pi/np.asarray(periods, dtype=float)
    n = len(values)
    total = 0.0
    for i in range(n):
        for j in range(n):
            beta = w[j]/w[i]
            rho = (8*zeta**2*(1+beta)*beta**1.5) / ((1-beta**2)**2 + 4*zeta**2*beta*(1+beta)**2)
            total += rho * values[i] * values[j]
    return np.sqrt(max(total, 0.0))

def multilinear_storey_drift(V, Vy, Vu, k, alpha, residual_ratio):
    if k <= 0:
        return 0.0, "Invalid stiffness"
    dy = Vy/k
    if V <= Vy:
        return V/k, "Elastic"
    du = dy + (Vu - Vy) / max(alpha*k, 1e-12)
    if V <= Vu:
        return dy + (V - Vy)/max(alpha*k, 1e-12), "Yielded"
    kres = max(alpha*k*residual_ratio, k*0.002)
    return du + (V - Vu)/kres, "Beyond ultimate"

def demand_spectrum(T, Ca, Cv, R=1.0, Ie=1.0):
    return ubc97_sa(np.array(T), Ca, Cv) * Ie / max(R, 1e-9)

def find_intersection(Sd_cap, Sa_cap, Sd_dem, Sa_dem):
    if len(Sd_cap) < 2:
        return None
    cap_sa_at = np.interp(Sd_dem, Sd_cap, Sa_cap, left=np.nan, right=np.nan)
    diff = cap_sa_at - Sa_dem
    valid = ~np.isnan(diff)
    if valid.sum() < 2:
        return None
    Sd_valid = Sd_dem[valid]
    Sa_valid = Sa_dem[valid]
    diff_valid = diff[valid]
    sign = np.where(np.sign(diff_valid[:-1]) != np.sign(diff_valid[1:]))[0]
    if len(sign) == 0:
        idx = int(np.nanargmin(np.abs(diff_valid)))
        return {"Sd_m": Sd_valid[idx], "Sa_g": Sa_valid[idx], "note": "closest approach only"}
    i = sign[0]
    x1, x2 = Sd_valid[i], Sd_valid[i+1]
    y1, y2 = diff_valid[i], diff_valid[i+1]
    x = x1 - y1*(x2-x1)/(y2-y1)
    sa = np.interp(x, Sd_dem, Sa_dem)
    return {"Sd_m": x, "Sa_g": sa, "note": "intersection"}

st.sidebar.header("Model Basis")
n = st.sidebar.number_input("Number of storeys", 1, 10, 3, 1)
frames_axis = st.sidebar.number_input("Number of similar frames in analyzed axis", 1, 30, 3, 1)
basis = st.sidebar.radio("Beam/column Mp input basis", ["Per frame (multiply by frames)", "Whole axis total (do not multiply)"], index=0)
frame_multiplier = frames_axis if basis.startswith("Per frame") else 1

alpha = st.sidebar.slider("Post-yield stiffness ratio α", 0.01, 0.25, 0.05, 0.01)
overstrength = st.sidebar.slider("Ultimate / yield shear ratio", 1.00, 2.00, 1.25, 0.05)
residual_ratio = st.sidebar.slider("Residual branch stiffness factor", 0.05, 1.00, 0.25, 0.05)

st.sidebar.header("UBC 97 Response Spectrum")
Ca = st.sidebar.number_input("Ca", value=0.44, step=0.01)
Cv = st.sidebar.number_input("Cv", value=0.768, step=0.01)
R = st.sidebar.number_input("R", value=8.5, step=0.5)
Ie = st.sidebar.number_input("I", value=1.0, step=0.1)
damping = st.sidebar.number_input("Damping ratio for CQC", value=0.05, min_value=0.01, max_value=0.20, step=0.01)

st.sidebar.header("Pushover Control")
max_base_factor = st.sidebar.slider("Plot pushover up to factor × first yield", 1.0, 8.0, 3.0, 0.5)
steps = st.sidebar.slider("Number of pushover steps", 30, 250, 100, 10)

default = []
default_W = [1509.12, 1283.42, 579.05] + [500]*(max(n-3,0))
default_k = [40090.0, 38510.0, 32400.0] + [25000]*(max(n-3,0))
for i in range(n):
    default.append({
        "Storey": i+1,
        "Floor weight W_i (kN)": float(default_W[i]),
        "Storey stiffness k_i (kN/m)": float(default_k[i]),
        "Height h_i (m)": 3.0,
        "Columns per frame": 3,
        "Bays per frame": 2,
        "Column Mp each (kN-m)": 300.0,
        "Beam Mp per end (kN-m)": 150.0,
        "Column participation factor": 1.0,
        "Beam participation factor": 1.0,
    })

tabs = st.tabs(["1 Input", "2 Modal RSA", "3 Yield Capacity", "4 Nonlinear Pushover", "5 ADRS", "6 Manual Calc", "7 Downloads"])

with tabs[0]:
    st.subheader("Input table")
    st.write("Paste/edit STAAD extracted floor weights and storey stiffness. Then input beam/column plastic moment per frame and number of frames per axis.")
    df = st.data_editor(pd.DataFrame(default), use_container_width=True, num_rows="fixed")
    st.warning("Basis rule: mass basis = stiffness basis = yield/plastic moment basis.")

W = df["Floor weight W_i (kN)"].to_numpy(float)
k = df["Storey stiffness k_i (kN/m)"].to_numpy(float)
h = df["Height h_i (m)"].to_numpy(float)
cols = df["Columns per frame"].to_numpy(float)
bays = df["Bays per frame"].to_numpy(float)
Mp_col = df["Column Mp each (kN-m)"].to_numpy(float)
Mp_beam = df["Beam Mp per end (kN-m)"].to_numpy(float)
col_part = df["Column participation factor"].to_numpy(float)
beam_part = df["Beam participation factor"].to_numpy(float)

modal_df, phi, K, M = modal_analysis(W, k)
T = modal_df["Period_s"].to_numpy(float)
Gamma = modal_df["Gamma"].to_numpy(float)

Sa_elastic = demand_spectrum(T, Ca, Cv, R=1.0, Ie=Ie)
Sa_design = demand_spectrum(T, Ca, Cv, R=R, Ie=Ie)
modal_eff_weight = modal_df["Effective_Modal_Mass"].to_numpy(float) * 9.80665
V_modal_elastic = Sa_elastic * modal_eff_weight
V_modal_design = Sa_design * modal_eff_weight
modal_df["Sa_elastic_g"] = Sa_elastic
modal_df["Sa_design_g"] = Sa_design
modal_df["Mode_Base_Shear_Elastic_kN"] = V_modal_elastic
modal_df["Mode_Base_Shear_Design_kN"] = V_modal_design
V_srss_design = float(np.sqrt(np.sum(V_modal_design**2)))
V_cqc_design = float(cqc_combine(V_modal_design, T, damping))

phi1 = phi[:,0]
raw = np.abs(W * phi1)
floor_force_ratio = raw / raw.sum()
storey_shear_ratio = np.array([floor_force_ratio[i:].sum() for i in range(n)])

Vy_col_pf = 2 * cols * Mp_col * col_part / h
Vy_beam_pf = bays * 2 * Mp_beam * beam_part / h
Vy_col_total = Vy_col_pf * frame_multiplier
Vy_beam_total = Vy_beam_pf * frame_multiplier
Vy = np.minimum(Vy_col_total, Vy_beam_total)
Vu = overstrength * Vy
dy = Vy / k * 1000

first_yield_base = np.min(Vy / np.maximum(storey_shear_ratio, 1e-12))
max_base = max_base_factor * first_yield_base
Vb = np.linspace(0, max_base, steps)
push_rows = []

for v in Vb:
    shears = storey_shear_ratio * v
    drifts = []
    states = []
    for i in range(n):
        d, state = multilinear_storey_drift(shears[i], Vy[i], Vu[i], k[i], alpha, residual_ratio)
        drifts.append(d)
        states.append(state)
    drifts = np.array(drifts)
    floor_disp = np.cumsum(drifts)
    row = {"Base shear kN": v, "Roof displacement mm": floor_disp[-1]*1000, "Global drift ratio %": floor_disp[-1]/np.sum(h)*100, "Yielded storeys": ", ".join([str(i+1) for i,s in enumerate(states) if s!="Elastic"]) or "None"}
    for i in range(n):
        row[f"S{i+1} shear kN"] = shears[i]
        row[f"S{i+1} drift mm"] = drifts[i]*1000
        row[f"F{i+1} displacement mm"] = floor_disp[i]*1000
        row[f"S{i+1} state"] = states[i]
    push_rows.append(row)

pushover_df = pd.DataFrame(push_rows)

yield_points = []
for i in range(n):
    for tag, cap in [("Yield", Vy[i]), ("Ultimate", Vu[i])]:
        vb = cap / max(storey_shear_ratio[i], 1e-12)
        shears = storey_shear_ratio * vb
        dr = np.array([multilinear_storey_drift(shears[j], Vy[j], Vu[j], k[j], alpha, residual_ratio)[0] for j in range(n)])
        yield_points.append({"Storey": i+1, "Point": tag, "Base shear kN": vb, "Roof displacement mm": dr.sum()*1000})

Gamma1 = Gamma[0]
phi_roof = phi1[-1]
roof_m = pushover_df["Roof displacement mm"].to_numpy()/1000
Sd = roof_m / max(abs(Gamma1*phi_roof), 1e-12)
W_eff1 = modal_df.loc[0, "Effective_Modal_Mass"] * 9.80665
Sa_cap = pushover_df["Base shear kN"].to_numpy() / W_eff1
pushover_df["Sd_m_ADRS"] = Sd
pushover_df["Sa_g_ADRS"] = Sa_cap

T_range = np.linspace(0.02, max(5, T[0]*4), 300)
Sa_dem_el = demand_spectrum(T_range, Ca, Cv, R=1.0, Ie=Ie)
Sa_dem_des = demand_spectrum(T_range, Ca, Cv, R=R, Ie=Ie)
Sd_dem_el = Sa_dem_el * 9.80665 * T_range**2/(4*np.pi**2)
Sd_dem_des = Sa_dem_des * 9.80665 * T_range**2/(4*np.pi**2)
perf_el = find_intersection(Sd, Sa_cap, Sd_dem_el, Sa_dem_el)
perf_des = find_intersection(Sd, Sa_cap, Sd_dem_des, Sa_dem_des)

with tabs[1]:
    st.subheader("Modal RSA")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("T1", f"{T[0]:.4f} s")
    c2.metric("Mode 1 mass", f"{modal_df.loc[0,'Eff_Mass_Ratio_%']:.2f}%")
    c3.metric("Design RSA SRSS", f"{V_srss_design:.2f} kN")
    c4.metric("Design RSA CQC", f"{V_cqc_design:.2f} kN")
    st.dataframe(modal_df.round(4), use_container_width=True)
    force_df = pd.DataFrame({"Floor": np.arange(1,n+1), "W_i kN": W, "phi1_i": phi1, "W_i*phi_i": raw, "Floor force ratio": floor_force_ratio, "Storey shear ratio": storey_shear_ratio})
    st.subheader("First mode pushover force pattern")
    st.dataframe(force_df.round(5), use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phi1, y=np.arange(1,n+1), mode="lines+markers", name="Mode 1"))
    fig.update_layout(title="Mode 1 shape normalized to roof = 1", xaxis_title="Mode shape", yaxis_title="Floor", height=420)
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Yield capacity from beam/column Mp")
    ydf = pd.DataFrame({"Storey": np.arange(1,n+1), "Column Vy per frame kN": Vy_col_pf, "Beam Vy per frame kN": Vy_beam_pf, "Frame multiplier": frame_multiplier, "Column Vy total kN": Vy_col_total, "Beam Vy total kN": Vy_beam_total, "Governing Vy kN": Vy, "Ultimate Vu kN": Vu, "Yield drift mm": dy, "Yield drift ratio %": (Vy/k)/h*100})
    st.dataframe(ydf.round(4), use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ydf["Storey"], y=ydf["Column Vy total kN"], name="Column-controlled"))
    fig.add_trace(go.Bar(x=ydf["Storey"], y=ydf["Beam Vy total kN"], name="Beam-controlled"))
    fig.add_trace(go.Scatter(x=ydf["Storey"], y=ydf["Governing Vy kN"], mode="lines+markers", name="Governing Vy"))
    fig.update_layout(title="Storey yield capacity comparison", xaxis_title="Storey", yaxis_title="Shear capacity kN", barmode="group", height=430)
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Nonlinear MDOF pushover")
    c1,c2,c3 = st.columns(3)
    c1.metric("First yield base shear", f"{first_yield_base:.2f} kN")
    c2.metric("RSA design CQC / first yield", f"{V_cqc_design/first_yield_base:.3f}")
    c3.metric("Max roof displacement plotted", f"{pushover_df['Roof displacement mm'].max():.1f} mm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pushover_df["Roof displacement mm"], y=pushover_df["Base shear kN"], mode="lines+markers", name="Pushover"))
    yp_df = pd.DataFrame(yield_points)
    for _, r in yp_df.iterrows():
        if r["Base shear kN"] <= max_base:
            fig.add_trace(go.Scatter(x=[r["Roof displacement mm"]], y=[r["Base shear kN"]], mode="markers+text", text=[f"S{int(r['Storey'])} {r['Point']}"], textposition="top center", name=f"S{int(r['Storey'])} {r['Point']}"))
    fig.add_hline(y=V_cqc_design, line_dash="dash", annotation_text="RSA design CQC")
    fig.update_layout(title="Base shear vs roof displacement", xaxis_title="Roof displacement (mm)", yaxis_title="Base shear (kN)", height=550)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pushover_df.round(4), use_container_width=True)

with tabs[4]:
    st.subheader("ADRS reconciliation")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Sd, y=Sa_cap, mode="lines+markers", name="Capacity spectrum"))
    fig.add_trace(go.Scatter(x=Sd_dem_el, y=Sa_dem_el, mode="lines", name="Elastic demand spectrum"))
    fig.add_trace(go.Scatter(x=Sd_dem_des, y=Sa_dem_des, mode="lines", name=f"Design demand spectrum /R={R}"))
    if perf_el:
        fig.add_trace(go.Scatter(x=[perf_el["Sd_m"]], y=[perf_el["Sa_g"]], mode="markers+text", text=["Elastic PP"], textposition="top center", name="Elastic performance point"))
    if perf_des:
        fig.add_trace(go.Scatter(x=[perf_des["Sd_m"]], y=[perf_des["Sa_g"]], mode="markers+text", text=["Design PP"], textposition="bottom center", name="Design performance point"))
    fig.update_layout(title="ADRS capacity vs demand spectrum", xaxis_title="Sd (m)", yaxis_title="Sa (g)", height=560)
    st.plotly_chart(fig, use_container_width=True)
    perf_rows = []
    for name, p in [("Elastic demand", perf_el), (f"Design demand R={R}", perf_des)]:
        if p:
            roof = p["Sd_m"] * Gamma1 * phi_roof * 1000
            perf_rows.append({"Case": name, "Sd m": p["Sd_m"], "Sa g": p["Sa_g"], "Roof displacement mm": roof, "Result type": p["note"]})
    if perf_rows:
        st.dataframe(pd.DataFrame(perf_rows).round(5), use_container_width=True)

with tabs[5]:
    st.subheader("Manual calculation sheet")
    st.markdown(r"""
**First-mode force pattern**

\[
F_i=V_b \frac{W_i\phi_i}{\sum W_i\phi_i}
\]

**Column mechanism**

\[
V_{y,col}=\frac{2\sum M_{p,col}}{h}
\]

**Beam mechanism**

\[
V_{y,beam}=\frac{\sum M_{p,beam}}{h}
\]

**Bilinear storey drift**

\[
\delta=\frac{V}{k}
\]

\[
\delta=\delta_y+\frac{V-V_y}{\alpha k}
\]

**ADRS**

\[
S_d=\frac{\Delta_{roof}}{\Gamma_1\phi_{roof}}
\]

\[
S_a=\frac{V_b}{W_{eff,1}}
\]
""")
    example_v = st.number_input("Select base shear for row check (kN)", value=float(first_yield_base), step=10.0)
    ex_shear = storey_shear_ratio * example_v
    ex_rows = []
    for i in range(n):
        d, s = multilinear_storey_drift(ex_shear[i], Vy[i], Vu[i], k[i], alpha, residual_ratio)
        ex_rows.append({"Storey": i+1, "Storey shear ratio": storey_shear_ratio[i], "Storey shear kN": ex_shear[i], "Vy kN": Vy[i], "Vu kN": Vu[i], "k kN/m": k[i], "drift mm": d*1000, "state": s})
    st.dataframe(pd.DataFrame(ex_rows).round(5), use_container_width=True)
    st.metric("Roof displacement for selected base shear", f"{sum([r['drift mm'] for r in ex_rows]):.3f} mm")

with tabs[6]:
    st.subheader("Download results")
    ydf_down = pd.DataFrame({"Storey": np.arange(1,n+1), "Vy_col_total": Vy_col_total, "Vy_beam_total": Vy_beam_total, "Vy_governing": Vy, "Vu": Vu, "Yield_drift_mm": dy})
    st.download_button("Download modal RSA CSV", modal_df.to_csv(index=False).encode(), "modal_rsa.csv", "text/csv")
    st.download_button("Download yield capacity CSV", ydf_down.to_csv(index=False).encode(), "yield_capacity.csv", "text/csv")
    st.download_button("Download pushover curve CSV", pushover_df.to_csv(index=False).encode(), "pushover_curve.csv", "text/csv")
