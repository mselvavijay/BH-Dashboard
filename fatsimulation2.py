import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import utils

# ---- Page / Theme ----
st.set_page_config(page_title="Turbomachinery Pre-FAT — Trip, Cooldown, Assistant", layout="wide", page_icon="⚙️")

# Inject Shared CSS
utils.inject_css()

BG, TEXT, CARD = utils.BG, utils.TEXT, utils.CARD
PRIMARY, ALERT, AMBER, OK = utils.PRIMARY, utils.ALERT, utils.AMBER, utils.OK
LINE = utils.LINE

# ---- Constants shared by both tabs ----
PARAMS = ["ThrustBearingTemp", "LubeOilPressure", "Vibration", "SealFlow"]
UNITS  = {"ThrustBearingTemp":"°C","LubeOilPressure":"bar","Vibration":"mm/s","SealFlow":"L/s"}
LIMITS = {"ThrustBearingTemp":95.0,"LubeOilPressure":3.6,"Vibration":9.0,"SealFlow":0.70}  # SealFlow is low-limit
START  = {"ThrustBearingTemp":70.0,"LubeOilPressure":2.20,"Vibration":4.50,"SealFlow":1.20}
JITTER = {"ThrustBearingTemp":0.20,"LubeOilPressure":0.05,"Vibration":0.20,"SealFlow":0.05}
NOISE  = {"ThrustBearingTemp":0.02,"LubeOilPressure":0.01,"Vibration":0.03,"SealFlow":0.01}
SPIKE  = {"ThrustBearingTemp":+0.80,"LubeOilPressure":+0.15,"Vibration":+0.70,"SealFlow":-0.06}
COUP   = dict(vib_from_temp=0.02, seal_from_press=-0.03, temp_from_seal=-0.10)

# ---- Session defaults used by both ----
ss = st.session_state
ss.setdefault("limits", LIMITS.copy())
ss.setdefault("confirm_s", 3.0)
ss.setdefault("cooldown_s", 30.0)
ss.setdefault("warn_band_pct", 5.0)

# ---- Tab switch (radio renders like tabs & lets us control auto-refresh) ----
tab = st.radio("Mode", ["Simulation", "CSV Analysis"], horizontal=True)

# ======================================================================
#                               SIMULATION
# ======================================================================
if tab == "Simulation":
    # --- HEADER & AUTO-REFRESH (simulation only) ---
    st.markdown("### Turbomachinery Pre‑FAT — Trip • Cooldown • Assistant")
    st_autorefresh(interval=1000, key="ui_refresh_sim")

    # --- (BEGIN) YOUR ORIGINAL SIMULATION CODE BLOCK: PRESERVED 100% ---
    def init():
        ss.setdefault("cur", START.copy())
        ss.setdefault("sim_running", False)
        ss.setdefault("state", "RUN")  # RUN | COOLDOWN | STOPPED
        ss.setdefault("trip_latched", False)
        ss.setdefault("trip_eval_s", 0.0)
        ss.setdefault("cooldown_left_s", 0.0)
        ss.setdefault("ambient_temp", 40.0)
        ss.setdefault("post_trip_seal", 0.30)
        ss.setdefault("sealflow_target_run", 1.10)
        ss.setdefault("sealflow_ctrl_gain", 0.15)
        ss.setdefault("fault_active", False)
        ss.setdefault("fault_left_s", 0.0)
        ss.setdefault("min_dt", 0.8)
        ss.setdefault("max_dt", 1.4)
        ss.setdefault("trip_n", 2)
        ss.setdefault("sim_t", 0.0)
        ss.setdefault("log", pd.DataFrame(columns=["t","clock",*PARAMS,"exceed","status","machine"]))
        ss.setdefault("hist", {p: deque(maxlen=2400) for p in PARAMS})
        ss.setdefault("last_trip_info", None)
        ss.setdefault("chat", [])
        ss.setdefault("bias_rate", {
            "ThrustBearingTemp": 0.00,
            "LubeOilPressure":   0.00,
            "Vibration":         0.00,
            "SealFlow":          0.00,
        })
    init()

    with st.sidebar:
        st.markdown("### Simulation Controls")
        c1,c2,c3 = st.columns(3)
        if c1.button("▶ Start", use_container_width=True):
            ss.sim_running = True
        if c2.button("⏸ Stop", use_container_width=True):
            ss.sim_running = False
        if c3.button("🧰 Reset", use_container_width=True):
            ss.update({
                "cur": START.copy(),"state":"RUN","trip_latched":False,"trip_eval_s":0.0,
                "cooldown_left_s":0.0,"fault_active":False,"fault_left_s":0.0,"sim_t":0.0,
                "log":pd.DataFrame(columns=ss.log.columns),
                "hist":{p: deque(maxlen=2400) for p in PARAMS},
                "chat":[]
            })

        with st.expander("⚙️ Limits", expanded=False):
            ss.limits["ThrustBearingTemp"] = st.slider("Temp High (°C)",   80.0,120.0, ss.limits["ThrustBearingTemp"], 0.5)
            ss.limits["LubeOilPressure"]   = st.slider("Press High (bar)",  3.0,  5.0, ss.limits["LubeOilPressure"],   0.05)
            ss.limits["Vibration"]         = st.slider("Vibration High (mm/s)",6.0,15.0, ss.limits["Vibration"], 0.1)
            ss.limits["SealFlow"]          = st.slider("Seal Flow Low (L/s)",  0.4, 1.0, ss.limits["SealFlow"], 0.01)

        with st.expander("🛑 Trip & Cooldown", expanded=False):
            ss.trip_n      = st.slider("Trip when ≥ N exceed", 1, 4, ss.trip_n, 1)
            ss.confirm_s   = st.slider("Trip Confirm (s)", 1.0, 10.0, ss.confirm_s, 0.5)
            ss.cooldown_s  = st.slider("Cooldown Duration (s)", 5, 120, int(ss.cooldown_s), 5)
            ss.ambient_temp= st.slider("Ambient Temp Target (°C)", 25.0, 60.0, float(ss.get("ambient_temp",40.0)), 0.5)
            ss.post_trip_seal = st.slider("Post‑Trip Seal Flow Target (L/s)", 0.0, 1.0, float(ss.get("post_trip_seal",0.30)), 0.01)

        with st.expander("💧 Seal Flow Regulation (RUN)", expanded=False):
            ss.sealflow_target_run = st.slider("Seal Flow RUN Target (L/s)", 0.8, 1.5, float(ss.sealflow_target_run), 0.01)
            ss.sealflow_ctrl_gain  = st.slider("Seal Flow Control Gain",     0.00,0.50, float(ss.sealflow_ctrl_gain), 0.01)

        with st.expander("📉 Bias (Gain) — Per‑second Ramp", expanded=False):
            ss.bias_rate["ThrustBearingTemp"] = st.slider("Temp Bias (°C/s)",       -1.0, 1.0, float(ss.bias_rate["ThrustBearingTemp"]), 0.01)
            ss.bias_rate["LubeOilPressure"]   = st.slider("Pressure Bias (bar/s)",  -0.5, 0.5, float(ss.bias_rate["LubeOilPressure"]),   0.01)
            ss.bias_rate["Vibration"]         = st.slider("Vibration Bias (mm/s/s)",-1.0, 1.0, float(ss.bias_rate["Vibration"]),         0.01)
            ss.bias_rate["SealFlow"]          = st.slider("Seal Flow Bias (L/s/s)", -0.2, 0.2, float(ss.bias_rate["SealFlow"]),          0.005)

        with st.expander("⚡ Timing & Spike", expanded=True):
            ss.min_dt = st.slider("Sim min dt (s)", 0.5, 2.0, float(ss.min_dt), 0.1)
            ss.max_dt = st.slider("Sim max dt (s)", 0.8, 3.0, float(ss.max_dt), 0.1)
            fault_dur = st.slider("Spike Duration (s)", 2.0, 60.0, 20.0, 1.0)
            d1,d2 = st.columns(2)
            if d1.button("🚨 Trigger Spike", use_container_width=True):
                ss.fault_active = True; ss.fault_left_s = float(fault_dur)
            if d2.button("⏹ Stop Spike", use_container_width=True):
                ss.fault_active = False; ss.fault_left_s = 0.0

    def _dt(a,b): return float(np.random.uniform(a,b))
    def _exceed(p,val,lim): return (val <= lim) if p=="SealFlow" else (val >= lim)

    def _simulate_run(dt):
        prev = ss.cur.copy()
        for p in PARAMS:
            jitter = np.random.uniform(-JITTER[p], JITTER[p]) * dt
            noise  = np.random.normal(0.0, NOISE[p] * (dt**0.5))
            bias   = ss.bias_rate[p] * dt
            spike  = (SPIKE[p]*dt) if (ss.fault_active and ss.fault_left_s>0) else 0.0
            ss.cur[p] += bias + jitter + noise + spike
        ss.cur["Vibration"] += COUP["vib_from_temp"]*(ss.cur["ThrustBearingTemp"]-prev["ThrustBearingTemp"])*dt
        ss.cur["SealFlow"]  += COUP["seal_from_press"]*(ss.cur["LubeOilPressure"]-prev["LubeOilPressure"])*dt
        seal_delta = ss.cur["SealFlow"] - prev["SealFlow"]
        ss.cur["ThrustBearingTemp"] += COUP["temp_from_seal"]*seal_delta*dt
        ss.cur["SealFlow"] = max(0.0, ss.cur["SealFlow"])
        if ss.fault_active:
            ss.fault_left_s -= dt
            if ss.fault_left_s <= 0: ss.fault_active=False; ss.fault_left_s=0.0
        if (ss.state=="RUN") and (not ss.fault_active) and (not ss.trip_latched):
            err = ss.sealflow_target_run - ss.cur["SealFlow"]
            ss.cur["SealFlow"] += ss.sealflow_ctrl_gain * err * dt

    def _simulate_cooldown(dt):
        targets = {"ThrustBearingTemp": ss.ambient_temp, "LubeOilPressure": 0.0, "Vibration": 0.0, "SealFlow": ss.post_trip_seal}
        taus    = {"ThrustBearingTemp": 20.0, "LubeOilPressure": 6.0, "Vibration": 3.0, "SealFlow": 4.0}
        for p in PARAMS:
            x, tgt, tau = ss.cur[p], targets[p], max(0.5, taus[p])
            ss.cur[p] = x + (tgt-x)*min(1.0, dt/tau)
        ss.cur["SealFlow"] = max(0.0, ss.cur["SealFlow"])
        ss.cur["Vibration"] = max(0.0, ss.cur["Vibration"])

    def step_once():
        if not ss.sim_running: return 0.0, False
        dt = _dt(ss.min_dt, ss.max_dt); ss.sim_t += dt
        if ss.state=="RUN": _simulate_run(dt)
        elif ss.state=="COOLDOWN":
            _simulate_cooldown(dt); ss.cooldown_left_s -= dt
            if ss.cooldown_left_s <= 0: ss.state="STOPPED"
        return dt, True

    def eval_trip(dt):
        vals = ss.cur
        ex = {p:_exceed(p,vals[p],ss.limits[p]) for p in PARAMS}
        ex_count = sum(1 for v in ex.values() if v)
        if ss.trip_latched: return ex_count, "TRIPPED"
        if dt>0 and ex_count>=ss.trip_n: ss.trip_eval_s += dt
        else: ss.trip_eval_s = 0.0
        status = "WARN" if ex_count>=1 else "NORMAL"
        if ss.trip_eval_s >= ss.confirm_s and ex_count>=ss.trip_n:
            ss.trip_latched=True; ss.state="COOLDOWN"; ss.cooldown_left_s=float(ss.cooldown_s)
            ss.fault_active=False; ss.fault_left_s=0.0
            ss.last_trip_info={"time":datetime.now().strftime("%H:%M:%S"),
                               "exceeded":{p:round(vals[p],2) for p,ok in ex.items() if ok},
                               "exceed_count":ex_count,"trip_n":ss.trip_n}
            status="TRIPPED"
        return ex_count, status

    def log_row(status, ex_count):
        row={"t":ss.sim_t,"clock":datetime.now().strftime("%H:%M:%S"),
             **{p:round(ss.cur[p],2) for p in PARAMS},"exceed":ex_count,"status":status,"machine":ss.state}
        ss.log.loc[len(ss.log)] = row
        if len(ss.log)>3000: ss.log = ss.log.tail(1500).reset_index(drop=True)
        for p in PARAMS: ss.hist[p].append((ss.sim_t, float(ss.cur[p])))

    dt, stepped = step_once()
    ex_count, status = eval_trip(dt if stepped else 0.0)
    if stepped: log_row(status, ex_count)

    ALM_ANY = 1 if status in ("WARN","TRIPPED") else 0

    bcol = OK if status=="NORMAL" else AMBER if status=="WARN" else ALERT
    st.markdown(
        f"""
<div style="padding:8px;border-radius:6px;background:{CARD};color:{TEXT};border-left:6px solid {bcol}">
<b>Sim:</b> {'RUNNING' if ss.sim_running else 'PAUSED'} &nbsp; | &nbsp;
<b>State:</b> {ss.state} &nbsp; | &nbsp; <b>Status:</b> {status} &nbsp; | &nbsp;
<b>Exceed:</b> {ex_count} ≥ {ss.trip_n} &nbsp; | &nbsp; 
<b>Confirm:</b> {ss.trip_eval_s:.1f}/{ss.confirm_s:.1f}s
</div>
""",
        unsafe_allow_html=True
    )

    k1,k2,k3,k4 = st.columns(4)
    def kpi(col, label, key):
        v = round(ss.cur[key], 2)
        lim = ss.limits[key]
        unit = UNITS[key]
        
        if key == "SealFlow":
            is_alert = v <= lim
            limit_type = "Low"
        else:
            is_alert = v >= lim
            limit_type = "High"
            
        utils.render_kpi_card(col, label, v, unit, lim, is_alert, limit_type)

    kpi(k1,"Thrust Bearing Temp","ThrustBearingTemp")
    kpi(k2,"Lube Oil Pressure","LubeOilPressure")
    kpi(k3,"Vibration","Vibration")
    kpi(k4,"Seal Flow","SealFlow")

    st.markdown("### 📈 Compact Trends")
    def series_all(param):
        hist=ss.hist[param]
        if not hist: return [],[]
        xs=[t for (t,_) in hist]; ys=[v for (_,v) in hist]; return xs,ys
    fig = make_subplots(rows=2, cols=2, shared_xaxes=False,
                        horizontal_spacing=0.07, vertical_spacing=0.12,
                        subplot_titles=("ThrustBearingTemp","LubeOilPressure","Vibration","SealFlow"))
    cells=[("ThrustBearingTemp",1,1),("LubeOilPressure",1,2),("Vibration",2,1),("SealFlow",2,2)]
    for p,r,c in cells:
        xs,ys = series_all(p); lim = ss.limits[p]
        fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines',line=dict(width=2, color=PRIMARY),showlegend=False),row=r,col=c)
        fig.add_trace(go.Scatter(x=xs,y=[lim]*len(xs),mode='lines',line=dict(dash='dash',color=ALERT, width=1),showlegend=False),row=r,col=c)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=420,
        margin=dict(l=10,r=10,t=30,b=10),
        font=dict(family="Inter, sans-serif", color=TEXT)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#30363D')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#30363D')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🤖 Explainable Assistant")
    
    # Chat styling is now in utils.inject_css()
    
    def chat_bubble(text, role="bot"):
        utils.render_chat_bubble(text, role)
    def last20_avg(p):
        df=ss.log
        if df.empty or p not in df.columns: return None
        return round(df.tail(20)[p].astype(float).mean(),2)
    def approaching_critical_now(pct=5.0):
        cur=ss.cur; lims=ss.limits; out={}
        for p in PARAMS:
            v=float(cur[p]); lim=float(lims[p])
            if p=="SealFlow":
                if v>lim and v<=lim+lim*(pct/100.0): out[p]={"v":v,"lim":lim,"dir":"low","pct":round((v-lim)/lim*100,1)}
            else:
                if v>=lim*(1-pct/100.0) and v<lim: out[p]={"v":v,"lim":lim,"dir":"high","pct":round((lim-v)/lim*100,1)}
        return out
    cA,cB,cC,cD = st.columns(4)
    b_temp=cA.button("🌡️ Avg Temp (20)"); b_press=cB.button("🟢 Avg Pressure (20)")
    b_vib =cC.button("📳 Avg Vibration (20)"); b_seal=cD.button("💧 Avg Seal Flow (20)")
    cE,cF = st.columns(2)
    b_near=cE.button("⚠️ Approaching Critical"); b_cause=cF.button("🧠 Trip Cause", disabled=not bool(ss.trip_latched))
    if "chat" not in ss: ss.chat=[]
    def add(u,b): ss.chat.extend([("you",u),("bot",b)])
    if b_temp: a=last20_avg("ThrustBearingTemp"); add("Temperature average?", "No data yet." if a is None else f"Avg Temperature (20): {a} {UNITS['ThrustBearingTemp']}.")
    if b_press:a=last20_avg("LubeOilPressure");   add("Pressure average?"   , "No data yet." if a is None else f"Avg Pressure (20): {a} {UNITS['LubeOilPressure']}.")
    if b_vib:  a=last20_avg("Vibration");         add("Vibration average?"  , "No data yet." if a is None else f"Avg Vibration (20): {a} {UNITS['Vibration']}.")
    if b_seal: a=last20_avg("SealFlow");          add("Seal flow average?"  , "No data yet." if a is None else f"Avg Seal Flow (20): {a} {UNITS['SealFlow']}.")
    if b_near:
        near=approaching_critical_now(pct=ss.warn_band_pct)
        if not near: add("What’s approaching critical?","No parameters near limits.")
        else:
            parts=[f"{p}: {d['v']} vs {d['lim']} ({'↑' if d['dir']=='high' else '↓'} {d['pct']}% to limit)" for p,d in near.items()]
            add("What’s approaching critical?","; ".join(parts)+".")
    if b_cause:
        info=ss.last_trip_info
        if info and ss.trip_latched:
            parts=", ".join([f"{k} {v}" for k,v in info["exceeded"].items()])
            add("What caused the trip?", f"Trip cause: {parts} (≥{info['trip_n']} exceeded) at {info['time']}.")
        else:
            add("What caused the trip?", "Trip cause is available only after a machine trip.")
    for role,msg in ss.chat[-8:]: chat_bubble(msg, "bot" if role=="bot" else "you")

    st.markdown("### 🧪 Digital Outputs")
    d1,d2,d3 = st.columns(3)
    for i,(k,v) in enumerate({"DO_TRIP_LATCH":int(ss.trip_latched),"ALM_ANY":int(ALM_ANY),"FAULT_ACTIVE":int(ss.fault_active)}.items()):
        (d1,d2,d3)[i].markdown(f"**{k}**  \nState: {v}")

    st.markdown("### 🧾 System Log")
    st.dataframe(ss.log.tail(15), use_container_width=True)
    st.download_button("⬇️ Download Log (CSV)", ss.log.to_csv(index=False).encode("utf-8"), file_name="fat_log.csv", mime="text/csv")

    # --- PLC export (Simulation) ---
    def plc_st(n, limits, confirm_s):
        return f"""(* Auto-generated *)
VAR_INPUT
    ThrustBearingTemp : REAL; LubeOilPressure : REAL; Vibration : REAL; SealFlow : REAL;
END_VAR
VAR
    TripLatch : BOOL; tConfirm : TON;
END_VAR
ALM_TBT := ThrustBearingTemp >= {limits['ThrustBearingTemp']:.3f};
ALM_LOP := LubeOilPressure   >= {limits['LubeOilPressure']:.3f};
ALM_VIB := Vibration         >= {limits['Vibration']:.3f};
ALM_SEAL:= SealFlow          <= {limits['SealFlow']:.3f};
ALM_ANY := ALM_TBT OR ALM_LOP OR ALM_VIB OR ALM_SEAL;
EXCEED_N := (BOOL_TO_INT(ALM_TBT)+BOOL_TO_INT(ALM_LOP)+BOOL_TO_INT(ALM_VIB)+BOOL_TO_INT(ALM_SEAL)) >= {n};
tConfirm(IN := EXCEED_N, PT := T#{int(round(confirm_s))}S);
IF tConfirm.Q THEN TripLatch := TRUE; END_IF;
"""
    st.markdown("### 🧩 Export PLC Logic")
    st.download_button("⬇️ IEC 61131-3 ST (Simulation)", plc_st(ss.trip_n, ss.limits, ss.confirm_s), file_name="plc_simulation.st")
    st.download_button("⬇️ TXT Spec (Simulation)",
                       f"Trip when ≥{ss.trip_n} exceeds for {ss.confirm_s:.1f}s; limits: {ss.limits}", file_name="plc_simulation.txt")

    # --- (END) YOUR ORIGINAL SIMULATION CODE BLOCK ---

# ======================================================================
#                              CSV ANALYSIS
# ======================================================================
else:
    st.markdown("### CSV Analysis — Live Replay from File")

    # ---- Upload (no fallback) ----
    up = st.file_uploader("Upload CSV", type=["csv"], key="csv_up")
    if up is None:
        st.info("Upload a CSV to start live replay. No demo and no event log will be shown until a file is uploaded.")
        st.stop()

    df = pd.read_csv(up)
    cols = list(df.columns)

    # ---- Column mapping ----
    c1,c2,c3,c4,c5 = st.columns(5)
    time_col = c1.selectbox("Time", options=["<none>"]+cols,
                            index=(["<none>"]+cols).index("t") if "t" in cols else 0, key="map_time")
    map_cols = {
        "ThrustBearingTemp": c2.selectbox("ThrustBearingTemp", options=cols,
                                          index=cols.index("ThrustBearingTemp") if "ThrustBearingTemp" in cols else 0, key="map_tbt"),
        "LubeOilPressure":   c3.selectbox("LubeOilPressure",   options=cols,
                                          index=cols.index("LubeOilPressure")   if "LubeOilPressure"   in cols else 0, key="map_lop"),
        "Vibration":         c4.selectbox("Vibration",         options=cols,
                                          index=cols.index("Vibration")         if "Vibration"         in cols else 0, key="map_vib"),
        "SealFlow":          c5.selectbox("SealFlow",          options=cols,
                                          index=cols.index("SealFlow")          if "SealFlow"          in cols else 0, key="map_seal"),
    }

    rule_n = st.radio("Trip Rule (≥N exceeds)", [1,2], index=1, horizontal=True, key="csv_rule")
    limits = ss.limits  # reuse global limits
    confirm_s  = float(ss.confirm_s)
    cooldown_s = float(ss.cooldown_s)

    # ---- Player state ----
    if "csv_init_token" not in ss: ss.csv_init_token = None
    token_now = (up.name, tuple(map_cols.items()), time_col)

    def reset_csv_player():
        ss.csv_idx = 0
        ss.csv_playing = False
        ss.csv_state = "RUN"
        ss.csv_trip = False
        ss.csv_eval_s = 0.0
        ss.csv_cool_left = 0.0
        ss.csv_log = pd.DataFrame(columns=["t", *PARAMS, "exceed", "status", "machine"])

    if ss.csv_init_token != token_now:
        reset_csv_player()
        ss.csv_init_token = token_now

    cA,cB,cC,cD = st.columns(4)
    if cA.button("▶ Play", use_container_width=True):  ss.csv_playing = True
    if cB.button("⏸ Pause", use_container_width=True): ss.csv_playing = False
    if cC.button("⟲ Reset", use_container_width=True): reset_csv_player()
    rate = cD.slider("Rows/refresh", 1, 10, 1)

    # ---- Time vector ----
    if time_col != "<none>" and time_col in df.columns:
        tvec = df[time_col].astype(float).values
    else:
        tvec = np.arange(len(df), dtype=float)

    # ---- Trip check ----
    def _exceed(p, val, lim): return (val <= lim) if p == "SealFlow" else (val >= lim)

    # ---- Autorefresh only when playing ----
    if ss.csv_playing:
        st_autorefresh(interval=1000, key="ui_refresh_csv")

    # ---- Step N rows each refresh ----
    steps = rate if ss.csv_playing else 0
    for _ in range(steps):
        i = ss.csv_idx
        if i >= len(df):
            ss.csv_playing = False
            break

        vals = {p: float(df[map_cols[p]].iloc[i]) for p in PARAMS}
        dt = 0.0 if i == 0 else float(max(0.0, tvec[i] - tvec[i-1]))

        ex = {p: _exceed(p, vals[p], limits[p]) for p in PARAMS}
        ex_count = sum(1 for v in ex.values() if v)

        # state machine
        status = "NORMAL"
        if ss.csv_trip:
            status = "TRIPPED"
            if ss.csv_state == "COOLDOWN":
                ss.csv_cool_left -= dt
                if ss.csv_cool_left <= 0: ss.csv_state = "STOPPED"
        else:
            if ex_count >= int(st.session_state.csv_rule):
                ss.csv_eval_s += dt
                status = "ALARM" if ss.csv_eval_s < confirm_s else "TRIPPED"
                if status == "TRIPPED":
                    ss.csv_trip = True
                    ss.csv_state = "COOLDOWN"
                    ss.csv_cool_left = cooldown_s
            else:
                ss.csv_eval_s = 0.0
                status = "WARNING" if ex_count >= 1 else "NORMAL"

        row = {"t": float(tvec[i]), **vals, "exceed": ex_count, "status": status, "machine": ss.csv_state}
        ss.csv_log.loc[len(ss.csv_log)] = row
        ss.csv_idx += 1

    # ---- KPI banner (from latest) ----
    if not ss.csv_log.empty:
        last = ss.csv_log.iloc[-1]
        ex_count = int(last["exceed"])
        status = str(last["status"])
        bcol = OK if status=="NORMAL" else AMBER if status in ("WARNING","ALARM") else ALERT
        st.markdown(
            f"""
<div style="padding:8px;border-radius:6px;background:{CARD};color:{TEXT};border-left:6px solid {bcol}">
<b>Replay:</b> {'PLAYING' if ss.csv_playing else 'PAUSED'} &nbsp; | &nbsp;
<b>Row:</b> {ss.csv_idx}/{len(df)} &nbsp; | &nbsp; <b>Status:</b> {status} &nbsp; | &nbsp;
<b>Exceed:</b> {ex_count} ≥ {int(st.session_state.csv_rule)} &nbsp; | &nbsp; 
<b>Confirm:</b> {ss.csv_eval_s:.1f}/{confirm_s:.1f}s &nbsp; | &nbsp; <b>Machine:</b> {ss.csv_state}
</div>
""",
            unsafe_allow_html=True
        )

    # ---- 4 compact Plotly charts (grow as rows stream) ----
    if not ss.csv_log.empty:
        st.markdown("### 📈 Compact Trends (CSV Live)")
        fig2 = make_subplots(rows=2, cols=2, shared_xaxes=False,
                             horizontal_spacing=0.07, vertical_spacing=0.12,
                             subplot_titles=("ThrustBearingTemp","LubeOilPressure","Vibration","SealFlow"))
        cells=[("ThrustBearingTemp",1,1),("LubeOilPressure",1,2),("Vibration",2,1),("SealFlow",2,2)]
        for p,r,c in cells:
            xs=ss.csv_log["t"]; ys=ss.csv_log[p]; lim = limits[p]
            fig2.add_trace(go.Scatter(x=xs,y=ys,mode='lines',line=dict(width=2, color=PRIMARY),showlegend=False),row=r,col=c)
            fig2.add_trace(go.Scatter(x=xs,y=[lim]*len(xs),mode='lines',line=dict(dash='dash',color=ALERT, width=1),showlegend=False),row=r,col=c)
        
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=420,
            margin=dict(l=10,r=10,t=30,b=10),
            font=dict(family="Inter, sans-serif", color=TEXT)
        )
        fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#30363D')
        fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#30363D')
        st.plotly_chart(fig2, use_container_width=True)

        # ---- Event log (now that a file is uploaded) ----
        st.markdown("### 🧾 CSV Event Log (Live)")
        st.dataframe(ss.csv_log.tail(30), use_container_width=True)

        # ---- CSV Intelligence (answers based on processed rows so far) ----
        st.markdown("#### 🤖 CSV Intelligence (Live)")
        cI,cJ,cK = st.columns(3)
        if cI.button("📊 Summary (so far)"):
            counts = ss.csv_log["status"].value_counts().to_dict()
            st.success(f"States (so far): {counts}. Limits: {limits}. Rule: ≥{int(st.session_state.csv_rule)}, "
                       f"Confirm: {confirm_s:.1f}s, Cooldown: {cooldown_s:.1f}s.")
        if cJ.button("🧠 Trip Cause"):
            trip_rows = ss.csv_log[ss.csv_log["status"]=="TRIPPED"]
            if trip_rows.empty:
                st.info("No TRIP yet in the processed rows.")
            else:
                first_idx = trip_rows.index[0]
                row = ss.csv_log.loc[first_idx]
                over = []
                for p in PARAMS:
                    if _exceed(p, row[p], limits[p]): over.append(f"{p} {row[p]:.2f}")
                st.warning(f"Trip latched at t={row['t']:.2f}s; exceeded: {', '.join(over)} "
                           f"(≥{int(st.session_state.csv_rule)}; confirm {confirm_s:.1f}s).")
        if cK.button("🔥 Worst Actors"):
            sub = ss.csv_log.copy()
            if not sub.empty:
                frac = {}
                for p in PARAMS:
                    if p=="SealFlow": frac[p] = float((sub[p] <= limits[p]).mean())
                    else:            frac[p] = float((sub[p] >= limits[p]).mean())
                worst = sorted(frac.items(), key=lambda x: x[1], reverse=True)
                st.info("Top contributors (so far): " + ", ".join([f"{k} ({v:.0%})" for k,v in worst]))
