import streamlit as st
import pandas as pd
import utils

# ---- Page / Theme ----
st.set_page_config(page_title="Spec-Driven Verification Tool", layout="wide", page_icon="🛠️")

# Inject Shared CSS
utils.inject_css()

# ---- Session State ----
ss = st.session_state
ss.setdefault("spec", None)
ss.setdefault("io_df", None)
ss.setdefault("project_scope", "Compressor")

# ---- Logic ----
def validate_and_build_spec(df):
    """
    Validates the uploaded dataframe and builds a JSON-like spec object.
    Returns: (spec, error_msg)
    """
    required_cols = ["Tag", "Limit_High", "Limit_Low", "Unit"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        return None, f"Missing required columns: {', '.join(missing)}"
    
    # Check for duplicate tags
    if df["Tag"].duplicated().any():
        dups = df[df["Tag"].duplicated()]["Tag"].tolist()
        return None, f"Duplicate tags found: {', '.join(dups)}"
    
    # Build Spec
    spec = {
        "meta": {
            "scope": ss.project_scope,
            "generated_at": pd.Timestamp.now().isoformat(),
            "tag_count": len(df)
        },
        "tags": {}
    }
    
    try:
        for _, row in df.iterrows():
            tag = str(row["Tag"]).strip()
            spec["tags"][tag] = {
                "description": row.get("Description", tag),
                "unit": row.get("Unit", ""),
                "limit_high": float(row["Limit_High"]),
                "limit_low": float(row["Limit_Low"]),
                # Optional fields with defaults
                "min": float(row.get("Min", 0.0)),
                "max": float(row.get("Max", 100.0)),
                "trip_delay_s": float(row.get("Trip_Delay_s", 3.0))
            }
    except ValueError as e:
        return None, f"Data type error (check numeric columns): {e}"
        
    return spec, None

# ---- Sidebar ----
with st.sidebar:
    st.title("🛠️ Project Setup")
    ss.project_scope = st.selectbox("Scope / Subsystem", ["Compressor", "Pump", "Turbine", "Generator"])
    
    st.divider()
    
    st.markdown("### 1. Upload I/O List")
    uploaded_file = st.file_uploader("Upload Spec (CSV/XLSX)", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate immediately
            spec, err = validate_and_build_spec(df)
            
            if err:
                st.error(err)
                ss.io_df = None
                ss.spec = None
            else:
                ss.io_df = df
                ss.spec = spec
                st.success(f"✅ Validated {len(df)} tags")
                
        except Exception as e:
            st.error(f"Error loading file: {e}")

# ---- Main Interface ----
st.title(f"Spec-Driven Verification: {ss.project_scope}")

tab1, tab2, tab3, tab4 = st.tabs(["📋 I/O Definition", "🎮 Simulation", "📜 Trip Logic", "📑 Reports"])

# TAB 1: I/O Definition
with tab1:
    if ss.spec is not None:
        st.markdown("### ✅ Specification Loaded")
        
        # Show summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Tags", ss.spec["meta"]["tag_count"])
        m2.metric("Scope", ss.spec["meta"]["scope"])
        m3.metric("Generated", ss.spec["meta"]["generated_at"][:16])
        
        st.markdown("#### Tag Details")
        st.dataframe(ss.io_df, use_container_width=True)
        
        with st.expander("🔍 View Internal Spec JSON"):
            st.json(ss.spec)
            
    else:
        st.info("👈 Please upload an I/O list to begin.")
        
        # Placeholder for Phase 0 default
        if st.button("Load Default Compressor Spec (Demo)"):
            # Create a dummy DF
            data = {
                "Tag": ["TI-101", "PI-201", "VI-301", "FI-401"],
                "Description": ["Thrust Bearing Temp", "Lube Oil Pressure", "Vibration", "Seal Flow"],
                "Unit": ["°C", "bar", "mm/s", "L/s"],
                "Limit_High": [95.0, 5.0, 9.0, 100.0],
                "Limit_Low": [0.0, 3.6, 0.0, 0.70],
                "Min": [0.0, 0.0, 0.0, 0.0],
                "Max": [150.0, 10.0, 20.0, 5.0],
                "Trip_Delay_s": [3.0, 2.0, 3.0, 5.0]
            }
            df = pd.DataFrame(data)
            spec, err = validate_and_build_spec(df)
            if not err:
                ss.io_df = df
                ss.spec = spec
                st.rerun()

# TAB 2: Simulation
with tab2:
    st.markdown("### I/O Driven Simulation")
    
    if ss.spec is None:
        st.warning("👈 Please upload and validate an I/O list first.")
    else:
        # --- Simulation State Init ---
        if "sim_running" not in ss: ss.sim_running = False
        if "sim_t" not in ss: ss.sim_t = 0.0
        if "sim_vals" not in ss: ss.sim_vals = {}
        if "sim_log" not in ss: ss.sim_log = []
        
        # Initialize values if empty
        if not ss.sim_vals:
            for tag, props in ss.spec["tags"].items():
                # Start at 50% of range or 0
                mid = (props["limit_low"] + props["limit_high"]) / 2
                ss.sim_vals[tag] = mid
        
        # --- Controls ---
        c1, c2, c3, c4 = st.columns([1, 1, 1, 4])
        if c1.button("▶ Start"): 
            ss.sim_running = True
            if not ss.sim_log: # Clear log on fresh start if needed, or keep appending? 
                # Let's clear for a fresh run
                ss.sim_log = []
                
        if c2.button("⏸ Stop"): 
            ss.sim_running = False
            
        if c3.button("⟲ Reset"):
            ss.sim_running = False
            ss.sim_t = 0.0
            ss.sim_vals = {}
            ss.sim_log = []
            st.rerun()
            
        st.caption(f"Time: {ss.sim_t:.1f}s | Status: {'RUNNING' if ss.sim_running else 'PAUSED'} | Logged Rows: {len(ss.sim_log)}")
        
        # --- Simulation Loop (Auto-Refresh) ---
        if ss.sim_running:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=1000, key="sim_refresh")
            
            # Step Logic
            dt = 1.0
            ss.sim_t += dt
            import numpy as np
            
            current_row = {"t": ss.sim_t, "timestamp": pd.Timestamp.now().isoformat()}
            
            for tag, val in ss.sim_vals.items():
                # Simple Random Walk / Noise
                props = ss.spec["tags"][tag]
                noise = np.random.normal(0, (props["max"] - props["min"]) * 0.01)
                new_val = val + noise
                # Clamp
                ss.sim_vals[tag] = max(props["min"], min(props["max"], new_val))
                current_row[tag] = ss.sim_vals[tag]
            
            ss.sim_log.append(current_row)
        
        # --- Dynamic Dashboard ---
        st.divider()
        st.markdown("#### Live Sensor Data")
        
        # Grid Layout
        tags = list(ss.spec["tags"].keys())
        cols = st.columns(4)
        
        for i, tag in enumerate(tags):
            val = ss.sim_vals.get(tag, 0.0)
            props = ss.spec["tags"][tag]
            
            # Determine Alert
            is_alert = (val >= props["limit_high"]) or (val <= props["limit_low"])
            limit_type = "High" if val >= props["limit_high"] else "Low"
            limit_val = props["limit_high"] if val >= props["limit_high"] else props["limit_low"]
            
            # Render Card
            col = cols[i % 4]
            utils.render_kpi_card(
                col, 
                tag, 
                round(val, 2), 
                props["unit"], 
                limit_val, 
                is_alert, 
                limit_type
            )

        # --- Post-Run Actions ---
        if not ss.sim_running and ss.sim_log:
            st.divider()
            st.markdown("### 🛑 Run Complete - Generators Unlocked")
            
            # Show Log Preview
            df_log = pd.DataFrame(ss.sim_log)
            with st.expander("📊 View Simulation Log", expanded=False):
                st.dataframe(df_log, use_container_width=True)
                st.download_button("⬇️ Download CSV Log", df_log.to_csv(index=False).encode('utf-8'), "sim_log.csv", "text/csv")

            c_gen1, c_gen2 = st.columns(2)
            
            # --- ST Code Generator ---
            with c_gen1:
                st.markdown("#### 🏭 PLC Logic (ST)")
                if st.button("Generate IEC 61131-3 ST"):
                    st_code = "(* Auto-Generated Trip Logic *)\n"
                    st_code += "VAR_INPUT\n"
                    for tag in tags:
                        st_code += f"    {tag} : REAL; (* {ss.spec['tags'][tag]['description']} *)\n"
                    st_code += "END_VAR\nVAR\n    Trip : BOOL;\nEND_VAR\n\n"
                    
                    st_code += "(* Limit Checks *)\n"
                    for tag in tags:
                        props = ss.spec['tags'][tag]
                        st_code += f"ALM_{tag} := ({tag} >= {props['limit_high']}) OR ({tag} <= {props['limit_low']});\n"
                    
                    st_code += "\n(* Global Trip *)\n"
                    alarms = [f"ALM_{tag}" for tag in tags]
                    st_code += f"Trip := {' OR '.join(alarms)};\n"
                    
                    st.code(st_code, language="iec61131")
                    st.download_button("⬇️ Download .st", st_code, "logic.st")

            # --- Ladder Logic Renderer ---
            with c_gen2:
                st.markdown("#### 🪜 Ladder Diagram")
                if st.button("Generate Ladder Diagram"):
                    mermaid = "graph LR\n"
                    mermaid += "    %% Ladder Logic Representation\n"
                    mermaid += "    Power[| |] --> Rail(( ))\n"
                    
                    for i, tag in enumerate(tags):
                        props = ss.spec['tags'][tag]
                        node = f"N{i}"
                        mermaid += f"    Rail --> {node}[{tag} > {props['limit_high']}]\n"
                        mermaid += f"    {node} --> Coil((TRIP))\n"
                    
                    st.markdown(f"```mermaid\n{mermaid}\n```")
                    st.info("Visual representation of the OR logic.")

# TAB 3: Trip Logic
with tab3:
    st.markdown("### Deterministic Trip Engine")
    st.write("This section will generate the IEC 61131-3 ST code and Ladder Logic diagrams.")

# TAB 4: Reports
with tab4:
    st.markdown("### FAT Report Builder")
    st.write("Generate PDF reports based on simulation runs.")
