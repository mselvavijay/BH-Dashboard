import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import os

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Baker Hughes IET Dashboard", layout="wide")

# ---------------------- COLORS ----------------------
BG_COLOR = "#0D1117"
TEXT_COLOR = "#EAEAEA"
PRIMARY_COLOR = "#00A878"
ALERT_COLOR = "#FF4B4B"
CARD_BG = "#1C1F26"

# ---------------------- AUTO REFRESH ----------------------
st_autorefresh(interval=5000, key="auto_refresh")

# ---------------------- CRITICAL LIMITS ----------------------
CRITICAL_LIMITS = {
    "ThrustBearingTemp": 80.0,
    "LubeOilPressure": 1.5,
    "Vibration": 8.0,
    "SealFlow": 0.7
}

CSV_FILE = "sensor_data_log.csv"

# ---------------------- SESSION STATE ----------------------
if "data" not in st.session_state:
    st.session_state.data = {
        "ThrustBearingTemp": round(np.random.uniform(60,75),2),
        "LubeOilPressure": round(np.random.uniform(1.8,2.5),2),
        "Vibration": round(np.random.uniform(4,6),2),
        "SealFlow": round(np.random.uniform(0.8,1.2),2),
        "status": "NORMAL",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
if "critical_injection_active" not in st.session_state:
    st.session_state.critical_injection_active = False
if "trip_in_progress" not in st.session_state:
    st.session_state.trip_in_progress = False
if "cooldown_phase" not in st.session_state:
    st.session_state.cooldown_phase = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "machine_stopped" not in st.session_state:
    st.session_state.machine_stopped = False

# ---------------------- CSV INIT ----------------------
def init_csv():
    df = pd.DataFrame(columns=["ThrustBearingTemp","LubeOilPressure","Vibration","SealFlow","status","timestamp"])
    df.to_csv(CSV_FILE,index=False)

def reset_system():
    st.session_state.critical_injection_active = False
    st.session_state.trip_in_progress = False
    st.session_state.cooldown_phase = False
    st.session_state.machine_stopped = False
    st.session_state.chat_history = []
    st.session_state.data = {
        "ThrustBearingTemp": round(np.random.uniform(60,75),2),
        "LubeOilPressure": round(np.random.uniform(1.8,2.5),2),
        "Vibration": round(np.random.uniform(4,6),2),
        "SealFlow": round(np.random.uniform(0.8,1.2),2),
        "status": "NORMAL",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
    init_csv()

if not os.path.exists(CSV_FILE):
    init_csv()

def save_to_csv(data):
    df = pd.DataFrame([data])
    df.to_csv(CSV_FILE, mode='a', header=False, index=False)

def fetch_latest_csv():
    df = pd.read_csv(CSV_FILE)
    if not df.empty:
        return df.iloc[-1]
    return None

# ---------------------- DATA GENERATION ----------------------
def generate_normal_data():
    return {
        "ThrustBearingTemp": round(np.random.uniform(60,75),2),
        "LubeOilPressure": round(np.random.uniform(1.8,2.5),2),
        "Vibration": round(np.random.uniform(4,6),2),
        "SealFlow": round(np.random.uniform(0.8,1.2),2)
    }

def update_data():
    data = st.session_state.data.copy()

    # If machine stopped and cooldown not yet started
    if st.session_state.machine_stopped and not st.session_state.cooldown_phase:
        data["status"] = "STOPPED"

    else:
        # Critical injection phase
        if st.session_state.critical_injection_active and not st.session_state.trip_in_progress:
            # Gradually increase ThrustBearingTemp
            data["ThrustBearingTemp"] += np.random.uniform(1.5,2.5)
            data["Vibration"] += np.random.uniform(0.2,0.5)
            data["LubeOilPressure"] -= np.random.uniform(0.05,0.15)
            data["SealFlow"] -= np.random.uniform(0.05,0.1)

            # Round all values to 2 decimals
            for k in ["ThrustBearingTemp","Vibration","LubeOilPressure","SealFlow"]:
                data[k] = round(data[k],2)

            # Check for breaches
            breaches = 0
            if data["ThrustBearingTemp"] >= CRITICAL_LIMITS["ThrustBearingTemp"]:
                breaches += 1
            if data["Vibration"] >= CRITICAL_LIMITS["Vibration"]:
                breaches += 1
            if data["LubeOilPressure"] <= CRITICAL_LIMITS["LubeOilPressure"]:
                breaches += 1
            if data["SealFlow"] <= CRITICAL_LIMITS["SealFlow"]:
                breaches += 1

            # Trip if >=2 breaches
            if breaches >= 2:
                data["status"] = "TRIPPED"
                st.session_state.trip_in_progress = True
                st.session_state.critical_injection_active = False
                st.session_state.machine_stopped = True
                st.session_state.cooldown_phase = True
            else:
                data["status"] = "ALERT"

        # Cooldown phase after trip
        elif st.session_state.cooldown_phase:
            # Gradually reduce temperature and vibration, increase pressure and seal flow
            data["ThrustBearingTemp"] = round(max(50, data["ThrustBearingTemp"] - np.random.uniform(1,2)),2)
            data["Vibration"] = round(max(1, data["Vibration"] - np.random.uniform(0.1,0.3)),2)
            data["LubeOilPressure"] = round(min(2.2, data["LubeOilPressure"] + np.random.uniform(0.05,0.1)),2)
            data["SealFlow"] = round(min(0.8, data["SealFlow"] + np.random.uniform(0.02,0.05)),2)
            data["status"] = "COOLDOWN"

            # Stop cooldown when values reach near-normal
            if (data["ThrustBearingTemp"] <= 60 and data["Vibration"] <= 2 and
                data["LubeOilPressure"] >= 1.8 and data["SealFlow"] >= 0.75):
                st.session_state.cooldown_phase = False
                data["status"] = "STOPPED"

        # Normal operation
        else:
            normal_data = generate_normal_data()
            for k,v in normal_data.items():
                data[k] = round(v,2)
            data["status"] = "NORMAL"

    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.data = data
    save_to_csv(data)

update_data()

# ---------------------- DASHBOARD UI ----------------------
st.markdown(f"<div style='background-color:{BG_COLOR}; padding:10px;'>"
            f"<h1 style='color:{PRIMARY_COLOR}; text-align:center;'>Baker Hughes Turbomachinery Dashboard</h1></div>",unsafe_allow_html=True)

cols = st.columns(4)
metrics = ["ThrustBearingTemp","LubeOilPressure","Vibration","SealFlow"]
for i,metric in enumerate(metrics):
    col = cols[i]
    val = round(st.session_state.data[metric],2)
    status = st.session_state.data["status"]
    color = PRIMARY_COLOR if status=="NORMAL" else ALERT_COLOR
    col.markdown(f"""
        <div style='background-color:{CARD_BG}; padding:15px; border-radius:10px; text-align:center;'>
        <strong style='color:{color}; font-size:16px'>{metric}</strong><br>
        <span style='color:{TEXT_COLOR}; font-size:28px; font-weight:bold'>{val:.2f}</span><br>
        <span style='color:#AAAAAA; font-size:12px;'>Limit: {CRITICAL_LIMITS[metric]}</span>
        </div>
    """,unsafe_allow_html=True)

st.markdown(f"<p style='color:{TEXT_COLOR}; text-align:center; font-size:14px;'>Current Data Timestamp: {st.session_state.data['timestamp']}</p>",unsafe_allow_html=True)

# Control Buttons
col_btns = st.columns(2)
with col_btns[0]:
    if st.button("⚠️ Trigger Critical Alert"):
        if not st.session_state.machine_stopped:
            st.session_state.critical_injection_active = True
with col_btns[1]:
    if st.button("🔄 RESET SYSTEM"):
        reset_system()

# Alerts
status = st.session_state.data["status"]
if status=="TRIPPED":
    st.error("🚨 Machine TRIPPED! Critical parameters exceeded limits!")
elif status=="ALERT":
    st.warning("⚠️ Alert: Parameters approaching critical limits")
elif status=="COOLDOWN":
    st.info("❄️ Machine in COOLDOWN phase")
elif status=="STOPPED":
    st.info("🛑 Machine stopped. Press RESET to restart.")
else:
    st.success("Perfect ,  All parameters are normal")


# ---------------------- Chatbot ----------------------
st.markdown("---")
st.subheader("🤖 Baker Hughes Monitoring Assistant")

latest_status = st.session_state.data["status"]

# Suggested Questions depending on machine status
if latest_status in ["NORMAL", "ALERT"]:
    suggested_questions = [
        "Check parameters approaching critical limits",
        "Avg performance of turbomachine in last 1 minute",
        "Check lubrication and seal health"
    ]
else:  # TRIPPED / COOLDOWN / STOPPED
    suggested_questions = [
        "Cause of machine trip",
        "Which parameter needs monitoring before restart",
        "Read more about machine support"
    ]

# Display horizontal buttons for suggestions
cols = st.columns(len(suggested_questions))
for i, col in enumerate(cols):
    if col.button(suggested_questions[i], key=f"suggest_{latest_status}_{i}"):
        user_input = suggested_questions[i]
        st.session_state.chat_history.append({"user": user_input})

        latest_df = pd.read_csv(CSV_FILE)
        latest_df["timestamp"] = pd.to_datetime(latest_df["timestamp"])

        bot_reply = ""

        if latest_df.empty:
            bot_reply = "No data available yet."
        else:
            # Use latest row for most current values
            latest = latest_df.iloc[-1]
            temp = latest["ThrustBearingTemp"]
            vib = latest["Vibration"]
            press = latest["LubeOilPressure"]
            flow = latest["SealFlow"]
            status_csv = latest["status"]
            timestamp_csv = latest["timestamp"]

            query = user_input.lower()

            # ---------- Responses ----------
            if "approaching critical limits" in query:
                messages = []
                if temp >= CRITICAL_LIMITS["ThrustBearingTemp"]*0.9:
                    messages.append("⚠️ Thrust Bearing Temp is rising rapidly – high risk")
                if vib >= CRITICAL_LIMITS["Vibration"]*0.9:
                    messages.append("⚠️ Vibration increasing – potential risk")
                if press <= CRITICAL_LIMITS["LubeOilPressure"]*1.1:
                    messages.append("⚠️ Lube Oil Pressure dropping – watch for low pressure")
                if flow <= CRITICAL_LIMITS["SealFlow"]*1.1:
                    messages.append("⚠️ Seal Flow decreasing – potential depletion")
                bot_reply = "\n".join(messages) if messages else "All parameters normal, no immediate risk detected."

            elif "avg performance" in query:
                last_minute = latest_df[latest_df["timestamp"] >= (datetime.now() - pd.Timedelta(minutes=1))]
                if not last_minute.empty:
                    avg_temp = last_minute["ThrustBearingTemp"].mean()
                    avg_vib = last_minute["Vibration"].mean()
                    avg_press = last_minute["LubeOilPressure"].mean()
                    avg_flow = last_minute["SealFlow"].mean()
                    bot_reply = (f"Last 1 min average:\n"
                                 f"Thrust Bearing Temp: {avg_temp:.2f} °C\n"
                                 f"Vibration: {avg_vib:.2f} mm/s\n"
                                 f"Lube Oil Pressure: {avg_press:.2f} bar\n"
                                 f"Seal Flow: {avg_flow:.2f} L/s")
                else:
                    bot_reply = "No data available for last 1 minute."

            elif "lubrication" in query:
                bot_reply = "Seal and lubrication parameters are within normal limits. Monitor regularly for deviations."

            elif "cause of machine trip" in query and latest_status in ["TRIPPED","COOLDOWN","STOPPED"]:
                # Find first timestamp when any param exceeded critical limit
                trip_row = latest_df[
                    (latest_df["ThrustBearingTemp"] >= CRITICAL_LIMITS["ThrustBearingTemp"]) |
                    (latest_df["Vibration"] >= CRITICAL_LIMITS["Vibration"]) |
                    (latest_df["LubeOilPressure"] <= CRITICAL_LIMITS["LubeOilPressure"]) |
                    (latest_df["SealFlow"] <= CRITICAL_LIMITS["SealFlow"])
                ]
                if not trip_row.empty:
                    first_trip = trip_row.iloc[0]
                    breaches = []
                    if first_trip["ThrustBearingTemp"] >= CRITICAL_LIMITS["ThrustBearingTemp"]:
                        breaches.append(f"ThrustBearingTemp:{first_trip['ThrustBearingTemp']:.2f}")
                    if first_trip["Vibration"] >= CRITICAL_LIMITS["Vibration"]:
                        breaches.append(f"Vibration:{first_trip['Vibration']:.2f}")
                    if first_trip["LubeOilPressure"] <= CRITICAL_LIMITS["LubeOilPressure"]:
                        breaches.append(f"LubeOilPressure:{first_trip['LubeOilPressure']:.2f}")
                    if first_trip["SealFlow"] <= CRITICAL_LIMITS["SealFlow"]:
                        breaches.append(f"SealFlow:{first_trip['SealFlow']:.2f}")
                    bot_reply = (f"Machine tripped at {first_trip['timestamp']} due to: {', '.join(breaches)}. "
                                 f"Other parameters gradually affected to avoid risk.")
                else:
                    bot_reply = "Machine tripped but no parameter breach detected."

            elif "parameter needs monitoring" in query:
                bot_reply = ("Before restarting, monitor Thrust Bearing Temp, Vibration, "
                             "Lube Oil Pressure, and Seal Flow closely. Ensure all are within normal limits.")

            elif "read more" in query:
                bot_reply = ("For detailed insights on turbomachine failure and support, "
                             "visit: machinesupport@bakerhughes.com article.")

            else:
                bot_reply = "Ask about temperature, vibration, pressure, seal flow, or trip causes."

        st.session_state.chat_history.append({"bot": bot_reply})

# Display full chat history in a chatbot style box
for chat in st.session_state.chat_history:
    user_msg = chat.get("user")
    bot_msg = chat.get("bot")
    if user_msg:
        st.markdown(
            f"<div style='text-align:right; margin-bottom:5px;'>"
            f"<span style='background-color:{PRIMARY_COLOR}; color:white; padding:10px; border-radius:12px; display:inline-block; max-width:80%'>{user_msg}</span></div>",
            unsafe_allow_html=True
        )
    if bot_msg:
        st.markdown(
            f"<div style='text-align:left; margin-bottom:10px;'>"
            f"<span style='background-color:{CARD_BG}; color:{TEXT_COLOR}; padding:10px; border-radius:12px; display:inline-block; max-width:80%'>{bot_msg}</span></div>",
            unsafe_allow_html=True
        )
