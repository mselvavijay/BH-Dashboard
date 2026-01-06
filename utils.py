import streamlit as st

# ---- Constants ----
BG, TEXT, CARD = "#0D1117", "#EAEAEA", "#1C1F26"
PRIMARY, ALERT, AMBER, OK = "#00A878", "#FF4B4B", "#F5A524", "#2ECC71"
LINE = "#AAAAAA"

def inject_css():
    """Injects the shared CSS styles for the application."""
    st.markdown("""
    <style>
        /* Global Theme Overrides */
        .stApp {
            background-color: #0E1117;
        }
        
        /* Headers */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            color: #EAEAEA;
            font-weight: 600;
        }
        
        /* Cards / Containers */
        .css-1r6slb0, .stMarkdown {
            color: #EAEAEA;
        }
        
        /* Custom Card Style */
        .kpi-card {
            background-color: #1C1F26;
            border-radius: 8px;
            padding: 16px;
            border: 1px solid #30363D;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            margin-bottom: 10px;
        }
        
        .kpi-value {
            font-size: 24px;
            font-weight: 700;
            margin: 8px 0;
        }
        
        .kpi-label {
            color: #8B949E;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        
        /* Sidebar adjustments */
        .css-1d391kg {
            background-color: #161B22;
        }

        /* Chat styling */
        .chat-bubble {
            padding: 10px 14px;
            border-radius: 12px;
            margin-bottom: 8px;
            max-width: 80%;
            font-size: 14px;
            line-height: 1.4;
        }
        .chat-bot {
            background-color: #1C1F26;
            border: 1px solid #30363D;
            color: #EAEAEA;
            margin-right: auto;
            border-bottom-left-radius: 2px;
        }
        .chat-user {
            background-color: #00A878;
            color: #FFFFFF;
            margin-left: auto;
            border-bottom-right-radius: 2px;
            text-align: right;
        }
    </style>
    """, unsafe_allow_html=True)

def render_kpi_card(col, label, value, unit, limit, is_alert, limit_type="High"):
    """Renders a standardized KPI card."""
    status_color = ALERT if is_alert else OK
    status_text = "CRITICAL" if is_alert else "NORMAL"
    
    col.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid {status_color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {status_color if is_alert else '#EAEAEA'}">
            {value} <span style="font-size: 14px; color: #8B949E;">{unit}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #8B949E; font-size: 12px;">{limit_type} Limit: {limit}</span>
            <span class="status-badge" style="background-color: {status_color}22; color: {status_color};">
                {status_text}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_chat_bubble(text, role="bot"):
    """Renders a styled chat bubble."""
    c_class = "chat-bot" if role=="bot" else "chat-user"
    icon = "🤖" if role=="bot" else "👤"
    
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: { 'flex-start' if role=='bot' else 'flex-end' };">
        <div class="chat-bubble {c_class}">
            <div style="font-size: 10px; opacity: 0.7; margin-bottom: 2px;">{icon} {role.upper()}</div>
            {text}
        </div>
    </div>
    """, unsafe_allow_html=True)
