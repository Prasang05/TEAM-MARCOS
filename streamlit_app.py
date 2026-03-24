import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import os
from dotenv import load_dotenv
import google.generativeai as genai
import warnings

warnings.filterwarnings('ignore')

# Securely load the API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="NakaAnalytics AI", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS TO MATCH REACT UI ---
st.markdown("""
<style>
    body { background-color: #f4f7f6; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #0f172a; font-weight: 700; }
    .kpi-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .kpi-title { font-size: 13px; color: #64748b; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px;}
    .kpi-value { font-size: 32px; color: #0f172a; font-weight: 800; }
    .stButton>button { width: 100%; font-size: 16px; font-weight: bold; background-color: #0f172a; color: white; padding: 12px; border-radius: 8px; border:none; box-shadow: 0 4px 6px rgba(15,23,42,0.1); }
    .stButton>button:hover { background-color: #1e293b; color: white; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# --- 1. SET UP CHAT STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello Commander! I am the Naka Copilot, now powered by **Google Gemini AI**. Ask me complex questions about the active traffic zones!"}
    ]

# --- 2. LOAD REAL DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('traffic_data.csv')
    except:
        df = pd.DataFrame({'location':['Sitabuldi'], 'lat':[21.1458], 'lng':[79.0882], 'time':['18:00'], 'violation_type':['helmet'], 'vehicle_type':['bike']})
        
    df.columns = [c.strip() for c in df.columns]
    
    def get_hour(t_str):
        if pd.isna(t_str): return 0
        return int(str(t_str).split(':')[0])
        
    df['hour'] = df['time'].apply(get_hour)
    df['vehicle'] = df['vehicle_type'].str.strip()
    df['violation'] = df['violation_type'].str.strip().str.upper()
    df['violation'] = df['violation'].str.replace("_", " ")
    return df

df = load_data()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.title("🎛️ Map Filters")

vehicles_selected = st.sidebar.multiselect("Select Vehicle Types", df['vehicle'].unique(), default=df['vehicle'].unique())
hours_selected = st.sidebar.slider("Select Time Window (Hour)", 0, 23, value=(16, 23))

filtered_df = df[(df['vehicle'].isin(vehicles_selected)) & 
                 (df['hour'] >= hours_selected[0]) & 
                 (df['hour'] <= hours_selected[1])]

# --- 4. TOP KPIs ---
st.markdown("<h1 style='margin-bottom:0;'>🚨 NakaAnalytics Dashboard</h1>", unsafe_allow_html=True)

agg_df = filtered_df.groupby(['location', 'lat', 'lng']).size().reset_index(name='count')
st.markdown("<br>", unsafe_allow_html=True)

# --- 5. MAP & DEMOGRAPHICS ---
col_map, col_charts = st.columns([1.3, 1])

with col_map:
    st.markdown("### 🗺️ Violation Density Map")
    
    m = folium.Map(location=[21.12, 79.07], zoom_start=13, tiles='OpenStreetMap')
    
    for idx, row in agg_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=row['count'] * 4,
            color='#ef4444',
            fill=True,
            fill_color='#ef4444',
            fill_opacity=0.6,
            weight=2,
            popup=f"<b>{row['location']}</b><br>Violations: {row['count']}"
        ).add_to(m)
        
    st_folium(m, width="100%", height=400, returned_objects=[])
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Filtered Points</div><div class="kpi-value">{len(filtered_df)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Active Hotspots</div><div class="kpi-value">{len(agg_df)}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card" style="background:#eff6ff; border-color:#bfdbfe;"><div class="kpi-title" style="color:#1e40af;">System Status</div><div class="kpi-value" style="color:#1d4ed8; font-size:26px;">Live Sync ✓</div></div>', unsafe_allow_html=True)

with col_charts:
    st.markdown("### 👥 Demographics Breakdown")
    
    if not filtered_df.empty:
        violation_counts = filtered_df['violation'].value_counts().reset_index()
        violation_counts.columns = ['Violation', 'Count']
        
        fig1 = px.pie(
            violation_counts, 
            values='Count', 
            names='Violation', 
            hole=0.65, 
            template='plotly_white',
            color_discrete_sequence=['#3b82f6', '#fbbf24', '#a78bfa', '#fb7185']
        )
        fig1.update_layout(margin=dict(l=0, r=0, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        fig1.update_traces(textposition='none')
        st.plotly_chart(fig1, use_container_width=True, height=260)
    else:
        st.write("No data matching filters")
        
    st.divider()

    st.markdown("### ⏱️ Smart Deployment")

    if st.button("Generate Deployment Plan"):
        with st.spinner("Processing Requirements..."):
            import time; time.sleep(0.4) 
            
            if len(agg_df) > 0:
                top_3 = agg_df.sort_values(by='count', ascending=False).head(3)
                
                html = "<table style='width:100%; border-collapse:collapse; background:white; text-align:left; border-radius:8px; overflow:hidden; border:1px solid #e2e8f0; margin-top: 15px;'>"
                html += "<tr style='background:#f8fafc; color:#475569; font-size:12px;'><th style='padding:12px;'>DEPLOY ZONE</th><th style='padding:12px;'>OPTIMAL TIME</th><th style='padding:12px; text-align:center;'>UNITS</th></tr>"
                
                for idx, row in top_3.iterrows():
                    loc_df = filtered_df[filtered_df['location'] == row['location']]
                    loc_peak = loc_df['hour'].mode()[0] if not loc_df.empty else 19
                    officers = max(2, int(row['count']))
                    
                    html += f"<tr>"
                    html += f"<td style='padding:12px; border-top:1px solid #e2e8f0; font-weight:600; font-size:14px; color:#334155;'>{row['location']}</td>"
                    html += f"<td style='padding:12px; border-top:1px solid #e2e8f0; color:#64748b; font-size:14px;'>{loc_peak}:00 - {(loc_peak+3)%24}:00</td>"
                    html += f"<td style='padding:12px; border-top:1px solid #e2e8f0; text-align:center;'><span style='background:#dcfce7; color:#166534; padding:4px 12px; border-radius:20px; font-weight:700; font-size:13px;'>{officers}</span></td>"
                    html += "</tr>"
                    
                html += "</table>"
                st.markdown(html, unsafe_allow_html=True)
                
            else:
                st.error("Not enough data points.")

st.divider()

# --- 6. AI COPILOT CHATBOT (GEMINI INTEGRATED) ---
st.markdown("## 💬 AI Naka Copilot (Powered by Gemini)")

# Suggested Questions Row
st.markdown("<p style='font-size:14px; color:#64748b; margin-bottom: 5px;'>Quick Queries:</p>", unsafe_allow_html=True)
sq_cols = st.columns(4)
btn_prompt = None
if sq_cols[0].button("📍 Most Active Zone?", use_container_width=True): btn_prompt = "Which area has the most traffic violations in the current data?"
if sq_cols[1].button("⏰ Peak Time?", use_container_width=True): btn_prompt = "What is the peak timeframe for violations and why?"
if sq_cols[2].button("🚗 Major Offender?", use_container_width=True): btn_prompt = "Which vehicle type is committing the most offenses right now?"
if sq_cols[3].button("🔥 Hotspot Analysis", use_container_width=True): btn_prompt = "Give me a detailed breakdown of the top 3 hotspots and suggested action for each."

# Display previous chats
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Combine Chat Input and Button Prompt
user_input = st.chat_input("Ask Gemini complex questions about this active traffic data...")
prompt = user_input if user_input else btn_prompt

# Process New Chat
if prompt:
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # TRUE AI Logic Engine
    with st.chat_message("assistant"):
        if not GEMINI_API_KEY:
            st.error("Gemini API key is missing or not loading from .env file!")
        else:
            with st.spinner("Gemini is analyzing the live dataset..."):
                try:
                    # Provide dynamic DataFrame context to the LLM
                    loc_counts = filtered_df['location'].value_counts()
                    top_locs = loc_counts.to_dict() if not loc_counts.empty else "No locations"
                    veh_counts = filtered_df['vehicle'].value_counts()
                    top_vehs = veh_counts.to_dict() if not veh_counts.empty else "No vehicles"
                    vio_counts = filtered_df['violation'].value_counts()
                    top_vios = vio_counts.to_dict() if not vio_counts.empty else "No violations"
                    
                    sys_instruction = f"""
                    You are NakaCopilot, an elite AI traffic enforcement assistant integrated directly into an active dashboard.
                    Here are the LIVE STATS of the user's actively filtered dashboard right now:
                    - Total Tracked Violations: {len(filtered_df)}
                    - Counts by Zone: {top_locs}
                    - Counts by Vehicle Class: {top_vehs}
                    - Counts by Offense Type: {top_vios}
                    Answer accurately based on this data. Be conversational but precise.
                    """
                    
                    model = genai.GenerativeModel("models/gemini-2.0-flash")
                    resp = model.generate_content(f"SYSTEM INSTRUCTION: {sys_instruction}\n\nUSER PROMPT: {prompt}")
                    res_text = resp.text
                    
                    st.markdown(res_text)
                    st.session_state.messages.append({"role": "assistant", "content": res_text})
                except Exception as e:
                    st.error(f"Gemini API Error: {str(e)}")
