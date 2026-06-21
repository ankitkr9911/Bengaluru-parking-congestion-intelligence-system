# ============================================================================
# app.py — Streamlit Dashboard + AI Chat Interface
# ============================================================================
# Two-tab layout:
#   Tab 1: Interactive hotspot map + CIS rankings + charts
#   Tab 2: AI Assistant chat (powered by LangGraph agents)
#
# Run: streamlit run dashboard/app.py
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import os
import sys
from pathlib import Path

# Load .env from project root (local dev)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Streamlit Cloud: pull secrets into env vars if not already set
try:
    import streamlit as _st_tmp
    if "OPENAI_API_KEY" in _st_tmp.secrets and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = _st_tmp.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ParkSight AI — Bengaluru Parking Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark premium theme overrides */
    .main { background-color: #0e1117; }
    .stMetric { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px; padding: 15px; border: 1px solid #1f4068; }
    .stMetric label { color: #8892b0 !important; font-size: 0.85rem; }
    .stMetric [data-testid="stMetricValue"] { color: #64ffda !important; font-size: 1.8rem; }
    h1, h2, h3 { color: #ccd6f6 !important; }
    .block-container { padding-top: 2.5rem; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a192f 0%, #112240 100%); }
    div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 { color: #64ffda !important; }
    .css-1d391kg { padding: 1rem; }
    /* Chat styling */
    .stChatMessage { border-radius: 12px; }

    /* Tab styling — make 3 tabs clearly visible */
    .stTabs [data-baseweb="tab-list"] {
        display: inline-flex !important;
        gap: 16px;
        padding: 10px;
        width: fit-content !important;

        background: rgba(26, 26, 46, 0.9);
        border: 1px solid #1f4068;
        border-radius: 14px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        border-radius: 10px;
        color: #8892b0;
        font-size: 1rem;
        font-weight: 500;

        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(100,255,218,0.15);

        margin-right: 4px;
        transition: all 0.2s ease;
    }
    .stTabs {
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(100,255,218,0.08);
        border-color: rgba(100,255,218,0.35);
        color: #ccd6f6;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(
            135deg,
            rgba(31,64,104,0.95),
            rgba(40,70,130,0.95)
        ) !important;

        color: #64ffda !important;
        border: 1px solid #64ffda !important;
        box-shadow: 0 0 12px rgba(100,255,218,0.25);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #64ffda;
        height: 3px;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load all pre-computed pipeline outputs."""
    data = {}
    files = {
        "impact": "impact_scores.csv",
        "hotspots": "hotspots.csv",
        "forecasts": "forecasts.csv",
        "patrol": "patrol_schedule.csv",
        "violations": "cleaned_violations.csv",
    }
    for key, filename in files.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()
            st.warning(f"⚠️ {filename} not found. Run the pipeline first.")
    return data


data = load_data()
impact_df = data.get("impact", pd.DataFrame())
hotspots_df = data.get("hotspots", pd.DataFrame())
forecasts_df = data.get("forecasts", pd.DataFrame())
patrol_df = data.get("patrol", pd.DataFrame())
violations_df = data.get("violations", pd.DataFrame())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🚦 ParkSight AI")
    st.markdown("### Bengaluru Parking Intelligence")
    st.markdown("---")

    if not impact_df.empty:
        confirmed = impact_df[
            impact_df["hotspot_status"].isin(["CONFIRMED_HIGH", "CONFIRMED"])
        ]
        st.metric("Total Zones Analyzed", f"{len(impact_df):,}")
        st.metric("Confirmed Hotspots", f"{len(confirmed)}")
        st.metric("Max CIS Score", f"{impact_df['CIS'].max():.1f}")

        if not violations_df.empty:
            st.metric("Total Violations", f"{len(violations_df):,}")

        st.markdown("---")
        st.markdown("### 🔍 Filter")

        # Station filter
        stations = sorted(impact_df["police_station"].dropna().unique())
        selected_station = st.selectbox(
            "Police Station", ["All Stations"] + list(stations)
        )

        # Hotspot status filter
        statuses = ["All"] + sorted(impact_df["hotspot_status"].dropna().unique())
        selected_status = st.selectbox("Hotspot Status", statuses)

        # CIS threshold
        cis_min = st.slider("Min CIS Score", 0, 100, 0)
    else:
        st.error("No data loaded. Run the pipeline first.")
        selected_station = "All Stations"
        selected_status = "All"
        cis_min = 0

    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border: 1px solid #1f4068; border-radius: 10px; padding: 14px;">
        <div style="color: #8892b0; font-size: 0.7rem; letter-spacing: 0.5px;
                    text-transform: uppercase; margin-bottom: 4px;">Built for</div>
        <div style="color: #64ffda; font-size: 1rem; font-weight: 600; margin-bottom: 12px;">
            Flipkart Gridlock 2.0
        </div>
        <div style="color: #8892b0; font-size: 0.7rem; letter-spacing: 0.5px;
                    text-transform: uppercase; margin-bottom: 6px;">Powered by</div>
        <div style="display: flex; flex-wrap: wrap; gap: 6px;">
            <span title="ST-DBSCAN (spatiotemporal clustering)"
                  style="background: rgba(100,255,218,0.1); border: 1px solid rgba(100,255,218,0.3);
                         color: #64ffda; font-size: 0.7rem; padding: 3px 9px; border-radius: 12px;
                         cursor: help;">Hotspot Clustering</span>
            <span title="Getis-Ord Gi* (statistical significance test)"
                  style="background: rgba(100,255,218,0.1); border: 1px solid rgba(100,255,218,0.3);
                         color: #64ffda; font-size: 0.7rem; padding: 3px 9px; border-radius: 12px;
                         cursor: help;">Significance Testing</span>
            <span title="Local Moran's I / LISA (neighborhood cluster check)"
                  style="background: rgba(100,255,218,0.1); border: 1px solid rgba(100,255,218,0.3);
                         color: #64ffda; font-size: 0.7rem; padding: 3px 9px; border-radius: 12px;
                         cursor: help;">Cluster Validation</span>
            <span title="M/M/∞ Queueing Model (congestion delay estimation)"
                  style="background: rgba(100,255,218,0.1); border: 1px solid rgba(100,255,218,0.3);
                         color: #64ffda; font-size: 0.7rem; padding: 3px 9px; border-radius: 12px;
                         cursor: help;">Delay Modeling</span>
            <span title="LangGraph multi-agent orchestration"
                  style="background: rgba(100,255,218,0.1); border: 1px solid rgba(100,255,218,0.3);
                         color: #64ffda; font-size: 0.7rem; padding: 3px 9px; border-radius: 12px;
                         cursor: help;">AI Agents</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
def apply_filters(df):
    filtered = df.copy()
    if selected_station != "All Stations":
        filtered = filtered[filtered["police_station"] == selected_station]
    if selected_status != "All":
        filtered = filtered[filtered["hotspot_status"] == selected_status]
    if "CIS" in filtered.columns:
        filtered = filtered[filtered["CIS"] >= cis_min]
    return filtered


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_map, tab_analysis, tab_chat = st.tabs([
    "🗺️ Hotspot Map", "📊 Analysis & Rankings", "💬 AI Assistant"
])


# =====================================================================
# TAB 1: HOTSPOT MAP
# =====================================================================
with tab_map:
    st.markdown("## 🗺️ Illegal Parking Hotspot Map")

    if not impact_df.empty and "lat" in impact_df.columns:
        filtered = apply_filters(impact_df)

        if not filtered.empty:
            # Summary metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Zones Shown", len(filtered))
            with col2:
                avg_cis = filtered["CIS"].mean()
                st.metric("Avg CIS", f"{avg_cis:.1f}")
            with col3:
                if "bpr_delay_min_per_km" in filtered.columns:
                    avg_delay = filtered["bpr_delay_min_per_km"].mean()
                    st.metric("Avg Delay", f"{avg_delay:.2f} min/km")
            with col4:
                high_risk = (filtered["CIS"] > 60).sum()
                st.metric("High Risk Zones", high_risk)

            # Build Folium map
            center_lat = filtered["lat"].mean()
            center_lon = filtered["lon"].mean()
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles="CartoDB dark_matter",
            )

            # Color scale by CIS
            max_cis = filtered["CIS"].max()
            min_cis = filtered["CIS"].min()

            for _, row in filtered.iterrows():
                # Color: green → yellow → red based on CIS
                cis_norm = (row["CIS"] - min_cis) / max(max_cis - min_cis, 1)
                if cis_norm > 0.7:
                    color = "#ff4444"  # Red - high impact
                elif cis_norm > 0.4:
                    color = "#ffaa00"  # Orange - medium
                else:
                    color = "#44ff44"  # Green - low

                # Circle size proportional to violation count
                radius = max(4, min(20, row.get("violation_count", 10) / 50))

                popup_html = f"""
                <div style="font-family:Arial; width:250px;">
                    <b>{row.get('location_name', 'Zone')}</b><br>
                    <b>CIS: {row['CIS']:.1f}/100</b> (Rank #{int(row.get('rank',0))})<br>
                    Status: {row.get('hotspot_status','N/A')}<br>
                    Violations: {int(row.get('violation_count',0))}<br>
                    Delay: {row.get('bpr_delay_min_per_km',0):.2f} min/km<br>
                    Road: {row.get('road_type','N/A')} ({int(row.get('lanes',0))} lanes)<br>
                    Vehicle: {row.get('dominant_vehicle','N/A')}
                </div>
                """

                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"CIS: {row['CIS']:.1f} | {row.get('location_name','')}",
                ).add_to(m)

            # Add legend
            legend_html = """
            <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                 background:rgba(0,0,0,0.8); padding:15px; border-radius:8px;
                 color:white; font-size:13px;">
                <b>CIS Legend</b><br>
                <span style="color:#ff4444">●</span> High Impact (CIS > 70%)<br>
                <span style="color:#ffaa00">●</span> Medium (CIS 40-70%)<br>
                <span style="color:#44ff44">●</span> Low (CIS < 40%)
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            st_folium(m, width=None, height=550, use_container_width=True)
        else:
            st.info("No zones match the current filters.")
    else:
        st.info("📊 Run the pipeline first to generate hotspot data.")


# =====================================================================
# TAB 2: ANALYSIS & RANKINGS
# =====================================================================
with tab_analysis:
    st.markdown("## 📊 Analysis & Rankings")

    if not impact_df.empty:
        filtered = apply_filters(impact_df)

        # --- TOP HOTSPOTS TABLE ---
        st.markdown("### 🏆 Top Hotspots by CIS")
        top_n = st.slider("Show top N", 5, 50, 15, key="top_n_slider")
        top = filtered.nlargest(top_n, "CIS")

        display_cols = ["rank", "location_name", "police_station", "CIS",
                       "hotspot_status", "violation_count", "bpr_delay_min_per_km",
                       "dominant_vehicle", "road_type", "peak_ratio"]
        available = [c for c in display_cols if c in top.columns]

        st.dataframe(
            top[available].style.background_gradient(subset=["CIS"], cmap="YlOrRd"),
            use_container_width=True,
            height=400,
        )

        st.markdown("---")

        # --- CHARTS ROW ---
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 CIS Distribution")
            fig_hist = px.histogram(
                filtered, x="CIS", nbins=30,
                color_discrete_sequence=["#64ffda"],
                labels={"CIS": "Congestion Impact Score"},
            )
            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.markdown("### 🏢 Hotspots by Station")
            if "police_station" in filtered.columns:
                station_counts = (
                    filtered[filtered["CIS"] > 50]
                    .groupby("police_station")
                    .size()
                    .nlargest(15)
                    .reset_index(name="high_cis_zones")
                )
                fig_bar = px.bar(
                    station_counts, x="high_cis_zones", y="police_station",
                    orientation="h",
                    color="high_cis_zones",
                    color_continuous_scale="YlOrRd",
                    labels={"high_cis_zones": "Zones with CIS > 50"},
                )
                fig_bar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # --- CIS COMPONENT BREAKDOWN ---
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### 🔬 CIS Components (Top 10 Hotspots)")
            cis_cols = [c for c in filtered.columns if c.startswith("cis_")]
            if cis_cols and len(top) > 0:
                top10 = filtered.nlargest(10, "CIS")
                component_means = top10[cis_cols].mean()
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=component_means.values,
                    theta=[c.replace("cis_", "").replace("_", " ").title()
                           for c in component_means.index],
                    fill="toself",
                    fillcolor="rgba(100, 255, 218, 0.2)",
                    line=dict(color="#64ffda"),
                    name="Top 10 Avg",
                ))
                fig_radar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    polar=dict(
                        radialaxis=dict(range=[0, 1], showticklabels=False),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        with col4:
            st.markdown("### 🚗 Vehicle Type Impact")
            if "dominant_vehicle" in filtered.columns:
                veh_dist = (
                    filtered.groupby("dominant_vehicle")["CIS"]
                    .mean()
                    .nlargest(10)
                    .reset_index()
                )
                fig_veh = px.bar(
                    veh_dist, x="CIS", y="dominant_vehicle",
                    orientation="h",
                    color="CIS",
                    color_continuous_scale="Viridis",
                    labels={"CIS": "Avg CIS", "dominant_vehicle": "Vehicle Type"},
                )
                fig_veh.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    showlegend=False,
                )
                st.plotly_chart(fig_veh, use_container_width=True)

        st.markdown("---")

        # --- PATROL SCHEDULE ---
        if not patrol_df.empty:
            st.markdown("### 📋 Patrol Schedule")
            patrol_filtered = patrol_df.copy()
            if selected_station != "All Stations":
                patrol_filtered = patrol_filtered[
                    patrol_filtered["police_station"] == selected_station
                ]

            if not patrol_filtered.empty:
                patrol_display = ["police_station", "priority_rank", "location_name",
                                 "CIS", "predicted_risk", "trend_direction",
                                 "recommended_shift", "reasoning"]
                available_p = [c for c in patrol_display if c in patrol_filtered.columns]
                st.dataframe(
                    patrol_filtered[available_p],
                    use_container_width=True,
                    height=300,
                )
    else:
        st.info("📊 Run the pipeline first to see analysis.")


# =====================================================================
# TAB 3: AI ASSISTANT
# =====================================================================
with tab_chat:
    st.markdown("## 💬 AI Parking Intelligence Assistant")
    st.markdown(
        "Ask questions about hotspots, enforcement, forecasts, and legal details. "
        "The system routes your query to the appropriate specialist agent."
    )

    # Example questions
    with st.expander("📝 Example questions"):
        st.markdown("""
        - *"What are the top 5 hotspots in Bengaluru?"*
        - *"Why is Koramangala a hotspot?"*
        - *"What should Madiwala station focus on this week?"*
        - *"Is the situation at Bellandur getting worse?"*
        - *"What's the legal basis for offence code 112?"*
        - *"Which areas have the most repeat offenders?"*
        - *"Give me overall system statistics"*
        """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about parking enforcement..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing with specialist agents..."):
                try:
                    from agents.graph import ask
                    response = ask(prompt)
                except ImportError:
                    # Fallback: use tools directly without LLM
                    response = _fallback_response(prompt)
                except Exception as e:
                    response = f"Agent error: {str(e)}\n\nMake sure OPENAI_API_KEY is set: $env:OPENAI_API_KEY='sk-...'"

            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def _fallback_response(question: str) -> str:
    """Fallback when LLM is not available — uses tools directly."""
    from agents.tools import (
        query_top_hotspots, get_zone_details, get_station_patrol_brief,
        get_general_stats, lookup_offence_code
    )

    q = question.lower()

    if any(w in q for w in ["top", "hotspot", "worst", "ranking"]):
        return query_top_hotspots(n=10)
    elif any(w in q for w in ["station", "patrol", "deploy", "focus"]):
        # Extract station name
        for word in question.split():
            if word[0].isupper() and len(word) > 3:
                result = get_station_patrol_brief(word)
                if "not found" not in result:
                    return result
        return get_station_patrol_brief("Madiwala")
    elif any(w in q for w in ["why", "detail", "breakdown", "explain"]):
        for word in question.split():
            if word[0].isupper() and len(word) > 3:
                result = get_zone_details(word)
                if "No data" not in result:
                    return result
        return get_zone_details("Koramangala")
    elif any(w in q for w in ["law", "legal", "offence", "code", "fine"]):
        return lookup_offence_code()
    elif any(w in q for w in ["stat", "overview", "summary"]):
        return get_general_stats()
    else:
        return get_general_stats()
