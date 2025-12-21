"""
Bikeshare Rebalancing Dashboard
===============================

Interactive Streamlit web application for solving and visualizing
the bikeshare rebalancing Mixed-Integer Linear Programming (MILP) model
using real or sample data from Capital Bikeshare.

Key Features:
- Upload real trip data (e.g., October 2025) and station locations
- Select top N busiest stations and optimization time horizon
- Choose between open-source SCIP (via PySCIPOpt) or commercial Gurobi solvers
- Visualize inventory levels, unmet demand, and truck rebalancing plans
- Interactive maps with Folium/GeoPandas, including time-based animations
- Export results as CSV for further analysis

Dependencies: Streamlit, Pandas, NumPy, Plotly, Folium, GeoPandas, Branca, PySCIPOpt (for SCIP), Gurobi (optional).
Setup: Run `streamlit run app_folium.py` after installing dependencies.

Author: Ali Rad Khorrami  
Date: December 2025  
Project: Final MILP Model for Bikeshare Rebalancing
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import solve_model
import data
import os
import time
from datetime import datetime, timedelta

# Imports for interactive mapping and geospatial visualization
import folium
from folium import plugins
import geopandas as gpd
from shapely.geometry import Point, LineString
import branca.colormap as cm
from streamlit_folium import st_folium
import json

# Set Streamlit page configuration (title and layout)
st.set_page_config(page_title="Bikeshare Rebalancing - Ali Rad Khorrami", layout="wide")

# ============================================
# INITIALIZE SESSION STATE FOR PERSISTENCE
# ============================================
# Note: Session state persists user inputs and results across reruns for a smoother experience.
if 'results' not in st.session_state:
    st.session_state.results = None
if 'status' not in st.session_state:
    st.session_state.status = None
if 'C' not in st.session_state:
    st.session_state.C = None
if 'config' not in st.session_state:
    st.session_state.config = {}
if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
# Removed unused animation session keys

# ============================================
# HELPER FUNCTIONS FOR MAP VISUALIZATION
# ============================================
# These functions generate geometric elements (e.g., arrowheads) for Folium maps.

def create_arrowhead(start_lat, start_lon, end_lat, end_lon, line_length_km=0.1):
    """
    Calculate coordinates for a triangular arrowhead at the end of a line segment.
    
    This uses spherical geometry approximations to create a small arrow pointing
    towards the end point, suitable for overlay on Folium maps.
    
    Args:
        start_lat: Latitude of the starting point.
        start_lon: Longitude of the starting point.
        end_lat: Latitude of the ending point.
        end_lon: Longitude of the ending point.
        line_length_km: Approximate length of the line in km (used to scale arrow size).
    
    Returns:
        List of (lat, lon) tuples forming a closed polygon for the arrowhead.
    """
    import math
    
    lat1 = math.radians(start_lat)
    lon1 = math.radians(start_lon)
    lat2 = math.radians(end_lat)
    lon2 = math.radians(end_lon)
    
    d_lon = lon2 - lon1
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.atan2(y, x)
    bearing_deg = math.degrees(bearing)
    
    arrow_length = line_length_km * 0.02
    arrow_angle = 30
    
    left_angle = math.radians(bearing_deg + 180 + arrow_angle)
    left_lat = end_lat + (arrow_length * math.cos(left_angle))
    left_lon = end_lon + (arrow_length * math.sin(left_angle) / math.cos(math.radians(end_lat)))
    
    right_angle = math.radians(bearing_deg + 180 - arrow_angle)
    right_lat = end_lat + (arrow_length * math.cos(right_angle))
    right_lon = end_lon + (arrow_length * math.sin(right_angle) / math.cos(math.radians(end_lat)))
    
    return [(end_lat, end_lon), (left_lat, left_lon), (right_lat, right_lon), (end_lat, end_lon)]

# ============================================
# SIDEBAR - CONFIGURATION
# ============================================
# This section handles user inputs for data, costs, fleet, and solver settings.
with st.sidebar:
    st.header("Configuration")
    
    use_sample = st.checkbox("Use Sample Data (for testing)", 
                             value=st.session_state.config.get('use_sample', False),
                             key='use_sample')
    
    if not use_sample:
        st.info("Upload your real files below")
        trip_file = st.file_uploader("202510-capitalbikeshare-tripdata.csv", type="csv", key='trip_file',
                                      help="Upload the trip data CSV for October 2025 (must include 'started_at' and 'start_station_name' columns).")
        station_file = st.file_uploader("Capital_Bikeshare_Locations.csv", type="csv", key='station_file',
                                        help="Upload the station locations CSV (must include 'NAME', 'LATITUDE', 'LONGITUDE', and 'CAPACITY' or equivalents).")
        time_bin = st.selectbox("Time granularity", ['1h', '2h', '4h'], index=1, key='time_bin',
                                help="Choose time bin size for aggregating demand (e.g., '1h' bins trips into hourly periods).")
        n_stations = st.slider("Top N busiest stations", 5, 812, 100, key='n_stations',
                               help="Select the number of busiest stations based on total demand; limits model size for faster solving.")
        n_periods = st.slider("Time periods to optimize", 3, 12, 12, key='n_periods',
                              help="Number of time periods to optimize over; starts from the beginning of the data.")
        
        st.session_state.config.update({
            'trip_file_uploaded': trip_file is not None,
            'station_file_uploaded': station_file is not None,
            'time_bin': time_bin,
            'n_stations': n_stations,
            'n_periods': n_periods
        })
    else:
        st.success("Using sample data")
        st.session_state.config['use_sample'] = True

    st.markdown("### Costs")
    h = st.number_input("Holding cost per bike-hour (h)", 0.01, 1.0, 0.1, 0.05, key='h',
                        help="Cost of holding one bike at a station for one time period (e.g., maintenance or opportunity cost).")
    p = st.number_input("Penalty per unmet demand (p)", 1.0, 50.0, 10.0, 1.0, key='p',
                        help="Penalty for each unmet rental demand (lost revenue or customer dissatisfaction).")
    
    st.session_state.config.update({'h': h, 'p': p})

    st.markdown("### Fleet")
    use_fleet = st.checkbox("Limit number of trucks", 
                            value=st.session_state.config.get('use_fleet', True), key='use_fleet',
                            help="If checked, constrain the maximum number of trucks; otherwise, unlimited trucks are assumed.")
    if use_fleet:
        F = st.number_input("Maximum trucks available", 1, 15, 5, key='F',
                            help="Maximum number of trucks available for rebalancing across all periods.")
    else:
        F = 5
    
    st.session_state.config.update({'use_fleet': use_fleet, 'F': F})
    
    st.markdown("#### ⏱️ Time Limit")
    time_limit = st.number_input("Max solving time (seconds)", 30, 1200, 300, key='time_limit',
                                 help="Maximum time (in seconds) allowed for the solver to find a solution.")
    st.session_state.config['time_limit'] = time_limit

    st.markdown("#### 📊 Optimality Tolerance")
    col1, col2 = st.columns([3, 1])
    with col1:
        gap_limit = st.slider(
            "Acceptable optimality gap (%)",
            min_value=0.1,
            max_value=20.0,
            value=st.session_state.config.get('gap_limit', 1.0),
            step=0.1,
            help="Acceptable deviation from the optimal solution (as %); higher values allow faster solving but less accurate results.",
            key='gap_limit'
        )
    with col2:
        st.metric("Gap", f"{gap_limit}%")
    
    st.session_state.config['gap_limit'] = gap_limit
    
    if gap_limit < 1:
        st.info("🔍 **High Accuracy**: Will try to find solution within 1% of optimal")
    elif gap_limit < 5:
        st.info("⚖️ **Balanced**: Good balance between speed and accuracy")
    else:
        st.info("⚡ **Fast**: Will accept solution within 5%+ of optimal for speed")
    
    with st.expander("ℹ️ What does 'optimality gap' mean?"):
        st.markdown("""
        **Optimality Gap** is how close the solution needs to be to the absolute best:
        
        - **0% gap** = Must find the mathematically optimal solution (slowest)
        - **1% gap** = Solution can be up to 1% worse than optimal (balanced)
        - **5% gap** = Solution can be up to 5% worse than optimal (faster)
        - **10% gap** = Solution can be up to 10% worse than optimal (fastest)
        
        For large problems, even 1% gap solutions are often good enough!
        """) 

    st.markdown("### Service Level")
    use_service_level = st.checkbox("Enforce Minimum Service Level", 
                                     value=st.session_state.config.get('use_service_level', False), key='use_service_level',
                                     help="If checked, add constraints to ensure at least X% of demand is met at each station-period.")
    if use_service_level:
        service_level_pct = st.slider(
            "Minimum demand fulfillment (%)",
            min_value=70,
            max_value=99,
            value=st.session_state.config.get('service_level_pct', 90),
            step=1,
            help="Minimum percentage of demand that must be fulfilled; enforces B_{i,t} ≤ (100 - %) * D_{i,t} for each station i and time t (see model documentation Section 8).",
            key='service_level_pct'
        )
        service_level = service_level_pct / 100.0
    else:
        service_level = None
    
    st.session_state.config.update({
        'use_service_level': use_service_level,
        'service_level': service_level
    })

    st.markdown("### Solver")
    solver_options = ["SCIP (pyscipopt)", "Gurobi"]
    solver_choice = st.radio(
        "Select MILP Solver",
        options=solver_options,
        index=0,
        help="Select solver: SCIP is free/open-source but slower; Gurobi is faster for large problems but requires a license (academic/free options available).",
        key='solver_choice'
    )
    
    st.session_state.config['solver_choice'] = solver_choice
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 RUN OPTIMIZATION", width='stretch', type="primary",
                     help="Start the MILP optimization based on current settings."):
            if not use_sample and (not trip_file or not station_file):
                st.error("Please upload both CSV files!")
            else:
                st.session_state.run_optimization = True
                st.rerun()
    
    with col2:
        if st.button("🔄 RESET RESULTS", width='stretch', type="secondary",
                     help="Clear all results and session state to start fresh."):
            for key in ['results', 'status', 'C', 'run_complete', 'data_loaded']:
                if key in st.session_state:
                    del st.session_state[key]
            if 'timed_map' in st.session_state:
                del st.session_state.timed_map
            st.session_state.run_optimization = False
            st.success("Results cleared!")
            st.rerun()
    
    if st.session_state.run_complete:
        st.markdown("---")
        st.success("✅ Optimization Complete!")
        if st.session_state.results:
            obj_val = st.session_state.results.get('obj_val')
            if obj_val:
                st.metric("Total Cost", f"${obj_val:,.2f}")

# ============================================
# MAIN CONTENT AREA
# ============================================
# This section triggers the model solve and renders results if available.

# ============================================
# RUN OPTIMIZATION WHEN TRIGGERED
# ============================================
# Load data (real or sample), subset stations/periods, and call solve_model().
# Results are stored in session_state for display in tabs.
if st.session_state.get('run_optimization', False):
    with st.spinner("Loading data and solving MILP... This may take a few minutes..."):
        try:
            if not use_sample and trip_file and station_file: # Save uploaded files temporarily for processing
                with open("temp_trip.csv", "wb") as f:
                    f.write(trip_file.getbuffer())
                with open("temp_station.csv", "wb") as f:
                    f.write(station_file.getbuffer())

                S_full, T_full, _, C_full, D_full, _, _, _, _, _, coords_full = data.load_real_data(
                    "temp_trip.csv", "temp_station.csv", time_bin=time_bin) # Load and preprocess real trip/station data; aggregate demand by time bin

                demand_by_station = {s: sum(D_full.get((s,t),0) for t in T_full) for s in S_full}
                top_stations = sorted(demand_by_station, key=demand_by_station.get, reverse=True)[:n_stations] # Select top N stations by total demand to reduce model size
                periods = T_full[:n_periods]

                subset_s = top_stations
                subset_t = periods
                data_source = 'real'
            else:
                subset_s = subset_t = None
                data_source = 'sample'
                C_full = None

            # Solve the MILP model with current configuration
                # Returns decision variables (I, B, f, etc.) and solver status
            results, status = solve_model(
                use_fleet_constraint=use_fleet,
                data_source=data_source,
                h=h, p=p, F=F,
                subset_stations=subset_s,
                subset_times=subset_t,
                time_limit=time_limit,
                gap_limit=gap_limit/100.0,
                service_level=service_level,
                solver="gurobi" if "Gurobi" in solver_choice else "scip"   
            )

            st.session_state.results = results
            st.session_state.status = status
            st.session_state.C = C_full if data_source == 'real' else None
            st.session_state.run_complete = True
            st.session_state.data_loaded = True
            st.session_state.run_optimization = False
            
        except Exception as e:
            st.error(f"Error during optimization: {str(e)}")
            st.session_state.run_optimization = False
            st.stop()

# ============================================
# DISPLAY RESULTS (IF AVAILABLE)
# ============================================
# Render summary metrics and detailed tabs if optimization has completed successfully.
if st.session_state.run_complete and st.session_state.results:
    results = st.session_state.results
    status = st.session_state.status
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        is_optimal = results.get('is_optimal', True)
        obj_val = results.get('obj_val')
        gap = results.get('gap', 0.0) * 100
        
        if obj_val and isinstance(obj_val, (int, float)):
            if status == "gaplimit":   # Note: "gaplimit" is the status string returned by PySCIPOpt when MIPGap is reached
                st.info(f"✅ Solution found within acceptable tolerance! (Gap: {gap:.2f}%)")
                st.success(f"Best Cost = ${obj_val:,.2f}")
            elif is_optimal:
                st.success(f"✅ **OPTIMAL SOLUTION FOUND!** Total Cost = ${obj_val:,.2f}")
            else:
                st.warning(f"⏱️ **BEST SOLUTION FOUND (Time Limit)**")
                st.info(f"Best Cost = ${obj_val:,.2f} | Gap = {gap:.2f}% | Not proven optimal")
        else:
            st.warning(f"Feasible solution found • Best known cost: {obj_val}")
    
    with col2:
        if 'I' in results:
            n_stations = len(set(s for (s, t) in results['I']))
            n_periods = len(set(t for (s, t) in results['I']))
            st.metric("Stations", n_stations)
    
    with col3:
        if 'I' in results:
            total_moves = sum(1 for v in results['f'].values() if v > 0.5)   # Count non-zero truck movements (threshold 0.5 to avoid floating-point noise)
            st.metric("Movements", total_moves)
    
    st.markdown("---")
    
    tabs = ["Inventory", "Unmet Demand", "Rebalancing Plan", "Visualization", "Rebalancing Map"]
    tab_objects = st.tabs(tabs)
    
    # Inventory Tab
    with tab_objects[0]:
        st.subheader("Bike Inventory Over Time (I_{i,t})")
        I_df = pd.DataFrame([
            {"Station": s, "Time": t, "Bikes": results['I'][(s,t)]}
            for (s,t) in results['I']
        ])
        
        # Display key inventory metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Inventory", f"{I_df['Bikes'].mean():.1f}")
        with col2:
            st.metric("Max Inventory", f"{I_df['Bikes'].max():.0f}")
        with col3:
            st.metric("Min Inventory", f"{I_df['Bikes'].min():.0f}")
        # Full-width pivot table
        st.dataframe(
            I_df.pivot(index="Station", columns="Time", values="Bikes").round(1),
            width='stretch'
        )            

    # Unmet Demand Tab
    with tab_objects[1]:
        st.subheader("Unmet Demand (B_{i,t})")
        B_df = pd.DataFrame([
            {"Station": s, "Time": t, "Lost Rentals": results['B'][(s,t)]}
            for (s,t) in results['B']
        ])
        
        # Calculate and display key unmet demand metrics, including achieved service level
        total_unmet = B_df['Lost Rentals'].sum()
        avg_unmet = B_df['Lost Rentals'].mean()
        service_level_achieved = 100.0  # Default if D not available
        if 'D' in results:
            total_demand = sum(results['D'].values())
            service_level_achieved = 100 * (1 - total_unmet / total_demand) if total_demand > 0 else 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Unmet Demand", f"{total_unmet:.0f}")
        with col2:
            st.metric("Average per Station", f"{avg_unmet:.1f}")
        with col3:
            st.metric("Service Level", f"{service_level_achieved:.1f}%")
        
        # Full-width pivot table with styling
        pivot = B_df.pivot(index="Station", columns="Time", values="Lost Rentals").fillna(0)
        st.dataframe(
            pivot.style.background_gradient(cmap='Reds'),
            width='stretch'
        )                
    
    # Rebalancing Plan Tab
    with tab_objects[2]:
        st.subheader("Truck Movements (f_{i,j,t} > 0)")
        # Extract positive truck movements (f > 0.5 to handle floating-point precision)
        moves = [(i,j,t,v) for (i,j,t),v in results['f'].items() if v > 0.5]
        
        if moves:
            move_df = pd.DataFrame(moves, columns=["From", "To", "Time", "Bikes Moved"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                total_moves = len(move_df)
                st.metric("Total Movements", total_moves)
            with col2:
                total_bikes = move_df['Bikes Moved'].sum()
                st.metric("Total Bikes Moved", f"{total_bikes:.0f}")
            with col3:
                avg_move = move_df['Bikes Moved'].mean()
                st.metric("Avg per Move", f"{avg_move:.1f}")
            
            st.dataframe(move_df)
            
            st.subheader("Movements by Time Period")
            time_periods = sorted(move_df['Time'].unique())
            selected_time = st.selectbox("Select Time Period", time_periods, key="movement_time")
            
            filtered_moves = move_df[move_df['Time'] == selected_time]
            st.dataframe(filtered_moves)
        else:
            st.info("No rebalancing needed — perfect balance!")
    
    # Visualization Tab
    with tab_objects[3]:
        st.subheader("Bike Levels Over Time")
        fig = go.Figure()
        
        top_stations = sorted(set(s for (s,t) in results['I']))
        # Limit to top 10 stations by average inventory for chart clarity
        if len(top_stations) > 10:
            station_avgs = {}
            for station in top_stations:
                station_inv = [results['I'][(station, t)] for t in set(t for (s,t) in results['I'] if s == station)]
                station_avgs[station] = np.mean(station_inv) if station_inv else 0
            
            top_10 = sorted(station_avgs, key=station_avgs.get, reverse=True)[:10]
        else:
            top_10 = top_stations
        
        for station in top_10:
            times = sorted([t for (s,t) in results['I'] if s == station])
            inventories = [results['I'][(station, t)] for t in times]
            
            fig.add_trace(go.Scatter(
                x=times, y=inventories,
                mode='lines+markers',
                name=str(station) if isinstance(station, int) else station[:30],
                hovertemplate='<b>%{fullName}</b><br>Time: %{x}<br>Bikes: %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Number of Bikes",
            legend_title="Station",
            height=600,
            hovermode='closest'
        )
        st.plotly_chart(fig, width='stretch')
    
    # Rebalancing Map Tab
    # Generate and display animated Folium map using TimestampedGeoJson plugin
    # Features: time-based station inventory (size/color) + animated truck movements with arrows
    with tab_objects[4]:
        st.subheader("Interactive Rebalancing Map")
        
        if 'coords' in results and results['coords']:
            # Extract stations, times, movements, inventories, and coordinates for mapping
            S = sorted(set(s for (s, t) in results['I']))
            T = sorted(set(t for (s, t) in results['I']))
            f_data = results['f']
            I_data = results['I']
            coords = results['coords']
            
            C = st.session_state.C
            if C is None:
                C = {s: 20 for s in S}
            
            # Configure time periods for TimestampedGeoJson animation (ISO 8601 format required)
            time_bin = st.session_state.config.get('time_bin', '1h')
            interval_hours = int(time_bin[:-1])
            period = f'PT{interval_hours}H'
            
            base_time = datetime(2025, 12, 1, 0, 0, 0)
            time_map = {t: (base_time + timedelta(hours=interval_hours * (t - min(T)))).isoformat() for t in T}
            
            # Inventory scaling & colormap
            max_inventory = max(I_data.values()) if I_data else 1
            colormap = cm.LinearColormap(
                colors=['green', 'yellow', 'red'],
                vmin=0,
                vmax=max_inventory,
                caption='Bike Inventory'
            )
            
            # Map style 
            map_style = "cartodbpositron"
            
            # Cache the Folium map in session_state to prevent regeneration on every rerun
            if 'timed_map' not in st.session_state:
                valid_coords = [coords[s] for s in S if s in coords]
                if not valid_coords:
                    st.error("No valid coordinates available.")
                else:
                    # Calculate geographic center for initial map view
                    center_lat = np.mean([c[0] for c in valid_coords])
                    center_lon = np.mean([c[1] for c in valid_coords])
                    
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=12,
                        tiles=map_style,
                        control_scale=True
                    )
                    
                    plugins.Fullscreen().add_to(m)
                    
                    # Create timed GeoJSON features for stations (one per station-time pair)
                    # Style: radius and color scaled by current inventory
                    stations_features = []
                    for t in T:
                        iso_time = time_map[t]
                        for station in S:
                            if station not in coords:
                                continue
                            lat, lon = coords[station]
                            inventory = I_data.get((station, t), 0)
                            capacity = C.get(station, 20)
                            fill_pct = (inventory / capacity * 100) if capacity > 0 else 0
                            
                            radius = 5 + (inventory / max_inventory) * 20 if max_inventory > 0 else 10
                            color = colormap(inventory)
                            
                            popup_content = f"""
                            <div style="font-family: Arial, sans-serif; width: 220px;">
                                <h4 style="color: #333; margin-bottom: 8px;">{station}</h4>
                                <table style="width: 100%; font-size: 13px;">
                                    <tr><td><strong>Time Period:</strong></td><td>{t}</td></tr>
                                    <tr><td><strong>Bikes:</strong></td><td>{inventory:.0f}</td></tr>
                                    <tr><td><strong>Capacity:</strong></td><td>{capacity}</td></tr>
                                    <tr><td><strong>Fill %:</strong></td><td>{fill_pct:.1f}%</td></tr>
                                </table>
                            </div>
                            """
                            
                            stations_features.append({
                                'type': 'Feature',
                                'geometry': {
                                    'type': 'Point',
                                    'coordinates': [lon, lat]
                                },
                                'properties': {
                                    'times': [iso_time],
                                    'popup': popup_content,
                                    'tooltip': f"{station}: {inventory:.0f} bikes (T={t})",
                                    'style': {
                                        'radius': radius,
                                        'fillColor': color,
                                        'color': color,
                                        'weight': 2,
                                        'opacity': 1,
                                        'fillOpacity': 0.7
                                    },
                                    'icon': 'circle'
                                }
                            })
                    
                    # Create timed GeoJSON features for truck movements
                    # - Red arrowhead (Polygon) for direction
                    # - Thick red line (MultiLineString) for path
                    movements_features = []
                    for (i, j, t), val in f_data.items():
                        if val > 0 and i in coords and j in coords:
                            iso_time = time_map[t]
                            lat1, lon1 = coords[i]
                            lat2, lon2 = coords[j]
                            
                            line_weight = 8 + min(val / 3, 12)  # Very thick for visibility
                            
                            popup_content = f"<b>Movement</b><br>From: {i}<br>To: {j}<br>Bikes: {val:.0f}<br>Time: {t}"
                            tooltip = f"{i} → {j}: {val:.0f} bikes"
                            
                          
                            # Arrowhead as filled Polygon

                            arrow_coords = create_arrowhead(lat1, lon1, lat2, lon2)
                            arrow_poly_coords = [[[lon, lat] for lat, lon in arrow_coords]]
                            movement_coords = [
                                [[lon1, lat1], [lon2, lat2]],  # Main line
                                arrow_poly_coords[0]  # Arrow as line (simplified)
                            ]

                            movements_features.append({
                                'type': 'Feature',
                                'geometry': {
                                    'type': 'Polygon',
                                    'coordinates': arrow_poly_coords
                                },
                                'properties': {
                                    'times': [iso_time],
                                    'style': {
                                        'fillColor': '#ff0000',
                                        'color': '#ff0000',
                                        'fillOpacity': 0.95,
                                        'weight': 1
                                    }
                                }
                            })

                            movements_features.append({
                                'type': 'Feature',
                                'geometry': {'type': 'MultiLineString', 'coordinates': movement_coords},
                                'properties': {'times': [iso_time], 'popup': popup_content, 
                                    'tooltip': tooltip, 'style': {'color': '#ff0000', 'weight': 4, 'opacity': 1}}
                            })                        
                    
                    # Combine all features
                    all_features = stations_features + movements_features
                    
                    geojson_data = {
                        'type': 'FeatureCollection',
                        'features': all_features
                    }
                    
                    # Add time-based animation plugin
                    # Features: auto-play, loop, speed control, manual slider
                    plugins.TimestampedGeoJson(
                        geojson_data,
                        period=period,
                        duration=period,
                        auto_play=True,
                        loop=True,
                        max_speed=10,
                        loop_button=True,
                        time_slider_drag_update=True,
                        add_last_point=False
                    ).add_to(m)
                    
                    folium.LayerControl(collapsed=False).add_to(m)
                    colormap.add_to(m)
                    
                    st.session_state.timed_map = m
            
            # Display map
            if 'timed_map' in st.session_state:
                st_folium(st.session_state.timed_map, width=1000, height=600, returned_objects=[])
            else:
                st.error("Failed to generate the animated map.")
            
        
        else:
            st.info("Interactive map not available for sample data or when coordinates are missing.")

else:
    st.markdown("""
    ## Welcome to the Bikeshare Rebalancing Dashboard!
    
    ### How to use:
    1. **Configure** your parameters in the sidebar
    2. **Upload** trip and station data files (or use sample data)
    3. **Click** 'RUN OPTIMIZATION' to solve the MILP model
    4. **Explore** results in the interactive tabs
    5. **In Rebalancing Map tab**, use the built-in timeline slider for animation (auto-plays with loop/speed controls)
    6. **Export**: Download results as CSV from relevant tabs (if implemented)
    
    ### Animation Features:
    - Animation runs client-side on the map without reloads
    - Use the bottom slider to select time periods manually
    - Stations update color/size based on bike inventory; movements appear/disappear over time
    
    ### Quick Start
    
    #### Option 1 – Quick Testing
    - Check **"Use Sample Data"** in the sidebar
    - Click **RUN OPTIMIZATION** → instantly solves a small synthetic problem
    
    #### Option 2 – Real Capital Bikeshare Data (October 2025)
    Upload the two required CSV files. The app matches stations by name and aggregates demand into time bins.
    - Station names in both files **must match exactly** (case-sensitive).

    **Required File 1: Trip Data** (`202510-capitalbikeshare-tripdata.csv`)

    | Column Name         | Data Type                  | Required | Description                                                                 |
    |---------------------|----------------------------|----------|-----------------------------------------------------------------------------|
    | started_at          | DateTime (e.g., 2025-10-01 08:15:23) | Yes      | Timestamp when the trip started. Used to bin trips into time periods.       |
    | start_station_name  | String                     | Yes      | Exact name of the starting station. Used as the station identifier.        |

    *Any additional columns are ignored.*

    **Required File 2: Station Locations** (`Capital_Bikeshare_Locations.csv`)

    | Column Name   | Alternative Names Accepted | Data Type | Required | Description                                      |
    |---------------|----------------------------|-----------|----------|--------------------------------------------------|
    | NAME          | —                          | String    | Yes      | Station name (must match `start_station_name` exactly) |
    | LATITUDE      | lat                        | Float     | Yes      | Latitude coordinate                              |
    | LONGITUDE     | lon                        | Float     | Yes      | Longitude coordinate                             |
    | CAPACITY      | capacity                   | Integer   | Yes      | Maximum number of bikes/docks at the station     |

    *Column names for latitude, longitude, and capacity are flexible (case-insensitive alternatives shown). Rows with missing data are dropped.*

    - After uploading both files → Click **RUN OPTIMIZATION**
    """)

# App footer with author and project credit
st.markdown("---")
st.markdown("**Ali Rad Khorrami** | Capital Bikeshare Rebalancing | December 2025")

# Custom CSS for improved visual design
# - Rounded, shadowed maps
# - Hover effects on buttons
# - Better spacing and tab layout
st.markdown("""
<style>
    .folium-map {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 2px solid #e0e0e0;
    }
    
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Improved spacing for Figma-like design */
    section[data-testid="stMarkdownContainer"] {
        margin-bottom: 20px;
    }
    
    .stTabs [data-testid="stHorizontalBlock"] {
        gap: 16px;
    }
</style>
""", unsafe_allow_html=True)