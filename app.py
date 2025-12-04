"""
Bikeshare Rebalancing Dashboard
===============================

Interactive Streamlit web application for solving and visualizing
the bikeshare rebalancing MILP model on real or sample data.

Features:
- Upload real October 2025 trip and station data
- Select top N busiest stations and time horizon
- Choose between SCIP and Gurobi solvers
- View inventory, unmet demand, and rebalancing plan
- Interactive Plotly visualizations
- Download results as CSV

Author: Ali Rad Khorrami  
Date: December 2025  
Project: Final MILP Model for Bikeshare Rebalancing
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import solve_model
import data
import os

# Page config
st.set_page_config(page_title="Bikeshare Rebalancing - Ali Rad Khorrami", layout="wide")
st.title("Bikeshare Rebalancing Problem: Final MILP Model")
st.markdown("**Ali Rad Khorrami** – December 2025")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    use_sample = st.checkbox("Use Sample Data (for testing)", value=False)
    
    if not use_sample:
        st.info("Upload your real files below")
        trip_file = st.file_uploader("202510-capitalbikeshare-tripdata.csv", type="csv")
        station_file = st.file_uploader("Capital_Bikeshare_Locations.csv", type="csv")
        time_bin = st.selectbox("Time granularity", ['1h', '2h', '4h'], index=1)
        n_stations = st.slider("Top N busiest stations", 5, 812, 100)
        n_periods = st.slider("Time periods to optimize", 3, 12, 12)
    else:
        st.success("Using sample data")

    st.markdown("### Costs")
    h = st.number_input("Holding cost per bike-hour (h)", 0.01, 1.0, 0.1, 0.05)
    p = st.number_input("Penalty per unmet demand (p)", 1.0, 50.0, 10.0, 1.0)

    st.markdown("### Fleet")
    use_fleet = st.checkbox("Limit number of trucks", value=True)
    F = st.number_input("Maximum trucks available", 1, 15, 5) if use_fleet else 5
    
    st.markdown("#### ⏱️ Time Limit")
    time_limit = st.number_input("Max solving time (seconds)", 30, 1200, 300)

    # GAP LIMIT CONTROL
    st.markdown("#### 📊 Optimality Tolerance")
    col1, col2 = st.columns([3, 1])
    with col1:
        gap_limit = st.slider(
            "Acceptable optimality gap (%)",
            min_value=0.1,
            max_value=20.0,
            value=1.0,
            step=0.1,
            help="Higher = faster solutions, Lower = more accurate"
        )
    with col2:
        st.metric("Gap", f"{gap_limit}%")
    
    # Visual indicator of what gap means
    if gap_limit < 1:
        st.info("🔍 **High Accuracy**: Will try to find solution within 1% of optimal")
    elif gap_limit < 5:
        st.info("⚖️ **Balanced**: Good balance between speed and accuracy")
    else:
        st.info("⚡ **Fast**: Will accept solution within 5%+ of optimal for speed")
    
    # Trade-off explanation
    with st.expander("ℹ️ What does 'optimality gap' mean?"):
        st.markdown("""
        **Optimality Gap** is how close the solution needs to be to the absolute best:
        
        - **0% gap** = Must find the mathematically optimal solution (slowest)
        - **1% gap** = Solution can be up to 1% worse than optimal (balanced)
        - **5% gap** = Solution can be up to 5% worse than optimal (faster)
        - **10% gap** = Solution can be up to 10% worse than optimal (fastest)
        
        For large problems, even 1% gap solutions are often good enough!
        """) 

    st.markdown("### Solver")
    solver_choice = st.radio(
        "Select MILP Solver",
        options=["SCIP (pyscipopt)", "Gurobi"],
        index=0,
        help="Gurobi is much faster for large instances if you have a license."
    )

# Main app
if st.button("RUN OPTIMIZATION", type="primary", use_container_width=True):
    if not use_sample and (not trip_file or not station_file):
        st.error("Please upload both CSV files!")
        st.stop()

    with st.spinner("Loading data and solving MILP..."):
        # Save uploaded files temporarily
        if not use_sample:
            with open("temp_trip.csv", "wb") as f:
                f.write(trip_file.getbuffer())
            with open("temp_station.csv", "wb") as f:
                f.write(station_file.getbuffer())

            # Load real data
            S_full, T_full, _, _, D_full, _, _, _, _, _ = data.load_real_data(
                "temp_trip.csv", "temp_station.csv", time_bin=time_bin)

            # Select top N stations
            demand_by_station = {s: sum(D_full.get((s,t),0) for t in T_full) for s in S_full}
            top_stations = sorted(demand_by_station, key=demand_by_station.get, reverse=True)[:n_stations]
            periods = T_full[:n_periods]

            subset_s = top_stations
            subset_t = periods
            data_source = 'real'
        else:
            subset_s = subset_t = None
            data_source = 'sample'

        # SOLVE!
        results, status = solve_model(
            use_fleet_constraint=use_fleet,
            data_source=data_source,
            h=h, p=p, F=F,
            subset_stations=subset_s,
            subset_times=subset_t,
            time_limit=time_limit,
            gap_limit=gap_limit/100.0,  # Convert percentage to decimal
            solver="gurobi" if "Gurobi" in solver_choice else "scip"   
        )

    if results:
        # optimality check - integrated handling
        is_optimal = results.get('is_optimal', True)
        obj_val = results.get('obj_val')
        gap = results.get('gap', 0.0)

        if obj_val and isinstance(obj_val, (int, float)):
            if status == "gaplimit":
                # Special message for gap limit reached
                st.info(f"✅ Solution found within acceptable tolerance! (Gap: {gap*100:.2f}%)")
                st.success(f"Best Cost = ${obj_val:,.2f}")
                st.balloons()
            elif is_optimal:
                st.success(f"✅ **OPTIMAL SOLUTION FOUND!** Total Cost = ${obj_val:,.2f}")
                st.balloons()
            else:
                st.warning(f"⏱️ **BEST SOLUTION FOUND (Time Limit)**")
                st.info(f"Best Cost = ${obj_val:,.2f} | Gap = {gap*100:.2f}% | Not proven optimal")
                st.balloons()
        else:
            st.warning(f"Feasible solution found • Best known cost: {obj_val}")

        # === Show results tabs even if not fully optimal ===
        tab1, tab2, tab3, tab4 = st.tabs(["Inventory", "Unmet Demand", "Rebalancing Plan", "Visualization"])

        with tab1:
            st.subheader("Bike Inventory Over Time (I_{i,t})")
            I_df = pd.DataFrame([
                {"Station": s, "Time": t, "Bikes": results['I'][(s,t)]}
                for (s,t) in results['I']
            ])
            st.dataframe(I_df.pivot(index="Station", columns="Time", values="Bikes").round(1))

        with tab2:
            st.subheader("Unmet Demand (B_{i,t})")
            B_df = pd.DataFrame([
                {"Station": s, "Time": t, "Lost Rentals": results['B'][(s,t)]}
                for (s,t) in results['B']
            ])
            pivot = B_df.pivot(index="Station", columns="Time", values="Lost Rentals").fillna(0)
            st.dataframe(pivot.style.background_gradient(cmap='Reds'))

        with tab3:
            st.subheader("Truck Movements (f_{i,j,t} > 0)")
            moves = [(i,j,t,v) for (i,j,t),v in results['f'].items() if v > 0.5]
            if moves:
                move_df = pd.DataFrame(moves, columns=["From", "To", "Time", "Bikes Moved"])
                st.dataframe(move_df)
            else:
                st.info("No rebalancing needed — perfect balance!")

        with tab4:
            st.subheader("Bike Levels Over Time")
            fig = go.Figure()
            for station in sorted(set(s for (s,t) in results['I']))[:10]:
                df_s = I_df[I_df["Station"] == station]
                fig.add_trace(go.Scatter(
                    x=df_s["Time"], y=df_s["Bikes"],
                    mode='lines+markers', name=station[:30]
                ))
            fig.update_layout(
                xaxis_title="Time Period",
                yaxis_title="Number of Bikes",
                legend_title="Station",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download button
        csv = pd.DataFrame.from_dict(results['I'], orient='index', columns=['Bikes']).to_csv()
        st.download_button(
            label="Download Full Results (CSV)",
            data=csv,
            file_name="bikeshare_rebalancing_solution.csv",
            mime="text/csv"
        )

    else:
        st.error(f"Solver status: {status}")
        st.info("Try reducing number of stations or time periods.")

# Footer
st.markdown("---")
st.markdown("**Ali Rad Khorrami** | Capital Bikeshare Rebalancing | December 2025")