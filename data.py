"""
Data Loading and Preprocessing for Bikeshare Rebalancing
========================================================

This module loads and preprocesses Capital Bikeshare trip and station data
(e.g., from October 2025) for the Mixed-Integer Linear Programming (MILP)
rebalancing model.

Key Features:
- Matches trips to stations using exact station names
- Aggregates outbound trip demand into time bins (e.g., '1h', '2h', '4h')
- Computes Euclidean distances as a proxy for transportation costs
- Initializes inventory at approximately 50% of station capacity
- Provides sample data for testing

Dependencies: NumPy, Pandas.
"""

import numpy as np
import pandas as pd

def get_sample_data():
    """Return small synthetic dataset for quick testing and debugging."""
    stations = [1, 2, 3]
    times = [1, 2]
    I0 = {1: 10, 2: 5, 3: 0}
    C = {1: 20, 2: 15, 3: 10}
    h = 0.1
    p = 10
    F = 5
    M = 10000
    D = {(1,1):5,(1,2):3,(2,1):2,(2,2):4,(3,1):1,(3,2):6}
    c = {(1,2):2,(1,3):3,(2,1):2,(2,3):4,(3,1):3,(3,2):4}
    return stations, times, I0, C, D, c, h, p, F, M


def process_trip_data_to_demands(trip_file, time_bin='2h'):
    """
    Aggregate outbound trip starts from CSV into a demand matrix D_{i,t}.

    Assumes demand is based on trip starts ('started_at' and 'start_station_name').
    Ignores other columns. Handles large files with low_memory=False.

    Parameters
    ----------
    trip_file : str
        Path to trip data CSV (e.g., '202510-capitalbikeshare-tripdata.csv').
    time_bin : str, optional
        Pandas offset alias for binning (e.g., '1h' for hourly). Default '2h'.

    Returns
    -------
    stations : list[str]
        Sorted unique start station names.
    times : list[int]
        Consecutive integers starting from 1 for each time bin.
    D : dict[tuple[str, int], int]
        Demand counts {(station, time_period): num_starts}.
    """
    print(f"   → Loading trip data from {trip_file}...")
    df = pd.read_csv(trip_file, low_memory=False)
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['time_bin'] = df['started_at'].dt.floor(time_bin)
    
    # Use 'start_station_name' as the station key (since IDs don't match UUIDs)
    demands = df.groupby(['start_station_name', 'time_bin']).size()
    D = {}
    # Map timestamps to integer periods
    time_map = {ts: i+1 for i, ts in enumerate(sorted(demands.index.get_level_values(1).unique()))}
    
    for (station_name, ts), count in demands.items():
        D[(station_name, time_map[ts])] = count
    
    stations = sorted(demands.index.get_level_values(0).unique())
    times = list(range(1, len(time_map)+1))
    
    print(f"   → Processed {len(df):,} trips across {len(stations)} stations and {len(times)} periods.")
    return stations, times, D


def load_station_data(station_file):
    """
    Parse station CSV for capacities, coordinates, and derived parameters.

    Requires 'NAME' (or equivalent) for station names. Flexible on lat/lon/capacity column names (case-insensitive).
    Drops rows with missing data. Computes pairwise Euclidean distances (in lat/lon units; not km).

    Parameters
    ----------
    station_file : str
        Path to station CSV (e.g., 'Capital_Bikeshare_Locations.csv').

    Returns
    -------
    stations : list[str]
        Sorted station names.
    C : dict[str, int]
        Capacity {station: max_bikes}.
    c : dict[tuple[str, str], float]
        Euclidean distance {(i, j): dist} for i != j.
    I0 : dict[str, int]
        Initial bikes {station: ~50% capacity, min 1}.
    coords : dict[str, tuple[float, float]]
        {station: (lat, lon)}.
    """
    print(f"   → Loading station data from {station_file}...")
    df = pd.read_csv(station_file)
    print(f"   → {len(df)} stations loaded")
    print(f"   → Columns: {list(df.columns)}")
    
    # Use 'NAME' as the station key (matches 'start_station_name' in trips)
    if 'NAME' not in df.columns:
        raise ValueError("No 'NAME' column found! This CSV must have station names.")
    
    # Lat/Lon and Capacity
    lat_col = 'LATITUDE' if 'LATITUDE' in df.columns else 'lat'
    lon_col = 'LONGITUDE' if 'LONGITUDE' in df.columns else 'lon'
    cap_col = 'CAPACITY' if 'CAPACITY' in df.columns else 'capacity'
    
    # Normalize column names (flexible matching for lat/lon/capacity)
    df = df.rename(columns={
        'NAME': 'station_name',
        lat_col: 'lat',
        lon_col: 'lon',
        cap_col: 'capacity'
    })
    
    df = df.dropna(subset=['station_name', 'lat', 'lon', 'capacity'])  # Drop incomplete rows and ensure capacity is integer
    df['capacity'] = df['capacity'].astype(int)
    
    stations = df['station_name'].tolist()
    C = dict(zip(stations, df['capacity']))
    coords = dict(zip(stations, zip(df['lat'], df['lon'])))
    
    # Compute pairwise Euclidean distances (in degrees; proxy for cost)
    c = {}
    for i in stations:
        for j in stations:
            if i != j:
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                c[(i,j)] = np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)
    
    # map each station to its initial bike inventory (roughly 50% of capacity, but at least 1 bike).
    I0 = {s: max(1, int(C[s] * 0.5)) for s in stations}
    
    print(f"→ Using 'station_name' as key for trip matching.")
    return stations, C, c, I0, coords


def load_real_data(trip_file, station_file, time_bin='2h'):
    """
    Integrate trip demands with station metadata for MILP inputs.

    Filters to stations common in both datasets (exact name match).
    Sets default costs (h=0.1, p=10.0) and fleet (F=5). Big-M as total capacity.
    Adds coordinates for visualization.

    Parameters
    ----------
    trip_file : str
        Path to trip CSV.
    station_file : str
        Path to station CSV.
    time_bin : str, optional
        Time bin size. Default '2h'.

    Returns
    -------
    S : list[str]
        Common stations.
    T : list[int]
        Time periods.
    I0 : dict[str, int]
        Initial inventory.
    C : dict[str, int]
        Capacities.
    D : dict[tuple[str, int], int]
        Demands.
    c : dict[tuple[str, str], float]
        Distances.
    h : float
        Holding cost.
    p : float
        Penalty cost.
    F : int
        Max trucks.
    M : int
        Big-M value.
    coords : dict[str, tuple[float, float]]
        Station coordinates.
    """
    stations1, times, D = process_trip_data_to_demands(trip_file, time_bin)
    stations2, C, c, I0, coords = load_station_data(station_file)
    
    common = sorted(set(stations1) & set(stations2))
    print(f"→ Found {len(common):,} common stations (exact name match).")
    
    if len(common) < len(stations1) * 0.5:
        print("⚠️ Warning: Only {len(common)/len(stations1):.1%} match rate. Verify station names match exactly (case-sensitive).")
    
    # Filter parameters to common stations
    D = {(i,t): D.get((i,t), 0) for i in common for t in times}
    I0 = {i: I0[i] for i in common}
    C = {i: C[i] for i in common}
    c = {(i,j): c.get((i,j), 0) for i in common for j in common if i != j}
    coords = {i: coords[i] for i in common}  
    
    h, p, F = 0.1, 10.0, 5  # Set default model hyperparameters
    M = sum(C.values())
    
    return common, times, I0, C, D, c, h, p, F, M, coords  