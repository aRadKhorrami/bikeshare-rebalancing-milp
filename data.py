"""
Data Loading and Preprocessing for Bikeshare Rebalancing
========================================================

This module handles loading and preprocessing of real Capital Bikeshare data
from October 2025 for use in the MILP rebalancing model.

Key features:
- Matches trip data with station metadata using station names
- Aggregates demand by user-defined time bins (1h, 2h, 4h)
- Computes Euclidean distance matrix as transportation cost proxy
- Sets realistic initial inventory (~50% capacity)
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
    Convert raw trip CSV into demand matrix D_it.

    Parameters
    ----------
    trip_file : str or file-like
        Path to Capital Bikeshare trip data CSV.
    time_bin : str, default='2h'
        Pandas frequency string for time aggregation (e.g., '1h', '2h', '4h').

    Returns
    -------
    stations : list
        List of station names appearing in trips.
    times : list
        Integer time period indices (1, 2, ..., T).
    D : dict
        Demand dictionary {(station, time): count}
    """
    print(f"   → Reading {trip_file}...")
    df = pd.read_csv(trip_file, low_memory=False)
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['time_bin'] = df['started_at'].dt.floor(time_bin)
    
    # Use 'start_station_name' as the station key (since IDs don't match UUIDs)
    demands = df.groupby(['start_station_name', 'time_bin']).size()
    D = {}
    time_map = {ts: i+1 for i, ts in enumerate(sorted(demands.index.get_level_values(1).unique()))}
    
    for (station_name, ts), count in demands.items():
        D[(station_name, time_map[ts])] = count
    
    stations = sorted(demands.index.get_level_values(0).unique())
    times = list(range(1, len(time_map)+1))
    
    print(f"   → {len(df):,} trips | {len(stations)} stations | {len(times)} time periods")
    return stations, times, D


def load_station_data(station_file):
    """
    Load station metadata (name, capacity, coordinates) from CSV.

    Parameters
    ----------
    station_file : str or file-like
        Path to station locations CSV (from DC Open Data).

    Returns
    -------
    stations : list
        Station names.
    C : dict
        Capacity {station: capacity}
    c : dict
        Distance matrix {(i,j): euclidean_distance}
    I0 : dict
        Initial inventory {station: bikes}
    coords : dict
        Coordinates {station: (lat, lon)}
    """
    print(f"   → Reading {station_file}...")
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
    
    df = df.rename(columns={
        'NAME': 'station_name',
        lat_col: 'lat',
        lon_col: 'lon',
        cap_col: 'capacity'
    })
    
    df = df.dropna(subset=['station_name', 'lat', 'lon', 'capacity'])
    df['capacity'] = df['capacity'].astype(int)
    
    stations = df['station_name'].tolist()
    C = dict(zip(stations, df['capacity']))
    coords = dict(zip(stations, zip(df['lat'], df['lon'])))
    
    # Distance matrix (Euclidean)
    c = {}
    for i in stations:
        for j in stations:
            if i != j:
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                c[(i,j)] = np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)
    
    # map each station to its initial bike inventory (roughly 50% of capacity, but at least 1 bike).
    I0 = {s: max(1, int(C[s] * 0.5)) for s in stations}
    
    print(f"   → Using 'NAME' as station key for matching")
    return stations, C, c, I0, coords


def load_real_data(trip_file, station_file, time_bin='2h'):
    """
    Load and merge real trip and station data for the MILP model.

    Matches stations by name, filters to common stations, and returns
    all parameters required by :func:`model.solve_model`.

    Parameters
    ----------
    trip_file, station_file : str
        Paths to trip and station CSVs.
    time_bin : str, default='2h'
        Time aggregation granularity.

    Returns
    -------
    S, T, I0, C, D, c, h, p, F, M
        All inputs required by the optimization model.
    """
    stations1, times, D = process_trip_data_to_demands(trip_file, time_bin)
    stations2, C, c, I0, coords = load_station_data(station_file)
    
    common = sorted(set(stations1) & set(stations2))
    print(f"   → {len(common)} common stations (matched by name)")
    
    if len(common) < len(stations1) * 0.5:
        print("   ⚠️ Warning: Low match rate - check if names are consistent between files")
    
    D = {(i,t): D.get((i,t), 0) for i in common for t in times}
    I0 = {i: I0[i] for i in common}
    C = {i: C[i] for i in common}
    c = {(i,j): c.get((i,j), 0) for i in common for j in common if i != j}
    coords = {i: coords[i] for i in common}  # NEW: Filter and return coords
    
    h, p, F = 0.1, 10.0, 5
    M = sum(C.values())
    
    return common, times, I0, C, D, c, h, p, F, M, coords  # NEW: Add coords to return