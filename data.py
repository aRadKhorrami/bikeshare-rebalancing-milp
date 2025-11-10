import numpy as np
import pandas as pd

def get_sample_data():
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
    
    I0 = {s: max(1, int(C[s] * 0.5)) for s in stations}
    
    print(f"   → Using 'NAME' as station key for matching")
    return stations, C, c, I0, coords


def load_real_data(trip_file, station_file, time_bin='2h'):
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
    
    h, p, F = 0.1, 10.0, 5
    M = sum(C.values())
    
    return common, times, I0, C, D, c, h, p, F, M