# test_data.py
from data import load_real_data, get_sample_data

TRIP_FILE = "202510-capitalbikeshare-tripdata.csv"
STATION_FILE = "Capital_Bikeshare_Locations.csv"

print("🚀 TESTING WITH REAL OCTOBER 2025 DATA")
print(f"Files: {TRIP_FILE} + {STATION_FILE}")
print()

try:
    S, T, I0, C, D, c, h, p, F, M = load_real_data(TRIP_FILE, STATION_FILE, time_bin='2H')
    print("\n🎉 SUCCESS! Everything loaded perfectly!")
    print(f"Stations: {len(S)}")
    print(f"Time periods: {len(T)} (2-hour bins)")
    print(f"Total trips: {sum(D.values()):,}")
    print(f"Ready for optimization!")

except Exception as e:
    print("❌ ERROR:", e)
    import traceback
    traceback.print_exc()