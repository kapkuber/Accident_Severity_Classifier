import pandas as pd
import numpy as np

# LOAD RAW DATA
acc = pd.read_csv("US_Accidents_March23.csv", low_memory=False)
weather = pd.read_csv("WeatherEvents_Jan2016-Dec2022.csv", low_memory=False)

acc_cols = [
    "ID","Severity","Start_Time","End_Time",
    "Start_Lat","Start_Lng",
    "Temperature(F)","Humidity(%)","Visibility(mi)",
    "Wind_Speed(mph)","Precipitation(in)","Weather_Condition",

    # Road infrastructure
    "Amenity","Bump","Crossing","Give_Way","Junction","No_Exit","Railway",
    "Roundabout","Station","Stop","Traffic_Calming","Traffic_Signal","Turning_Loop",

    # Road / location descriptors
    "Street","City","County","State"
]

acc = acc[acc_cols]

weather_cols = [
    "EventId","Type","Severity","StartTime(UTC)","EndTime(UTC)",
    "Precipitation(in)","LocationLat","LocationLng","City","County","State"
]

weather = weather[weather_cols]

# Clean Timetsamps
acc["Start_Time"] = pd.to_datetime(acc["Start_Time"], errors="coerce")
acc["End_Time"]   = pd.to_datetime(acc["End_Time"], errors="coerce")

weather["StartTime(UTC)"] = pd.to_datetime(weather["StartTime(UTC)"], errors="coerce")
weather["EndTime(UTC)"]   = pd.to_datetime(weather["EndTime(UTC)"], errors="coerce")

# Remove invalid rows
acc = acc.dropna(subset=["Start_Time","Start_Lat","Start_Lng"])
weather = weather.dropna(subset=["StartTime(UTC)","LocationLat","LocationLng"])

# TIME FEATURES
acc["Date"] = acc["Start_Time"].dt.date
acc["Hour"] = acc["Start_Time"].dt.hour
acc["DayOfWeek"] = acc["Start_Time"].dt.dayofweek
acc["Month"] = acc["Start_Time"].dt.month

acc["Is_Weekend"] = acc["DayOfWeek"].isin([5,6]).astype(int)
acc["Is_Night"] = acc["Hour"].isin([0,1,2,3,4,5]).astype(int)
acc["Is_Rush_Hour"] = acc["Hour"].isin([7,8,9,16,17,18]).astype(int)

acc["Is_Winter"] = acc["Month"].isin([12,1,2]).astype(int)
acc["Is_Summer"] = acc["Month"].isin([6,7,8]).astype(int)

# Duration feature
acc["Duration_minutes"] = (acc["End_Time"] - acc["Start_Time"]).dt.total_seconds() / 60
acc["Duration_minutes"] = acc["Duration_minutes"].fillna(acc["Duration_minutes"].median())

num_cols = [
    "Temperature(F)","Humidity(%)","Visibility(mi)",
    "Wind_Speed(mph)","Precipitation(in)"
]

for col in num_cols:
    acc[col] = pd.to_numeric(acc[col], errors="coerce")

acc[num_cols] = acc[num_cols].fillna(acc[num_cols].median())

# WEATHER-BASED ENGINEERED FEATURES
acc["Freezing_Temp"] = (acc["Temperature(F)"] < 32).astype(int)
acc["High_Wind"] = (acc["Wind_Speed(mph)"] > 25).astype(int)
acc["Low_Visibility"] = (acc["Visibility(mi)"] < 3).astype(int)

# NORMALIZE WEATHER CONDITIONS
def normalize_weather_condition(x):
    x = str(x).upper()
    if "CLEAR" in x or "FAIR" in x:
        return "CLEAR"
    if "CLOUD" in x or "OVERCAST" in x:
        return "CLOUDY"
    if "RAIN" in x or "DRIZZLE" in x or "SHOWER" in x:
        return "RAIN"
    if "SNOW" in x:
        return "SNOW"
    if "FOG" in x or "MIST" in x or "HAZE" in x:
        return "FOG"
    if "THUNDER" in x or "T-STORM" in x or "STORM" in x:
        return "STORM"
    if any(word in x for word in ["FREEZING","ICE","SLEET","WINTRY","PELLET"]):
        return "WINTRY_MIX"
    if any(word in x for word in ["DUST","SAND","SMOKE","ASH"]):
        return "DUST_SMOKE"
    return "OTHER"

acc["Weather_Condition_Clean"] = acc["Weather_Condition"].apply(normalize_weather_condition)

acc["Rain_At_Night"] = ((acc["Weather_Condition_Clean"] == "RAIN") & (acc["Is_Night"] == 1)).astype(int)
acc["Snow_At_Night"] = ((acc["Weather_Condition_Clean"] == "SNOW") & (acc["Is_Night"] == 1)).astype(int)

# WEATHER EVENT NORMALIZATION
weather["Type"] = weather["Type"].astype(str).str.upper()

weather_map = {
    "COLD":"COLD","FOG":"FOG","HAIL":"HAIL","PRECIPITATION":"PRECIPITATION",
    "RAIN":"RAIN","SNOW":"SNOW","STORM":"STORM"
}

weather["Weather_Event"] = weather["Type"].map(lambda x: weather_map.get(x, "OTHER"))

# ROUND COORDS FOR MERGE
acc["Lat_round"] = acc["Start_Lat"].round(1)
acc["Lng_round"] = acc["Start_Lng"].round(1)

weather["Lat_round"] = weather["LocationLat"].round(1)
weather["Lng_round"] = weather["LocationLng"].round(1)

# SAVE CLEANED DATA
acc.to_csv("clean_accidents.csv", index=False)
weather.to_csv("clean_weather.csv", index=False)

print("Cleaning completed!")
