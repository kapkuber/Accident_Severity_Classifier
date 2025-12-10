import pandas as pd

acc = pd.read_csv("clean_accidents.csv", parse_dates=["Start_Time","End_Time"])
weather = pd.read_csv("clean_weather.csv",
                      parse_dates=["StartTime(UTC)","EndTime(UTC)"])

weather = weather.rename(columns={
    "StartTime(UTC)": "Weather_Start",
    "EndTime(UTC)": "Weather_End"
})

weather = weather.sort_values("Weather_Start")

merged = pd.merge_asof(
    acc.sort_values("Start_Time"),
    weather,
    left_on="Start_Time",
    right_on="Weather_Start",
    by=["Lat_round","Lng_round"],
    tolerance=pd.Timedelta("3h"),
    direction="backward",
    suffixes=("_acc","_wx")
)

merged["Weather_Event"] = merged["Weather_Event"].fillna("NONE")
merged["Severity_wx"] = merged["Severity_wx"].fillna(0)

valid = (
    (merged["Weather_Event"] != "NONE") &
    (merged["Start_Time"] <= merged["Weather_End"])
)

merged.loc[~valid, "Weather_Event"] = "NONE"
merged.loc[~valid, "Severity_wx"] = 0

merged.to_csv("merged_accident_weather.csv", index=False)

print("Merge complete!")
