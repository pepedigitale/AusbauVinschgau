import pandas as pd

df = pd.read_csv("2019viaggi_check.csv", sep=";", encoding="cp1252")

delay = pd.to_numeric(df.iloc[:, 6], errors="coerce")

# All trains are included in the denominator
share_delayed = (delay.fillna(0) > 0).mean()

# Only trains with an actual recorded delay are used here
delayed = delay[delay > 0]

average_delay = delayed.mean()
std_delay = delayed.std()

print(f"Share of delayed trains: {share_delayed:.2%}")
print(f"Average delay (delayed trains only): {average_delay:.2f} min")
print(f"Standard deviation (delayed trains only): {std_delay:.2f} min")