import polars as pl

VISITS_DATA_PATH = "data/visits.csv"

data_visits = pl.read_csv(VISITS_DATA_PATH)

print(f"{data_visits.shape=}")

print(f"{data_visits.head()=}")
