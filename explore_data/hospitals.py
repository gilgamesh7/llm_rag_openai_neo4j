import polars as pl

HOSPITAL_DATA_PATH = "data/hospitals.csv"

data_hospitals = pl.read_csv(HOSPITAL_DATA_PATH)

print(f"{data_hospitals.shape=}")

print(f"{data_hospitals.head()=}")