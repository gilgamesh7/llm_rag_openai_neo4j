import polars as pl

PHYSICIANS_DATA_PATH = "data/physicians.csv"

data_physicians = pl.read_csv(PHYSICIANS_DATA_PATH)

print(f"{data_physicians.shape=}")

print(f"{data_physicians.head()=}")