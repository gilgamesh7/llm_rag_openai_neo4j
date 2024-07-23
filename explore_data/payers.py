import polars as pl

PAYERS_DATA_PATH = "data/payers.csv"

data_payers = pl.read_csv(PAYERS_DATA_PATH)

print(f"{data_payers.shape=}")

print(f"{data_payers.head()=}")