import polars as pl

REVIEWS_DATA_PATH = "data/reviews.csv"

data_reviews = pl.read_csv(REVIEWS_DATA_PATH)

print(f"{data_reviews.shape=}")

print(f"{data_reviews.head()=}")
