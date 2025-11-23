from datasets import load_dataset
import pandas as pd

print("Downloading dataset...")
ds = load_dataset("Tobi-Bueck/customer-support-tickets")

df = ds["train"].to_pandas()
df.to_csv("data/raw_tickets.csv", index=False)

print("Saved dataset to data/raw_tickets.csv")
print(df.head())
