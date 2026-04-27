import pandas as pd
import os

print("Training started...")

data_path = "data"

for file in os.listdir(data_path):
    if file.endswith("_ekg.csv"):
        file_path = os.path.join(data_path, file)
        print("Reading:", file)

        df = pd.read_csv(file_path)
        print(df.head())
        break