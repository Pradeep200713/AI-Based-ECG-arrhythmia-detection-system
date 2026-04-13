import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Training started...")

data_path = "data"
all_data = []

# Read all CSV files
for file in os.listdir(data_path):
    if file.endswith(".csv"):
        file_path = os.path.join(data_path, file)
        df = pd.read_csv(file_path)
        all_data.append(df)

# Combine all data
data = pd.concat(all_data)

print("Data loaded successfully")

# Assume last column is target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained!")

# Save model
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/model.pkl", "wb"))

print("Model saved in models/model.pkl")
