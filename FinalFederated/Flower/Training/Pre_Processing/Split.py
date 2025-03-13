import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
dataset_path = ".\\trainingfeatures.csv"
df = pd.read_csv(dataset_path)

# Split dataset into two non-overlapping sets (50% for each client)
df_client1, df_client2 = train_test_split(df, test_size=0.5, random_state=42, shuffle=True)

# Save each split dataset
df_client1.to_csv(".\\client_1.csv", index=False)
df_client2.to_csv(".\\client_2.csv", index=False)

print("✅ Dataset split successfully into client_1.csv and client_2.csv")
