import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading data...")
df = pd.read_csv('testing_results_clean.csv')
print(f"Total rows: {len(df)}")

# Split: 70% train, 30% test
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label_enc'])

print(f"Training set: {len(train_df)} rows")
print(f"Testing set: {len(test_df)} rows")

train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print("✅ Train/Test split complete!")
print("Files saved: train_data.csv, test_data.csv")