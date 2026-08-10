import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

print("Loading training data...")
df = pd.read_csv('train_data.csv')
X = df[['duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state']].values
print(f"Training on {len(X)} rows...")

print("Training Isolation Forest...")
iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso.fit(X)

print("Training Local Outlier Factor...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
lof.fit(X)

print("Saving models...")
joblib.dump(iso, 'iso_split.pkl')
joblib.dump(lof, 'lof_split.pkl')

print("✅ Models trained on split data!")