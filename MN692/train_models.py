import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

print("=" * 50)
print("TRAINING MODELS")
print("=" * 50)

print("\n[1] Loading cleaned data...")
df = pd.read_csv('testing_results_clean.csv')
X = df[['duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state']].values
print(f"    Loaded {len(X)} rows for training.")

print("\n[2] Training Isolation Forest...")
iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso.fit(X)
print("    Isolation Forest trained.")

print("\n[3] Training Local Outlier Factor...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
lof.fit(X)
print("    LOF trained.")

print("\n[4] Saving models...")
joblib.dump(iso, 'isolation_forest_model.pkl')
joblib.dump(lof, 'lof_model.pkl')
print("    Models saved as:")
print("    - isolation_forest_model.pkl")
print("    - lof_model.pkl")

print("\n" + "=" * 50)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 50)