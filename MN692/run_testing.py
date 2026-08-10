import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load cleaned test data
print("Loading test data...")
df = pd.read_csv('testing_results_clean.csv')
print(f"Loaded {len(df)} rows.")

# 2. Load models (make sure they are in the same folder or update path)
print("Loading models...")
iso = joblib.load('isolation_forest_model.pkl')
lof = joblib.load('lof_model.pkl')

# 3. Prepare features
X = df[['duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state']].values
y_true = df['label_enc'].values

# 4. Predict
print("Running predictions...")
iso_pred_raw = iso.predict(X)  # 1 for normal, -1 for anomaly
lof_pred_raw = lof.predict(X)  # 1 for normal, -1 for anomaly

# Convert to 0 (Benign) and 1 (Malicious) to match labels
iso_pred = [1 if p == -1 else 0 for p in iso_pred_raw]
lof_pred = [1 if p == -1 else 0 for p in lof_pred_raw]

# Ensemble: If EITHER model says malicious (1), flag it.
ensemble = [1 if iso_pred[i]==1 or lof_pred[i]==1 else 0 for i in range(len(iso_pred))]

# 5. Metrics
print("\n=== TESTING SCENARIO RESULTS ===")
print(classification_report(y_true, ensemble, target_names=['Benign', 'Malicious']))
cm = confusion_matrix(y_true, ensemble)
print("Confusion Matrix:")
print(cm)

# 6. Save final results
df['iso_pred'] = iso_pred
df['lof_pred'] = lof_pred
df['ensemble_pred'] = ensemble
df.to_csv('testing_final_results.csv', index=False)
print("\n✅ Results saved to testing_final_results.csv")

# 7. Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Malicious'], 
            yticklabels=['Benign', 'Malicious'])
plt.title('Confusion Matrix - Testing Scenario')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('testing_confusion_matrix.png', dpi=150)
print("✅ Confusion matrix saved as testing_confusion_matrix.png")

# 8. Plot Anomaly Score Distribution
plt.figure(figsize=(10, 4))
scores = iso.score_samples(X)
plt.hist(scores[ensemble == 0], bins=30, alpha=0.5, label='Benign (Predicted)', color='green')
plt.hist(scores[ensemble == 1], bins=30, alpha=0.5, label='Malicious (Predicted)', color='red')
plt.xlabel('Anomaly Score (lower = more anomalous)')
plt.ylabel('Frequency')
plt.title('Anomaly Score Distribution - Testing Scenario')
plt.legend()
plt.savefig('testing_anomaly_scores.png', dpi=150)
print("✅ Anomaly score chart saved as testing_anomaly_scores.png")
print("\n🎯 TESTING TASK COMPLETE!")