import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading test data...")
df = pd.read_csv('test_data.csv')
X = df[['duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state']].values
y_true = df['label_enc'].values
print(f"Testing on {len(X)} rows...")

print("Loading models...")
iso = joblib.load('iso_split.pkl')
lof = joblib.load('lof_split.pkl')

print("Running predictions...")
iso_pred = [1 if p == -1 else 0 for p in iso.predict(X)]
lof_pred = [1 if p == -1 else 0 for p in lof.predict(X)]

# Ensemble: If EITHER model says malicious, flag it
ensemble = [1 if iso_pred[i]==1 or lof_pred[i]==1 else 0 for i in range(len(iso_pred))]

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, ensemble, target_names=['Benign', 'Malicious']))

cm = confusion_matrix(y_true, ensemble)
print("\n=== CONFUSION MATRIX ===")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
plt.title('Confusion Matrix – Proper Split Test')
plt.savefig('split_confusion_matrix.png', dpi=150)
print("✅ Confusion matrix saved as split_confusion_matrix.png")

# Save results
df['ensemble_pred'] = ensemble
df.to_csv('split_test_results.csv', index=False)
print("✅ Results saved to split_test_results.csv")