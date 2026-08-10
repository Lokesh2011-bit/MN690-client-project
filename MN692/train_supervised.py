import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 50)
print("SUPERVISED LEARNING - TRAINING")
print("=" * 50)

# Load training data
print("\n[1] Loading training data...")
train_df = pd.read_csv('train_data.csv')
X_train = train_df[['duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state']].values
y_train = train_df['label_enc'].values
print(f"Training on {len(X_train)} rows...")

# Load test data
test_df = pd.read_csv('test_data.csv')
X_test = test_df[['duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state']].values
y_test = test_df['label_enc'].values
print(f"Testing on {len(X_test)} rows...")

# Train Random Forest
print("\n[2] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest trained.")

# Predict
print("\n[3] Running predictions...")
y_pred = rf.predict(X_test)

# Results
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))

cm = confusion_matrix(y_test, y_pred)
print("\n=== CONFUSION MATRIX ===")
print(cm)

# Save model
joblib.dump(rf, 'random_forest_model.pkl')
print("\n✅ Model saved as: random_forest_model.pkl")

# Confusion matrix plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
plt.title('Confusion Matrix – Supervised Random Forest')
plt.savefig('supervised_confusion_matrix.png', dpi=150)
print("✅ Confusion matrix saved as supervised_confusion_matrix.png")

# Save predictions
test_df['rf_pred'] = y_pred
test_df.to_csv('supervised_results.csv', index=False)
print("✅ Results saved to supervised_results.csv")

print("\n" + "=" * 50)
print("🎯 SUPERVISED TRAINING COMPLETE!")
print("=" * 50)