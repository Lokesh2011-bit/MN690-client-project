import pandas as pd
import os

print("=" * 60)
print("MN692 - TESTING SCENARIO (Processing IoT-23 Data)")
print("=" * 60)

# Load your IoT-23 file
file_path = r"C:\Users\manik\OneDrive\Desktop\MN690.csv"
print(f"\n[1] Loading: {file_path}")

try:
    df = pd.read_csv(file_path, low_memory=False)
    print(f"    ✓ Loaded {len(df)} rows")
    print(f"    ✓ Columns: {list(df.columns)[:8]}...")
except FileNotFoundError:
    print("    ✗ File not found! Check the path.")
    exit()

# Check for label column
print("\n[2] Checking labels...")
if 'label' in df.columns:
    labels = df['label'].value_counts()
    print(f"    ✓ Label distribution:")
    for label, count in labels.items():
        print(f"        - {label}: {count}")
    
    # Create clean label column
    df['label_clean'] = df['label'].apply(
        lambda x: 'Malicious' if 'Malicious' in str(x) or 'Attack' in str(x) or 'Botnet' in str(x) else 'Benign'
    )
    
    benign = (df['label_clean'] == 'Benign').sum()
    malicious = (df['label_clean'] == 'Malicious').sum()
    print(f"\n[3] Clean labels:")
    print(f"    ✓ Benign: {benign}")
    print(f"    ✓ Malicious: {malicious}")
    print(f"    ✓ Attack rate: {malicious/(benign+malicious)*100:.1f}%")

# Save clean data
output_path = r"C:\Users\manik\OneDrive\Desktop\testing_results_clean.csv"
df.to_csv(output_path, index=False)
print(f"\n[4] Clean data saved to: {output_path}")

# Summary
summary = f"""
=== MN692 TESTING SCENARIO RESULTS ===
Source: IoT-23 Dataset (MN690.csv)
Total flows analysed: {len(df)}
Benign flows: {benign}
Malicious flows: {malicious}
Attack rate: {malicious/(benign+malicious)*100:.1f}%
Attack types included: Botnet, C&C, DDoS, Port Scan, SSH Brute Force
"""
print(summary)

with open(r"C:\Users\manik\OneDrive\Desktop\testing_summary.txt", "w") as f:
    f.write(summary)

print("=" * 60)
print("✅ DONE! Clean data ready for Navoda")
print("=" * 60)