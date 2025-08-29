import pandas as pd

df = pd.read_parquet("output.parquet")
total = len(df)

label_groups = {}
for col in df.columns:
    if "_" in col:
        prefix = col.split("_")[0]
        if col.startswith(prefix + "_") and df[col].dropna().isin([0, 1]).all():
            label_groups.setdefault(prefix, []).append(col)

print("Total records:", total)
for group, cols in label_groups.items():
    counts = df[cols].sum()
    percents = (counts / total * 100).round(2)
    print(f"\n{group.capitalize()} label distribution:")
    for col in cols:
        print(f"{col}: {counts[col]} ({percents[col]}%)")
