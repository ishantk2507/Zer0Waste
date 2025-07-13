import pandas as pd
from sklearn.utils import resample

# 1. Load your data
df = pd.read_csv("data/mock_inventory.csv")

# 2. Separate majority and minority classes
df_major = df[df.spoilage == 0]
df_minor = df[df.spoilage == 1]

# 3. Upsample the minority class to match the majority size
df_minor_up = resample(
    df_minor,
    replace=True,                  # sample with replacement
    n_samples=len(df_major),       # match number in majority
    random_state=42
)

# 4. Combine back into a single DataFrame
df_balanced = pd.concat([df_major, df_minor_up]).sample(frac=1, random_state=42).reset_index(drop=True)

# 5. (Optional) Verify the new class proportions
print(df_balanced.spoilage.value_counts())

# 6. Write out the balanced CSV
df_balanced.to_csv("data/mock_inventory_balanced.csv", index=False)
print("Balanced dataset written to data/mock_inventory_balanced.csv")
