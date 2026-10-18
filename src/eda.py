
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"
OUTPUT_DIR = PROJECT_ROOT / "screenshots" / "eda"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Basic dataset information
    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nTarget distribution:\n", df["target"].value_counts())

    # 1. Target class balance
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Target Distribution")
    plt.xlabel("Target (0 = Absence, 1 = Presence)")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=150)
    plt.close()

    # 2. Numeric feature distributions
    df.hist(figsize=(15, 12), bins=20, edgecolor="black")
    plt.suptitle("Heart Disease Feature Distributions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150)
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\nEDA plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
