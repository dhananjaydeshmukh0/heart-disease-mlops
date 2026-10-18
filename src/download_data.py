
from pathlib import Path

from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # UCI Heart Disease dataset, repository ID 45
    dataset = fetch_ucirepo(id=45)

    features = dataset.data.features
    targets = dataset.data.targets

    if targets is None or targets.empty:
        raise ValueError("The dataset does not contain target labels.")

    # Original UCI target: 0 = absence, 1–4 = presence
    # Binary assignment target: 0 = absence, 1 = presence
    target = targets.iloc[:, 0].apply(
        lambda value: 0 if value == 0 else 1
    )


    data = features.copy()
    data["target"] = target

    output_path = RAW_DATA_DIR / "heart_disease.csv"
    data.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Dataset shape: {data.shape}")
    print(f"Target distribution:\n{data['target'].value_counts()}")


if __name__ == "__main__":
    main()
