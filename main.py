
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_data(path):
    """Load customer dataset."""
    return pd.read_csv(path)


def preprocess_data(df):
    """Remove unnecessary columns."""

    X = df.drop(columns=["ID"])

    return X


def scale_features(X):
    """Standardize features."""

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return scaler, X_scaled


def train_model(X_scaled, n_clusters=4):
    """Train K-Means model."""

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    return model, labels


def save_results(df, labels):

    df["Cluster"] = labels

    df.to_csv(
        "results/Customer_Segmentation_Results.csv",
        index=False
    )

    print("Results saved successfully.")


def main():

    print("=" * 60)
    print("CUSTOMER SEGMENTATION USING K-MEANS")
    print("=" * 60)

    df = load_data("data/segmentation data.csv")

    X = preprocess_data(df)

    scaler, X_scaled = scale_features(X)

    model, labels = train_model(X_scaled)

    save_results(df, labels)

    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()