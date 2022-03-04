"""
Test ks_2samp behaviour.
"""

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler


def get_data():
    """Fetch iris dataset."""
    names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    data = pd.read_csv(url, names=names)

    return data


def get_features(data: pd.DataFrame):
    """Generate features."""
    data = data.copy()

    numeric_fields = [
        "sepal-length",
        "sepal-width",
        "petal-length",
        "petal-width",
    ]

    data_numeric = data[numeric_fields]
    scaler = MinMaxScaler()
    scaler.fit(data_numeric)
    data[numeric_fields] = scaler.transform(data_numeric)

    return data[numeric_fields]


if __name__ == "__main__":
    raw_data = get_data()
    features = get_features(raw_data)
    features.to_csv("reference.csv", index=False)
    reference = pd.read_csv("reference.csv")

    for feature_name in features.columns:
        scipy_p_value = ks_2samp(reference[feature_name], features[feature_name])[1]
        print(f"Feature '{feature_name}' p value: {scipy_p_value}")

    # When you save the data to csv and load it again, the floating point representations
    # are changing slightly (~1e-17). Since the two-sample KS test depends on orderings
    # rather than precise numerical, this makes a big difference in the resulting calculation.
    # When you convert to 32-bit, those numerical differences vanish.
    for feature_name in features.columns:
        scipy_p_value = ks_2samp(
            reference[feature_name].astype("float32"),
            features[feature_name].astype("float32"),
        )[1]
        print(f"Feature '{feature_name}' p value: {scipy_p_value}")
