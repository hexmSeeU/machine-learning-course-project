import numpy as np

from src.data import load_wdbc_from_zip, standardize_train_test, train_test_split


def test_load_wdbc_from_zip_shape_and_labels():
    X, y, feature_names = load_wdbc_from_zip("data_a1.zip")

    assert X.shape == (569, 27)
    assert y.shape == (569,)
    assert len(feature_names) == 27
    assert set(np.unique(y)) == {-1.0, 1.0}


def test_train_test_split_is_reproducible():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([1, -1] * 10, dtype=float)

    split_a = train_test_split(X, y, test_size=0.25, seed=7)
    split_b = train_test_split(X, y, test_size=0.25, seed=7)

    for a, b in zip(split_a, split_b):
        np.testing.assert_array_equal(a, b)
    assert split_a[0].shape[0] == 15
    assert split_a[1].shape[0] == 5


def test_standardize_uses_train_statistics_only():
    X_train = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    X_test = np.array([[7.0, 22.0]])

    X_train_std, X_test_std, mean, scale = standardize_train_test(X_train, X_test)

    np.testing.assert_allclose(mean, [3.0, 14.0])
    np.testing.assert_allclose(scale, [np.sqrt(8.0 / 3.0), np.sqrt(32.0 / 3.0)])
    np.testing.assert_allclose(X_train_std.mean(axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(X_train_std.std(axis=0), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(X_test_std, (X_test - mean) / scale)
