from aim5005.features import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import unittest
from unittest.case import TestCase

### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    # TODO: Add a test of your own below this line

    def test_standard_scaler_with_negative_values(self):
        scaler = StandardScaler()
        data = [[-10, -5], [-5, -1], [0, 0], [5, 2], [10, 7]]
        scaler.fit(data)
        result = scaler.transform(data)
        expected_mean = np.mean(data, axis=0)  # [-0.2, 0.2]
        expected_std = np.std(data, axis=0)    # [7.744, 3.365]
        expected_result = (np.array(data) - expected_mean) / expected_std
        assert np.allclose(result, expected_result, atol=1e-6), \
            f"Expected {expected_result}, but got {result}"

    def test_label_encoder_basic(self):
        encoder = LabelEncoder()
        data = ['cat', 'dog', 'dog', 'cat', 'fish']
        encoded = encoder.fit_transform(data)
        expected_classes = ['cat', 'dog', 'fish']
        expected_encoding = [0, 1, 1, 0, 2]
        assert (encoder.classes_ == expected_classes).all(), \
            f"Expected classes: {expected_classes}, but got: {encoder.classes_}"
        assert (encoded == expected_encoding).all(), \
            f"Expected encoding: {expected_encoding}, but got: {encoded}"

    def test_label_encoder_reverse_mapping(self):
        encoder = LabelEncoder()
        data = ['apple', 'banana', 'apple', 'orange', 'banana']
        encoder.fit(data)
        encoded = encoder.transform(data)
        reverse_mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
        assert all(reverse_mapping[label] == encoded[idx] for idx, label in enumerate(data)), \
            f"Mapping failed! Encoded: {encoded}, Reverse mapping: {reverse_mapping}"

    def test_label_encoder_fitted(self):
        encoder = LabelEncoder()
        data = ['apple', 'banana', 'orange']
        encoder.fit(data)
        try:
            encoder_invalid = LabelEncoder()
            encoder_invalid.transform(['apple', 'banana'])
            assert False, "Expected ValueError, but no error was raised."
        except ValueError as e:
            assert str(e) == "This LabelEncoder instance is not fitted yet. Call 'fit' with appropriate data.", \
                f"Unexpected error message: {e}"


if __name__ == '__main__':
    unittest.main()