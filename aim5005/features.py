import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully! 
        return (x-self.minimum)/(diff_max_min)
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
# References:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# - Machine Learning Mastery: https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/
# - Towards Data Science: https://towardsdatascience.com/all-about-feature-scaling-bcc0ad75cb35

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. 
        If it can't be cast, raise an error.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        assert isinstance(x, np.ndarray), "Expected the input to be a NumPy array"
        return x

    def fit(self, x: np.ndarray) -> None:
        """
        Compute the mean and standard deviation for scaling.
        """
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

    def transform(self, x: np.ndarray) -> list:
        """
        Standardize the given vector.
        """
        x = self._check_is_array(x)

        # Prevent division by zero by replacing zero std with 1
        adjusted_std = np.where(self.std == 0, 1, self.std)

        return (x - self.mean) / adjusted_std

    def fit_transform(self, x: list) -> np.ndarray:
        """
        Compute the mean and std, then standardize the input.
        """
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
# Reference:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html 
# - Machine Learning Mastery: https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
# - Towards Data Science: https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63

class LabelEncoder:
    def __init__(self):
        self.classes_ = None
    
    def fit(self, y: List) -> 'LabelEncoder':
        """
        Fit the encoder by learning the unique labels.
        """
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        return self
    
    def transform(self, y: List) -> np.ndarray:
        """
        Transform the labels into integers based on the learned classes.
        """
        y = np.array(y)
        
        if self.classes_ is None:
            raise ValueError("This LabelEncoder instance is not fitted yet. Call 'fit' with appropriate data.")
        
        label_to_int = {label: idx for idx, label in enumerate(self.classes_)}
        return np.array([label_to_int[label] for label in y])
    
    def fit_transform(self, y: List) -> np.ndarray:
        """
        Fit the encoder and transform the labels in one step.
        """
        self.fit(y)
        return self.transform(y)
