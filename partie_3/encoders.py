from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Computes the haversine distance between two GPS points.
    Returns a copy of the DataFrame X with only one column: 'distance'.
    """
    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        lat1 = np.radians(X[self.start_lat])
        lon1 = np.radians(X[self.start_lon])
        lat2 = np.radians(X[self.end_lat])
        lon2 = np.radians(X[self.end_lon])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c
        X_ = X.copy()
        X_['distance'] = km
        return X_[['distance']]

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    def __init__(self,time_column = 'pickup_datetime' ):
       self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self.time_column] = pd.to_datetime(X_[self.time_column])
        X_["dow"] = X_[self.time_column].dt.dayofweek
        X_["hour"] = X_[self.time_column].dt.hour
        X_["month"] = X_[self.time_column].dt.month
        X_["year"] = X_[self.time_column].dt.year
        return X_[['dow', 'hour', 'month', 'year']]