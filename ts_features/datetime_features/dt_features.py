from typing import List, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from pandas import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self,
                 dt_columns: List[str],
                 exclude_features: Optional[List[str]] = None,
                 drop_original_columns: bool = False
                 ) -> None:
        self.dt_columns = dt_columns
        self.exclude_features = exclude_features
        self.drop_original_columns = drop_original_columns

        self.included_features_and_methods_map = {
            "year": self._get_year,
            "month": self._get_month,
            "day-of-month": self._get_day_of_month,
            "day-of-year": self._get_day_of_year,
            "day-of-week": self._get_day_of_week,
            "is-weekend": self._get_is_weekend,
            "hour": self._get_hour,
            "minute": self._get_minute,
            "second": self._get_second
        }

        self.handle_excluded_features()

    def handle_excluded_features(self) -> None:

        if self.exclude_features is None:
            return
        features_to_pop = []
        for feature in self.exclude_features:
            assert feature in self.included_features_and_methods_map.keys(), """
            The specified feature {} for exclusion is not permitted. \n 
            permitted values are {}
            """.format(feature, list(self.included_features_and_methods_map.keys()))

            features_to_pop.append(feature)
        for feature in features_to_pop:
            self.included_features_and_methods_map.pop(feature)

    def fit(self,
            X: DataFrame,
            y: Optional = None
            ) -> 'DateTimeFeatures':
        return self

    def transform(self,
                  X: DataFrame,
                  y: Optional = None
                  ) -> DataFrame:

        for original_column in self.dt_columns:
            print()
            for new_feature_name, method in self.included_features_and_methods_map.items():
                new_column_name = self._get_new_feature_name(original_column, new_feature_name)
                X.loc[:, new_column_name] = X.loc[:, original_column].map(method)

        if self.drop_original_columns is True:
            X.drop(columns=self.dt_columns, inplace=True)

        return X

    @staticmethod
    def _get_new_feature_name(original_column_name: str,
                              new_feature_name: str
                              ) -> str:

        return original_column_name + "_{}".format(new_feature_name.upper())

    @staticmethod
    def _get_year(dt: Timestamp) -> int:
        return dt.year

    @staticmethod
    def _get_month(dt: Timestamp) -> int:
        return dt.month

    @staticmethod
    def _get_day_of_month(dt: Timestamp) -> int:
        return dt.day

    @staticmethod
    def _get_day_of_year(dt: Timestamp) -> int:
        return dt.dayofyear

    @staticmethod
    def _get_day_of_week(dt: Timestamp) -> int:
        return dt.dayofweek

    @staticmethod
    def _get_hour(dt: Timestamp) -> int:
        return dt.hour

    @staticmethod
    def _get_minute(dt: Timestamp) -> int:
        return dt.minute

    @staticmethod
    def _get_second(dt: Timestamp) -> int:
        return dt.second

    @staticmethod
    def _get_is_weekend(dt: Timestamp) -> bool:
        dayofweek = dt.dayofweek

        match dayofweek:
            case 5:
                return True
            case 6:
                return True
            case _:
                return False


if __name__ == "__main__":
    import pandas as pd

    dti1 = pd.date_range("15-08-1994 00:00:00", periods=500000, freq="S")
    dti2 = pd.date_range("22-07-2021 00:00:00", periods=500000, freq="S")

    df = DataFrame(
        {
            "dt1": dti1,
            "dt2": dti2,
        }
    )

    DTF = DateTimeFeatures(["dt1", "dt2"], exclude_features=["is-weekend", "second"])
    print(DTF.fit_transform(df))
    print()
