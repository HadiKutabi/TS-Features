from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional
from pandas import DataFrame, Series


class Lag(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_periods: int = 1,
                 shift_columns: Optional[List[str]] = None,
                 drop_na_rows: bool = False,
                 drop_original_columns: bool = False,
                 generate_one_col_pro_period=False
                 ) -> None:

        """
        Generates lag columns by shifting.

        Parameters
        ----------
        :param n_periods: int
            The number of periods to shift per column. Must be positive.
        :param shift_columns: list
            The column names to shift
        :param drop_na_rows: bool
            Upon shifting, some rows will have NaNs. When True, rows with NaNs will be dropped.
        :param drop_original_columns: bool
            When True, the columns from shift_columns will be dropped after transformation.
        :param generate_one_col_pro_period: bool:
            whether to shift the data n_periods once (generating only one column), or to generate
            multiple columns (one per period step), ascending from 1 to n_periods.
        """
        # todo saving last rows from training data and using them in testing data

        assert n_periods > 0, "n_periods must be greater than 0!"

        self.n_periods = n_periods
        self.shift_columns = shift_columns
        self.drop_na_rows = drop_na_rows
        self.drop_original_columns = drop_original_columns
        self.only_one_column_for_periods = generate_one_col_pro_period

        self.new_columns = []
        self.column_dtype_map = {}
        self.lag_name_suffix = "_LAG-{}"

    def fit(self,
            X: DataFrame,
            y: Optional = None
            ) -> "Lag":
        if self.shift_columns is None:
            self.shift_columns = X.columns
        return self

    def transform(self,
                  X: DataFrame,
                  y: Optional = None
                  ) -> DataFrame:
        for original_column in self.shift_columns:
            if self.only_one_column_for_periods is False:
                for lag in range(1, self.n_periods + 1):
                    new_column_name = original_column + self.lag_name_suffix.format(lag)
                    X.loc[:, new_column_name] = self._generate_lag_column_and_update_dtype_map(
                        original_col_name=original_column,
                        new_column_name=new_column_name,
                        period=lag,
                        X=X
                    )
            elif self.only_one_column_for_periods is True:
                new_column_name = original_column + self.lag_name_suffix.format(self.n_periods)
                X.loc[:, new_column_name] = self._generate_lag_column_and_update_dtype_map(
                    original_col_name=original_column,
                    new_column_name=new_column_name,
                    period=self.n_periods,
                    X=X
                )

        if self.drop_na_rows is True:
            X.dropna(inplace=True)
            X = self._cast_cols_dtypes_to_original(X)  # we can only cast types when column has no NaNs
        if self.drop_original_columns:
            X.drop(columns=self.shift_columns, inplace=True)
        return X

    def _generate_lag_column_and_update_dtype_map(self,
                                                  original_col_name: str,
                                                  new_column_name: str,
                                                  period: int,
                                                  X: DataFrame
                                                  ) -> Series:

        assert period > 0, "A period must be greater than 0!"
        self.new_columns.append(new_column_name)
        self.column_dtype_map[new_column_name] = X.loc[:, original_col_name].dtype
        return X.loc[:, original_col_name].shift(period)

    def _cast_cols_dtypes_to_original(self,
                                      X: DataFrame
                                      ) -> DataFrame:

        for col, _type in self.column_dtype_map.items():
            X[col] = X[col].astype(_type, copy=False)
        return X


def lag_test() -> None:
    import pandas as pd
    X = pd.DataFrame(dict(date=["2022-09-18",
                                "2022-09-19",
                                "2022-09-20",
                                "2022-09-21",
                                "2022-09-22"],
                          x1=[1, 2, 3, 4, 5],
                          x2=[6, 7, 8, 9, 10]
                          ))

    lag_transformer = Lag(n_periods=2,
                          shift_columns=["x1", "x2"],
                          drop_na_rows=False,
                          drop_original_columns=False,
                          generate_one_col_pro_period=True)
    X_with_lag = lag_transformer.fit_transform(X)
    print(X_with_lag)


if __name__ == "__main__":
    lag_test()
