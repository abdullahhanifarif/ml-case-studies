import pandas as pd
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess(self, y,x):
        # Selecting the target variable (Y)
        Y = self.df[y]

        # Selecting the feature variables (X)
        X = self.df[x]

        # Splitting the data into training and testing sets (80:20)
        return train_test_split(
                X, Y,
                test_size=0.2,
                random_state=42
            )


    def preprocess_with_dummies(self, y, x, dummy_columns):
        # Selecting the target variable (Y)
        Y = self.df[y]

        # Selecting the feature variables (X)
        X = self.df[x].copy()

        # Converting specified categorical columns into dummy/one-hot encoded variables
        X = pd.get_dummies(
            X,
            columns=dummy_columns,
            drop_first=True,
            dtype=int
        )

        # Splitting the data into training and testing sets (80:20)
        return train_test_split(
            X,
            Y,
            test_size=0.2,
            random_state=42
        )



