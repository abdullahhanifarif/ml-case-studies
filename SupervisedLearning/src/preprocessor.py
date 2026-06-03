import pandas as pd
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess(self, y,x):
        Y = self.df[y]
        X = self.df[x]
        return train_test_split(
                X, Y,
                test_size=0.2,
                random_state=42
            )
    
    def preprocess_with_dummies(self, y, x, dummy_columns):
        Y = self.df[y]
        X = self.df[x].copy()
        X = pd.get_dummies(
            X,
            columns=dummy_columns,
            drop_first=True,
            dtype=int
        )

        return train_test_split(
            X,
            Y,
            test_size=0.2,
            random_state=42
        )



