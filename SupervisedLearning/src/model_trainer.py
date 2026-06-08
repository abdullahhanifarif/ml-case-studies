from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier



class ModelTrainer:
    def __init__(self):
        self.models = {}

    def train_regression(self, X_train, y_train):
        self.models = {
            "Linear Regression": LinearRegression(),

            "Decision Tree": DecisionTreeRegressor(
                max_depth=5,
                min_samples_split=10,
                random_state=42
            ),

            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),

            "XGBoost": XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )
        }

        for model in self.models.values():
            model.fit(X_train, y_train)

        return self.models


  
    
    def train_classification(self, X_train, y_train):
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),

            "DecisionTreeClassifier": DecisionTreeClassifier(
                max_depth=5, 
                random_state=42
            )
        }

        for model in self.models.values():
            model.fit(X_train, y_train)

        return self.models
