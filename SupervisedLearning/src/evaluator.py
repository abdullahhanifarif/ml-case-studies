import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Evaluator:

    def evaluate_regression(self, models, X_test, y_test):
        results = {}
        for name, model in models.items():
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)

            results[name] = {
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "Prediction": pred
            }
        return results
    

    def evaluate_classification(self, models, X_test, y_test):
        results = {}
        for name, model in models.items():
            pred = model.predict(X_test)
            accuracyscore = accuracy_score(y_test, pred)
            classificationreport = classification_report(y_test, pred)
            confusionmatrix = confusion_matrix(y_test, pred)

            results[name] = {
                "accuracy_score": accuracyscore,
                "classification_report": classificationreport,
                "confusion_matrix" : confusionmatrix,
                "Prediction": pred         
            }

        return results
