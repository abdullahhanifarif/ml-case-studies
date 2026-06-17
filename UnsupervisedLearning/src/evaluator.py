from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

class Evaluator:
    def __init__(self):
        self.results = {}

    def evaluate_models(self, X, models):
        self.results = {}

        for name, model in models.items():
            labels = model.labels_

            if len(set(labels)) <= 1:
                self.results[name] = {
                    "Silhouette Score": None,
                    "Davies-Bouldin Index": None,
                    "Calinski-Harabasz Score": None
                }
                continue

            self.results[name] = {
                "Silhouette Score": round(
                    silhouette_score(X, labels), 4
                ),
                "Davies-Bouldin Index": round(
                    davies_bouldin_score(X, labels), 4
                ),
                "Calinski-Harabasz Score": round(
                    calinski_harabasz_score(X, labels), 4
                )
            }

        return self.results
