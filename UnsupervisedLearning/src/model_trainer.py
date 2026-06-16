from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


class ModelTrainer:
    def __init__(self):
        self.models = {}

    def elbow_method(self, X, max_k=15):
        """
        Menghitung WCSS untuk Elbow Method
        """
        wcss = []

        for k in range(1, max_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                init = 'k-means++'
            )

            kmeans.fit(X)

            wcss.append(kmeans.inertia_)

        return wcss
    
    def train_clustering(self, x, n):
        self.models = {
            "KMeans": KMeans(
                n_clusters=n
            ),
            "Agglomerative Clustering": AgglomerativeClustering(
                n_clusters=n
            ),
            "DBSCAN": DBSCAN(
                eps=0.3, 
                min_samples=3
            )
        }

        for model in self.models.values():
            model.fit(x)

        return self.models
    
    
