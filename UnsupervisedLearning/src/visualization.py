import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self):
        pass

    def plot_clusters(self, X_original, labels, centers=None, title="Clustering"):
        plt.figure(figsize=(8, 6))
        plt.scatter(
            X_original.iloc[:, 0],
            X_original.iloc[:, 1],
            c=labels
        )

        if centers is not None:
            plt.scatter(
                centers[:, 0],
                centers[:, 1],
                marker='X',
                s=250
            )

        plt.xlabel(X_original.columns[0])
        plt.ylabel(X_original.columns[1])
        plt.title(title)

        plt.show()
