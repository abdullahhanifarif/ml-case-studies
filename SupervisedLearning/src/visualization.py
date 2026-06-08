import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Visualization:

    def compare_models(self, results, type):

        model_names = list(results.keys())
        r2_scores = [results[m]['R2'] for m in model_names]

        plt.figure(figsize=(8, 5))

        bars = plt.bar(model_names, r2_scores)

        plt.title(f"Model Comparison (R2 Score) - {type}")
        plt.ylabel("R2 Score")
        plt.ylim(0, 1)

        # Tampilkan nilai di atas bar
        for bar in bars:

            yval = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width()/2,
                yval + 0.01,
                round(yval, 4),
                ha='center'
            )
        plt.tight_layout()
        plt.show()

   
