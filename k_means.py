import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class Clustering:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.X = None
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.kmeans = None
        self.labels = None
        self.centroids = None
        self.selected_x = None
        self.selected_y = None

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        print(self.df.head())

    def select_features(self, x_label, y_label):
        self.selected_x = x_label
        self.selected_y = y_label
        self.X = self.df[[x_label, y_label]].values

    def scale_features(self):
        self.X_scaled = self.scaler.fit_transform(self.X)

    def plot_elbow(self, max_k=10, save_path='static/img/elbow.png'):
        wcss = []
        K_range = range(1, max_k + 1)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(self.X_scaled)
            wcss.append(km.inertia_)
        plt.figure(figsize=(6,4))
        plt.plot(K_range, wcss, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.title('Elbow Method for Optimal K')
        plt.xticks(K_range)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()

    def plot_clusters(self, save_path='static/img/kmeans_clusters.png'):
        plt.figure(figsize=(8,6))
        for i in range(self.kmeans.n_clusters):
            plt.scatter(
                self.X[self.labels == i, 0],
                self.X[self.labels == i, 1],
                label=f'Cluster {i}'
            )
        plt.scatter(
            self.centroids[:, 0],
            self.centroids[:, 1],
            c='black',
            marker='X',
            label='Centroids'
        )
        plt.xlabel(self.selected_x.replace('_', ' ').title())
        plt.ylabel(self.selected_y.replace('_', ' ').title())
        plt.title(f'K-Means Clustering (K={self.kmeans.n_clusters})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()

    def fit_kmeans(self, optimal_k=7):
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.X_scaled)
        self.df['Cluster'] = self.labels
        print(self.df['Cluster'].value_counts())
        centroids_scaled = self.kmeans.cluster_centers_
        self.centroids = self.scaler.inverse_transform(centroids_scaled)

    def save_labeled_data(self, output_path='fifa_players.csv'):
        self.df.to_csv(output_path, index=False)

    def get_numeric_columns(self):
        if self.df is not None:
            return self.df.select_dtypes(include=[np.number]).columns.tolist()
        return []

if __name__ == "__main__":
    clustering = Clustering('fifa_players.csv')
    clustering.load_data()
    numeric_cols = clustering.get_numeric_columns()
    x_label, y_label = numeric_cols[0], numeric_cols[1]
    clustering.select_features(x_label, y_label)
    clustering.scale_features()
    clustering.plot_elbow(max_k=10)
    clustering.fit_kmeans(optimal_k=7)
    clustering.plot_clusters()
    clustering.save_labeled_data()