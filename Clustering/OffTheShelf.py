import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

two_dim_easy = np.array(pd.read_csv("TwoDimEasy.csv", usecols=[1, 2]))
two_dim_hard = np.array(pd.read_csv("TwoDimHard.csv", usecols=[1, 2]))
shit_wine = np.array(pd.read_csv("wine.csv")).tolist()

wine = []
for each in shit_wine:
    k = each[0].split(';')
    point = np.array([float(each) for each in k])
    wine.append(point[:-1])
wine = np.array(wine)
wine_k = [1, 2, 3, 4, 5, 6, 7, 8]

if __name__ == "__main__":
    print("*" * 100)
    print("-" * 10, "k-Means Cluster", "-" * 10)
    print()
    plt.figure()
    fig, axs = plt.subplots(2, 2)
    print("#" * 10, "TwoDimEasy  k = 2", "#" * 10)
    X = two_dim_easy
    estimator = KMeans(n_clusters=2)
    estimator.fit(X)
    print("SSE is", estimator.inertia_)
    label_pred = estimator.labels_
    x0 = X[label_pred == 0]
    x1 = X[label_pred == 1]
    axs[0, 0].scatter(x0[:, 0], x0[:, 1], c="red", marker='.')
    axs[0, 0].scatter(x1[:, 0], x1[:, 1], c="green", marker='*')
    axs[0, 0].set_title('TwoDimEasy k=2')

    print()
    print("#" * 10, "TwoDimHard  k = 4", "#" * 10)
    X = two_dim_hard
    estimator = KMeans(n_clusters=4)
    estimator.fit(X)
    print("SSE is", estimator.inertia_)
    label_pred = estimator.labels_
    x0 = X[label_pred == 0]
    x1 = X[label_pred == 1]
    x2 = X[label_pred == 2]
    x3 = X[label_pred == 3]
    axs[0, 1].scatter(x0[:, 0], x0[:, 1], c="red", marker='.')
    axs[0, 1].scatter(x1[:, 0], x1[:, 1], c="green", marker='*')
    axs[0, 1].scatter(x2[:, 0], x2[:, 1], c="blue", marker='o')
    axs[0, 1].scatter(x3[:, 0], x3[:, 1], c="yellow", marker='^')
    axs[0, 1].set_title('TwoDimHard k=4')

    print()
    print("#" * 10, "TwoDimEasy  k = 3", "#" * 10)
    X = two_dim_easy
    estimator = KMeans(n_clusters=3)
    estimator.fit(X)
    print("SSE is", estimator.inertia_)
    label_pred = estimator.labels_
    x0 = X[label_pred == 0]
    x1 = X[label_pred == 1]
    x2 = X[label_pred == 2]
    axs[1, 0].scatter(x0[:, 0], x0[:, 1], c="red", marker='.')
    axs[1, 0].scatter(x1[:, 0], x1[:, 1], c="green", marker='*')
    axs[1, 0].scatter(x2[:, 0], x2[:, 1], c="blue", marker='^')
    axs[1, 0].set_title('TwoDimEasy k=3')

    print()
    print("#" * 10, "TwoDimHard  k = 3", "#" * 10)
    X = two_dim_hard
    estimator = KMeans(n_clusters=3)
    estimator.fit(X)
    print("SSE is", estimator.inertia_)
    label_pred = estimator.labels_
    x0 = X[label_pred == 0]
    x1 = X[label_pred == 1]
    x2 = X[label_pred == 2]
    axs[1, 1].scatter(x0[:, 0], x0[:, 1], c="red", marker='.')
    axs[1, 1].scatter(x1[:, 0], x1[:, 1], c="green", marker='*')
    axs[1, 1].scatter(x2[:, 0], x2[:, 1], c="blue", marker='^')
    axs[1, 1].set_title('TwoDimHard k=3')
    print()
    plt.figure()
    distortions = []
    for k in wine_k:
        km = KMeans(n_clusters=k)
        km.fit(wine)
        distortions.append(km.inertia_)
    plt.plot(range(1, 9), distortions, marker="o")
    plt.xlabel("k")
    plt.ylabel("SSE")

    print()
    plt.figure()
    fig, axs = plt.subplots(2, 2)

    X = two_dim_easy
    y_pred = DBSCAN(eps=0.2, min_samples=10).fit_predict(X)
    axs[0, 0].scatter(X[:, 0], X[:, 1], c=y_pred)
    axs[0, 0].set_title('TwoDimEasy eps=0.2')

    X = two_dim_easy
    y_pred = DBSCAN(eps=0.1, min_samples=10).fit_predict(X)
    axs[0, 1].scatter(X[:, 0], X[:, 1], c=y_pred)
    axs[0, 1].set_title('TwoDimEasy eps=0.1')

    X = two_dim_hard
    y_pred = DBSCAN(eps=0.03, min_samples=10).fit_predict(X)
    axs[1, 0].scatter(X[:, 0], X[:, 1], c=y_pred)
    axs[1, 0].set_title('TwoDimHard eps=0.03')

    X = two_dim_hard
    y_pred = DBSCAN(eps=0.06, min_samples=10).fit_predict(X)
    axs[1, 1].scatter(X[:, 0], X[:, 1], c=y_pred)
    axs[1, 1].set_title('TwoDimHard eps=0.06')

    print()
    plt.figure()
    fig, axs = plt.subplots(2, 2)

    X = two_dim_easy
    y_pred = GaussianMixture(n_components=2).fit_predict(X)
    axs[0, 0].scatter(X[:, 0], X[:, 1], c=y_pred)
    axs[0, 0].set_title('TwoDimEasy n=2')

    X = two_dim_hard
    y_pred = GaussianMixture(n_components=4).fit_predict(X)
    axs[0, 1].scatter(X[:, 0], X[:, 1], c=y_pred)
    axs[0, 1].set_title('TwoDimHard n=4')

    X = two_dim_easy
    y_pred = GaussianMixture(n_components=3).fit_predict(X)
    axs[1, 0].scatter(X[:, 0], X[:, 1], c=y_pred)
    axs[1, 0].set_title('TwoDimEasy n=3')

    X = two_dim_hard
    y_pred = GaussianMixture(n_components=3).fit_predict(X)
    axs[1, 1].scatter(X[:, 0], X[:, 1], c=y_pred)
    axs[1, 1].set_title('TwoDimHard n=3')

    plt.show()