import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt


two_dim_easy = np.array(pd.read_csv("TwoDimEasy.csv", usecols=[1, 2]))
two_dim_hard = np.array(pd.read_csv("TwoDimHard.csv", usecols=[1, 2]))
shit_wine = np.array(pd.read_csv("wine.csv")).tolist()

wine = []
for each in shit_wine:
    k = each[0].split(';')
    point = np.array([float(each) for each in k])
    wine.append(point[:-1])
wine = np.array(wine)


true_easy = [two_dim_easy[:138], two_dim_easy[138:]]
true_hard = [two_dim_hard[:89], two_dim_hard[89:189], two_dim_hard[189:286], two_dim_hard[286:]]

wine_k = [1, 2, 3, 4, 5, 6, 7, 8]


def statistics(source, table):
    sse_set, ssb, total_center = [], 0, source.mean(axis=0)
    for each in table:
        center = each.mean(axis=0)
        ssb += pow(distance.euclidean(center, total_center), 2) * len(each)
        sse = 0
        for point in each:
            sse += pow(distance.euclidean(point, center), 2)
        sse_set.append(sse)
    print("cluster SSE is", sse_set)
    print("overall SSE is", sum(sse_set))
    print("SSB is", ssb)
    return sse_set


# First, pick up a random point to be the first center.
# Then, pick the second point which is the farthest one away from the first one.
# Next, pick the third point which has the biggest shorter distance with those two points.
# (选择与前两个中心点中最近距离最大的点作为第三个点)
# etc.
def get_origin_centers(k, source):
    centers = [np.random.randint(0, len(source))]
    for _ in range(k-1):
        max_distance, max_index = 0, -1
        for i, point in enumerate(source):
            min_distance = float('inf')
            for center in centers:
                min_distance = min(min_distance, distance.euclidean(point, source[center]))
            if min_distance > max_distance:
                    max_distance, max_index = min_distance, i
        centers.append(max_index)
    return [source.tolist()[each] for each in centers]


def update_centers(index_cluster, source):
    data_cluster = []
    for each in index_cluster:
        data_cluster.append([source[index] for index in each])
    return [np.array(each).mean(axis=0).tolist() for each in data_cluster]


def k_means(k, source):
    centers = get_origin_centers(k, source)
    index_cluster, last_centers, flag = [], centers, 1
    while flag or last_centers != centers:
        last_centers, index_cluster = centers, [[] for _ in range(k)]
        flag = 0
        for i, each in enumerate(source):
            distances = []
            for center in centers:
                distances.append(distance.euclidean(center, each))
            index_cluster[distances.index(min(distances))].append(i)
        centers = update_centers(index_cluster, source)
    return index_cluster


def index_to_table(index_cluster, source):
    table = []
    for each in index_cluster:
        table.append(np.array([source[index] for index in each]))
    return table

def getxy(dataset):
    x, y = [], []
    for each in dataset:
        x.append(each[0])
        y.append(each[1])
    return x, y

if __name__ == "__main__":
    plt.figure()
    fig, axs = plt.subplots(2, 2)

    print("*" * 100)
    print("-" * 10, "True Cluster", "-" * 10)
    print()

    print("#" * 10, "TwoDimEasy", "#" * 10)
    statistics(two_dim_easy, true_easy)
    print()
    print("#" * 10, "TwoDimHard", "#" * 10)
    statistics(two_dim_hard, true_hard)
    print()

    print("-" * 10, "k-Means Cluster", "-" * 10)
    print()

    print("#" * 10, "TwoDimEasy  k = 2", "#" * 10)
    easy_cluster_2 = k_means(2, two_dim_easy)
    for i, each in enumerate(easy_cluster_2):
        print("Cluster", i + 1, "[", len(each), "]")
        for point in each:
            print(point + 1, end=",")
        print()
        print()
    easy_table_2 = index_to_table(easy_cluster_2, two_dim_easy)
    statistics(two_dim_easy, easy_table_2)

    for cent, c, marker in zip(range(2), ['g', 'r'], ['.', '.']):
        x1, x2 = getxy(easy_table_2[cent])
        axs[0, 0].scatter(x1, x2, c=c, marker=marker, alpha=0.3)
    for cent, c, marker in zip(range(2), ['g', 'r'], ['o', 'o']):
        x1, x2 = getxy(true_easy[cent])
        axs[0, 0].scatter(x1, x2, c=c, marker=marker, alpha=0.3)
    axs[0, 0].set_title('TwoDimEasy & True k=2')

    print()
    print("#" * 10, "TwoDimHard  k = 4", "#" * 10)
    hard_cluster_4 = k_means(4, two_dim_hard)
    for i, each in enumerate(hard_cluster_4):
        print("Cluster", i + 1, "[", len(each), "]")
        for point in each:
            print(point + 1, end=",")
        print()
        print()
    hard_table_4 = index_to_table(hard_cluster_4, two_dim_hard)
    statistics(two_dim_hard, hard_table_4)

    for cent, c, marker in zip(range(4), ['g', 'r', 'b', 'y'], ['.', '.', '.', '.']):
        x1, x2 = getxy(hard_table_4[cent])
        axs[0, 1].scatter(x1, x2, c=c, marker=marker, alpha=0.3)
    for cent, c, marker in zip(range(4), ['g', 'r', 'b', 'y'], ['o', 'o', 'o', 'o']):
        x1, x2 = getxy(true_hard[cent])
        axs[0, 1].scatter(x1, x2, c=c, marker=marker, alpha=0.3)
    axs[0, 1].set_title('TwoDimHard & True k=4')

    print()
    print("#" * 10, "TwoDimEasy  k = 3", "#" * 10)
    easy_cluster_3 = k_means(3, two_dim_easy)
    for i, each in enumerate(easy_cluster_3):
        print("Cluster", i + 1, "[", len(each), "]")
        for point in each:
            print(point + 1, end=",")
        print()
        print()
    easy_table_3 = index_to_table(easy_cluster_3, two_dim_easy)
    statistics(two_dim_easy, easy_table_3)

    for cent, c, marker in zip(range(3), ['g', 'r', 'b'], ['.', '.', '.']):
        x1, x2 = getxy(easy_table_3[cent])
        axs[1, 0].scatter(x1, x2, c=c, marker=marker, alpha=0.3)
    axs[1, 0].set_title('TwoDimEasy k=3')

    print()
    print("#" * 10, "TwoDimHard  k = 3", "#" * 10)
    hard_cluster_3 = k_means(3, two_dim_hard)
    for i, each in enumerate(hard_cluster_3):
        print("Cluster", i + 1, "[", len(each), "]")
        for point in each:
            print(point + 1, end=",")
        print()
        print()
    hard_table_3 = index_to_table(hard_cluster_3, two_dim_hard)
    statistics(two_dim_hard, hard_table_3)

    for cent, c, marker in zip(range(3), ['g', 'r', 'b'], ['.', '.', '.']):
        x1, x2 = getxy(hard_table_3[cent])
        axs[1, 1].scatter(x1, x2, c=c, marker=marker, alpha=0.3)
    axs[1, 1].set_title('TwoDimEasy k=3')

    print()
    plt.figure()
    print("-" * 10, "Wine", "-" * 10)
    distortions = []
    for k in wine_k:
        print("#" * 10, "k =", k, "#" * 10)
        wine_cluster = k_means(k, wine)
        for i, each in enumerate(wine_cluster):
            print("Cluster", i + 1)
            for point in each:
                print(point + 1, end=',')
            print()
            print()
        table = index_to_table(wine_cluster, wine)
        distortions.append(sum(statistics(wine, table)))
    plt.plot(range(1, 9), distortions, marker="o")
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.show()