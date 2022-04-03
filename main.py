import random
import time
import numpy as np
import pandas as pd
from scipy.spatial import distance

data = pd.read_csv('data.csv', header=None)
labels = pd.read_csv('label.csv', header=None)
labels = labels.to_numpy()
data = data.to_numpy()

num_centers = 10
iteration = 100
#initializatiom: Randomly select centers
centers = data[np.random.choice(data.shape[0], num_centers, replace=False), :]
p_cluster = []
p_sse = 0.0
for i in range(len(data)):
    dist = []
    for j in range(num_centers):
        euc = distance.euclidean(data[i], centers[j])
        #cosine = distance.cosine(data[i], centers[j])
        #jacc = 1 - np.divide(np.sum(np.minimum(data[i], centers[j])), np.sum(np.maximum(data[i], centers[j])))
        #jacc = distance.jaccard(data[i], centers[j])
        dist.append(euc)
    p_cluster.append(np.argmin(dist))

for d, c in zip(data, p_cluster):
    dis = distance.euclidean(d, centers[c])
    p_sse += np.square(dis).sum()

new_df = pd.concat([pd.DataFrame(data), pd.DataFrame(p_cluster, columns=['cluster'])], axis=1)

start = time.time()
for n in range(iteration):
    sse = 0.0
    new_cluster = []
    new_center =[]

    for c in set(new_df['cluster']):
    #divide the data into their clusters
        current_cluster = new_df[new_df['cluster'] == c]
        cluster_mean = current_cluster.mean(axis=0)
        cluster_mean.to_numpy()
        new_center.append(cluster_mean)

    centers = np.array(new_center, dtype='f')
    centers = centers[ : , :-1]

    for i in range(len(data)):
        distt = []
        for j in range(num_centers):
            euc = distance.euclidean(data[i], centers[j])
            #cosine = distance.cosine(data[i], centers[j])
            #jacc = distance.jaccard(data[i], centers[j])
            #jacc = 1 - np.divide(np.sum(np.minimum(data[i], centers[j])), np.sum(np.maximum(data[i], centers[j])))
            distt.append(euc)
        new_cluster.append(np.argmin(distt))

    new_df = pd.concat([pd.DataFrame(data), pd.DataFrame(new_cluster, columns=['cluster'])], axis=1)

    for d, c in zip(data, new_cluster):
        dis1 = distance.euclidean(d, centers[c])
        sse += np.square(dis1).sum()

    #if np.array_equal(p_cluster, new_cluster) or sse > p_sse or n == 500:
    if n==100:
        break
    p_cluster = new_cluster
    p_sse = sse

end = time.time()
print("sse")
print(sse)
print("time")
print(end-start)
print("iterations")
print(n)

cluster_score = np.zeros((len(labels), len(labels)), dtype=int)
# Calculate the scores using confusion matrix
for p in range(len(labels)):
    cluster_score[new_cluster[p], labels[p, 0]] += 1
correctly_clustered = 0
total_scores = 0
for x in range(len(labels)):
    cluster_w= 0
    cluster_max = 0
    for y in range(len(labels)):
        if cluster_score[x][y] > cluster_max:
            cluster_w = y
            cluster_max = cluster_score[x][y]
    # Calculate total scores and the correct scores to find accuracy
    for y in range(len(labels)):
        total_scores += cluster_score[x][y]
        if y == cluster_w:
            correctly_clustered += cluster_score[x][y]

print("accuracy = ")
print(correctly_clustered / total_scores)