from sklearn import datasets
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

print("\n")
print("The class each data point belongs to :")
print(y)


[a,b] = X.shape

no_of_clusters = input("Enter the number of clusters: ")
max_itr = input("Enter the number of iterations: ")

centroids = np.zeros((no_of_clusters, b))
old_centroids= centroids
#taking first few points as the starting centroids
for i in range (0,no_of_clusters):
	centroids[i] = X[i]

#print(centroids)


clus_list = [ []for i in range(no_of_clusters)]

label = np.zeros(a)
label=label.view('int64')
itr=0

while( itr<max_itr  ):
	#Assigning labels
	for i in range (0, a):
		distance = np.zeros(no_of_clusters)
		for j in range(0, no_of_clusters):
			for k in range(0, b):
				distance[j] = distance[j] + pow((X[i][k] - centroids[j][k]),2)
			distance[j] = pow(distance[j], 0.5)
		distance
		label[i]=np.argmin(distance)
		if(i in clus_list[label[i]]):
			clus_list[label[i]].remove(i)
		clus_list[np.argmin(distance)].append(i)
	#print(clus_list)


	#Calculation of centroids after assigning the point to clusters.
	for i in range(0,no_of_clusters):
		temp=np.zeros(b)
		for j in range(0,len(clus_list[i])):
			temp += X[clus_list[i][j]]		
		centroids[i]=temp/(len(clus_list[i]))
	
	itr= itr+1
 
print("The centroids after clustering : ")
print(centroids)
print("\n")

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

print("The cluster each data point belongs to :")
print(label)
print("\n")

ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=label.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')

no_of_classes = 3
total_data_points = 150
#Cluster & Class matrix
b = np.zeros((no_of_clusters, no_of_classes))

for i in range (0, 150):
    b[label[i], y[i]] = b[label[i], y[i]] + 1
print(b)
print("\n")

precision_of_clustering = 0
recall_of_clustering = 0
elements_in_cluster = 0
max = 0
for i in range (0,no_of_clusters):
    for j in range (0, no_of_classes):
        elements_in_cluster = elements_in_cluster + b[i][j] 
        if max < b[i][j]: 
            max = b[i][j]
    precision = max/elements_in_cluster
    if precision_of_clustering < precision:
        precision_of_clustering = precision
    recall = max/50
    if recall_of_clustering < recall:
        recall_of_clustering = recall
    print("Precision of cluster ", i+1, ": ", precision)
    print("Recall of cluster", i+1, ": ", recall)
    print("F-Measure of cluster", i+1, ": ", (2*precision*recall)/(precision + recall))
    print("\n")    
    elements_in_cluster = 0
    max = 0

print("\n")
print("Precision of clustering : ", precision_of_clustering)
print("Recall of clustering : ", recall_of_clustering)
print("F-Measure of clustering : ", (2*precision_of_clustering*recall_of_clustering)/(precision_of_clustering + recall_of_clustering))
