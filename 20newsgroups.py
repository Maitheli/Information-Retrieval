
import numpy as np
import math


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


from sklearn.datasets import fetch_20newsgroups


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)



cat = ['alt.atheism',   'comp.graphics',   'comp.os.ms-windows.misc',   'comp.sys.ibm.pc.hardware',   'comp.sys.mac.hardware',   'comp.windows.x',   'misc.forsale',   'rec.autos',
   'rec.motorcycles',   'rec.sport.baseball',   'rec.sport.hockey',   'sci.crypt',   'sci.electronics',   'sci.med',   'sci.space',]


no_of_classes = 15

dataset = fetch_20newsgroups(subset='all', categories = cat)   

docs_in_each_class = np.zeros(no_of_classes)
vectorizer = CountVectorizer(min_df=1)
raw = vectorizer.fit_transform(dataset.filenames)
transformer = TfidfTransformer()
X = transformer.fit_transform(raw)
X=X.toarray()
y=dataset.target

#print(X)
[a,b] = X.shape

no_of_clusters = input("Enter the number of clusters: ")
max_itr = input("Enter the number of iterations: ")


centroids = np.zeros((no_of_clusters, b))
#old_centroids= centroids
#taking first few points as the starting centroids
for i in range (0,no_of_clusters):
	centroids[i] = X[i]

#print(centroids)


clus_list = [ []for i in range(no_of_clusters)]

label = np.zeros(a)
label=label.view('int64')
itr=0

while( itr<max_itr ):
	#Assigning labels
	for i in range (0, a):
		distance = np.zeros(no_of_clusters)
		for j in range(0, no_of_clusters):
			for k in range(0, b):
				distance[j] = distance[j] + pow((X[i][k] - centroids[j][k]),2)
			#distance[j] = pow(distance[j], 0.5)
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

	# cdist = np.zeros(nCr(no_of_clusters,2))
	
	# for i in range(0,no_of_clusters):
	# 	for j in range(0, b):
	# 		cdist[i] =  pow((centroids[i][j]-old_centroids[i][j]),2)
	# 	cdist[i] = pow(cdist[i], 0.5)

	#old_centroids = centroids
	
	itr+=1

total_data_points = a
#Cluster & Class matrix
Z = np.zeros((no_of_clusters, no_of_classes))

for i in range (0, a):
    Z[label[i], y[i]] = Z[label[i], y[i]] + 1

print("PRINTING THE MATRIX\n") 
print(Z)
print("\nPRINTING THE STATS\n") 

for i in range (0, a):
    docs_in_each_class[y[i]] = docs_in_each_class[y[i]] + 1


precision_of_clustering = 0
recall_of_clustering = 0
elements_in_cluster = 0
maxi = 0
max_recall=0
for i in range (0,no_of_clusters):
    for j in range (0, no_of_classes):
        elements_in_cluster = elements_in_cluster + Z[i][j]			#finding total element in cluster i
        recall=Z[i][j]/docs_in_each_class[j] 					 	#calculating recall for each i wrt j
        if max_recall < recall:
        	max_recall=recall 										#finding the max recall
        if maxi < Z[i][j]:
            maxi = Z[i][j]											#finding the max no of docs in the m(i,j)
    
    precision = maxi/elements_in_cluster		
    if precision_of_clustering < precision:
        precision_of_clustering = precision
  
  	
    if recall_of_clustering < max_recall:
        recall_of_clustering = max_recall

    print("Precision of cluster ", i+1, ": ", precision)
    print("Recall of cluster", i+1, ": ", max_recall)
    print("F-Measure of cluster", i+1, ": ", (2*precision*max_recall)/(precision + max_recall))
    print("\n")    
    elements_in_cluster = 0
    maxi = 0
    max_recall=0

print("\n")
print("Precision of clustering : ", precision_of_clustering)
print("Recall of cluster : ", recall_of_clustering)
print("F-Measure of clustering : ", (2*precision_of_clustering*recall_of_clustering)/(precision_of_clustering + recall_of_clustering))
