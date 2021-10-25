import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\elcot\\Desktop\\machine learning tutorial\\clustering\\wine_customer.csv")

x=data.iloc[:,0:13].values

import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='centroid'))
plt.title("dendrogram")
plt.show()

from sklearn.cluster import AgglomerativeClustering
aggora=AgglomerativeClustering(n_clusters=3)
y_pred=aggora.fit_predict(x)


df=data
df['customer']=y_pred
df.to_csv("D:\\real estate project\\hc_customer.csv")