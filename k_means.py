import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\elcot\\Desktop\\machine learning tutorial\\clustering\\wine_customer.csv")

x=data.iloc[:,0:13].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,20):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.figure(dpi=300)   
plt.plot(range(1,20),wcss)
plt.title("elbow method")
plt.show()

kmeans=KMeans(n_clusters=3)
y_pred=kmeans.fit_predict(x)
c="D:\\real estate project\\cutomer.csv"

df=data
df['customers']=y_pred
df.to_csv(c,index=False)