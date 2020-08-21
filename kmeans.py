import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('.\kmeans.csv')
df = pd.DataFrame(df)
print(df)

data = df['F']
cityName = df['地区']
km = KMeans(n_clusters=5)  # 构造聚类器
y = km.fit_predict(data.values.reshape(-1,1))  # 聚类
mpl.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.scatter(cityName,data)
plt.tick_params(labelsize=6)
plt.xticks(rotation=60)
plt.axhline(0.9897806,color='c',linewidth=2,linestyle= "dashed")
plt.axhline(-0.40190772,color='c',linewidth=2,linestyle= "dashed")
plt.axhline(0.18540474,color='c',linewidth=2,linestyle= "dashed")
plt.axhline(-0.70173354,color='c',linewidth=2,linestyle= "dashed")
plt.axhline(-0.15881039,color='c',linewidth=2,linestyle= "dashed")
plt.show()
label_pred = km.labels_  # 获取聚类标签
centroids = km.cluster_centers_  # 获取聚类中心
print(label_pred)
print(centroids)

plt.scatter(cityName,y)
plt.xticks(rotation=60)
plt.tick_params(labelsize=6)
plt.show()

CityCluster = [[],[],[],[],[]]
for i in range(len(cityName)):
        CityCluster[label_pred[i]].append(cityName[i])
for i in range(len(CityCluster)):
    print(CityCluster[i])




