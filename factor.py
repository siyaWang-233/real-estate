import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from factor_analyzer import FactorAnalyzer
from sklearn.cluster import KMeans
from sklearn import  datasets
from sklearn import metrics
from sklearn.datasets import make_blobs
from matplotlib import pyplot
import csv
from pylab import *
import seaborn as sns

datafile = u'./factor22.xlsx'
data = pd.read_excel(datafile)
print(data)
data.plot()
mpl.rcParams['font.sans-serif']=['SimHei']

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(data)
print(u'KMO值:'+str(kmo_model))
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(data)
print('Bartlett近似卡方:'+str(chi_square_value))
print('BartlettP值:'+str(p_value))


fa=FactorAnalyzer(11,rotation=None) ##11个指标
fa.fit(data)
ev,v=fa.get_eigenvalues()
print(ev)
import matplotlib.pyplot as plt
import matplotlib
plt.scatter(range(1,data.shape[1]+1),ev)
plt.plot(range(1,data.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
fa = FactorAnalyzer(3,rotation='varimax')
fa.fit(data)
l = pd.DataFrame(fa.loadings_)
print("\n旋转后的成分矩阵：\n"+str(l))

v=fa.get_factor_variance()
for i in v:
    print(i)
a = fa.transform(data)
print("\n输出因子的系数：\n"+str(a))

df_cm = pd.DataFrame(np.abs(l))
plt.figure(figsize = (10,15))
ax = sns.heatmap(df_cm, annot=True, cmap="BuPu")
ax.yaxis.set_tick_params(labelsize=10)
plt.title('Factor Analysis', fontsize='small')
plt.ylabel('Sepal Width', fontsize='small')
plt.savefig('factorAnalysis.png', dpi=500)
plt.show()

df=pd.DataFrame(a)
df.columns = ['F1','F2','F3']
print(df[['F1','F2','F3']])
datafile = u'./factorname.xlsx'
df1 = pd.read_excel(datafile)
print(df1)
area = df1.pop('地区')
df.insert(0,'地区',area)
print(df)

F1 = df.sort_values(by='F1',ascending=False)
print(F1)
F2 = df.sort_values(by='F2',ascending=False)
print(F2)
F3 = df.sort_values(by='F3',ascending=False)
print(F3)


df.eval('F = 0.46*F1+0.21*F2+0.12*F3',inplace=True)
print(df)
F = df.sort_values(by='F',ascending=False)
print(F)













'''
x = df['F']
km = KMeans(n_clusters=4)
km.fit(x.values.reshape(-1,1))
print(km.cluster_centers_)


df['F'],target=make_blobs(n_samples=31,n_features=1,centers=4)
pyplot.scatter(df['F'][:,0],c=target)
pyplot.show()
'''


df2=df[['地区','F']]
print(df2)

df2.to_csv ("kmeans.csv" , encoding = "utf-8",index=False)
