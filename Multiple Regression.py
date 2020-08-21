import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import math
import seaborn as sns
from pandas import DataFrame,Series
from matplotlib import font_manager as fm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from pylab import *
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
mpl.rcParams['font.sans-serif']=['SimHei']

datafile = u'./guangdong.xlsx'
data = pd.read_excel(datafile)
print(data)
dataDf = DataFrame(data)
print(dataDf)
print(dataDf.corr())#相关系数
sns.heatmap(dataDf.corr())
plt.xticks(rotation=0)
sns.pairplot(dataDf, x_vars=['房地产开发企业个数','房地产从业人员数','年末常住人口','商品房销售面积'], y_vars='商品房平均销售价格', aspect=0.8, kind='reg')
plt.show()

X=dataDf.loc[:,('房地产开发企业个数','房地产从业人员数','年末常住人口','商品房销售面积')]
y=dataDf.loc[:,'商品房平均销售价格']
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=100)
print ('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n y_test.shape={}'.format(X_train.shape,y_train.shape, X_test.shape,y_test.shape))
model = LinearRegression()
model.fit(X_train,y_train)
a = model.intercept_
b = model.coef_
print(a,b)
print("最佳拟合线: Y = ",round(a,2),"+",round(b[0],2),"* X1 + ",round(b[1],2),"* X2+",round(b[2],2),"* X3+",round(b[3],2),"* X4")
y_pred = model.predict(X_test)
print(y_pred)

plt.plot(range(len(y_pred)),y_pred,'red', linewidth=2.5,label="predict data")
plt.plot(range(len(y_test)),y_test,'green',label="test data")
plt.legend(loc=2)
plt.show()

score=model.score(X_test,y_test)
print("\nscore：\n"+str(score))
print("Mean squared error:%.2f"
      %mean_squared_error(y_test, y_pred))
print('Variance score:%.2f' % r2_score(y_test, y_pred))



