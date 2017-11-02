import pandas as pd
import numpy as np
import quandl
import math, datetime
from sklearn import preprocessing,cross_validation,svm,linear_model
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
quandl.ApiConfig.api_key = 'xbPnXMAxKyEVqwzW9TWv'
df  = quandl.get_table('WIKI/PRICES')
print(df.head())
df['HL_PCT'] =(df['adj_high']-df['adj_close'])/df['adj_close'] * 100.0
df['PCT_change'] =(df['adj_close']-df['adj_open']) / df['adj_open'] * 100.0
df = df[['adj_close','HL_PCT','PCT_change','adj_volume']]
forecast_col = df['adj_close']
forecast_out = int(math.ceil(0.001*len(forecast_col)))
df['label'] = forecast_col.shift(-forecast_out)
df.dropna(inplace=True)

X=np.array(df.drop(['label'],1))#,'adj_close'],1))

X=preprocessing.scale(X)
#print("X after preprocessing.scale ",X)
X_lately = X[-forecast_out:]
#print("X_lately",X_lately)
X=X[:-forecast_out]

#print(df)
#print("X",X)
Y=np.array(df['label'])
#Y=preprocessing.scale(Y)
Y=Y[:-forecast_out]
#print("Y ",Y)

x_train,x_test,y_train,y_test=cross_validation.train_test_split(X,Y, test_size=0.2)
clf=linear_model.LinearRegression(n_jobs=-1)
clf=clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
forecast_set=clf.predict(X_lately)
#forecast_set_whole=clf.predict(X)
print(accuracy,forecast_set)#,forecast_set_whole)
forecast_set=np.array(forecast_set)
df['Forecast'] = np.nan

last_date=df.iloc[-1].name
last_unix=last_date#.timestamp()
one_day=1
next_unix=last_unix+one_day

for i in forecast_set:
 next_date = next_unix#datetime.datetime.fromtimestamp(next_unix)
 next_unix+=one_day
 df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]

print(df.label[:-forecast_out])
print(forecast_set)
df['adj_close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()