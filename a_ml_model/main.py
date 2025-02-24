import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('placement.csv')

df.head()

plt.scatter(df['cgpa'],df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')

X = df.iloc[:,0:1]
y = df.iloc[:,-1]

X

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression
 
lr = LinearRegression()

lr.fit(X_train,y_train)

X_test

y_test

lr.predict(X_test.iloc[0].values.reshape(1,1))

plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = lr.predict(X_test)

y_test.values

print("MAE",mean_absolute_error(y_test,y_pred))


print("MSE",mean_squared_error(y_test,y_pred))

print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))

print("MSE",r2_score(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

# Adjusted R2 score
X_test.shape

1 - ((1-r2)*(40-1)/(40-1-1))

new_df1 = df.copy()
new_df1['random_feature'] = np.random.random(200)

new_df1 = new_df1[['cgpa','random_feature','package']]
new_df1.head()

plt.scatter(new_df1['random_feature'],new_df1['package'])
plt.xlabel('random_feature')
plt.ylabel('package(in lpa)')

X = new_df1.iloc[:,0:2]
y = new_df1.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print("R2 score",r2_score(y_test,y_pred))
r2 = r2_score(y_test,y_pred)


1 - ((1-r2)*(40-1)/(40-1-2))

new_df2 = df.copy()

new_df2['iq'] = new_df2['package'] + (np.random.randint(-12,12,200)/10)

new_df2 = new_df2[['cgpa','iq','package']]

new_df2.sample(5)

plt.scatter(new_df2['iq'],new_df2['package'])
plt.xlabel('iq')
plt.ylabel('Package(in lpa)')

np.random.randint(-100,100)

X = new_df2.iloc[:,0:2]
y = new_df2.iloc[:,-1]

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print("R2 score",r2_score(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

1 - ((1-r2)*(40-1)/(40-1-2))


m = lr.coef_

b = lr.intercept_


# y = mx + b

m * 8.58 + b
