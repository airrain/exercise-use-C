import numpy as np
import matplotlib.pyplot as plt
n_dots = 200

X = np.linspace(0, 1, n_dots)                   
y = np.sqrt(X) + 0.2*np.random.rand(n_dots) - 0.1
X = X.reshape(-1,1)
y = y.reshape(-1,1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import LinearRegression

def polynomial_model(degree = 1):
    polynomial_features = PolynomialFeatures(degree = degree,include_bias = False)
    linear_regression= LinearRegression()
    pipeline = Pipeline([("PolynomialFeatures",PolynomialFeatures),("linear_regression",linear_regression)
    ])
    return pipeline

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator,title,X,y,ylim = None,cv = None,n_jobs = 1,train_size = np.linspace(.1,1.0,5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes,train_scores,test_scores = learning_curve(estimator,X,y,cv = cv,n_jobs = n_jobs,train_size = train_size)
    train_scores_mean = np.mean(train_scores,axis = 1)
    train_scores_std = np.std(train_scores,axis = 1)
    test_scores_mean = np.mean(test_scores,axis = 1)
    test_scores_std = np.std(test_scores,axis = 1)
    plt.grid()
    plt.fill_between(train_sizes,train_scores_mean - train_scores_std,train_scores_mean + train_scores_std,alpha = 0.1,color = "r")
    plt.fill_between(train_sizes,test_scores_mean - test_scores_std,test_scores_mean + test_scores_std,alpha = 0.1,color = "f")
    plt.plot(train_sizes,train_scores_mean,'o--',color = "r",label = "Training score")
    plt.plot(train_sizes,test_scores_mean,'o-',color = "g",label = "Cross_validation score")
    plt.legend("best")
    return plt

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
titles = ['Learning Curves',
          'Learning Curves(UnderFitting)',
          'Learning Curves(OverFiting)'

]
degrees = [1,3,10]
plt.figure(figsize=(18,4))
for i in range(len(degrees)):
    plt.subplot(1,3,i + 1)
    plot_learning_curve(polynomial_model(degrees[i]),titles[i],X,y,ylim = (0.75,1.01),cv = cv)
plt.show()

#knn算法分类
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
centers = [[-2,2],[2,2],[0,4]]
X,y = make_blobs(n_samples=60, centers=centers, cluster_std=0.6, random_state=0)
plt.figure(figsize = (16,8))
c = np.array(centers)
plt.scatter(X[:,0],X[:,1],c = y,s = 100,cmap = 'cool')
plt.scatter(c[:,0],c[:,1],s = 100,marker = '^',cmap = 'orange')

from sklearn.neighbors import KNeighborsClassifier
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X,y)
X_sample = [0,2]
X_sample = np.array((X_sample)).reshape(1,-1)
y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(X_sample, return_distance=False)
plt.figure(figsize = (16,10))
plt.scatter(X[:,0],X[:,1],c = y,s = 100,cmap = 'cool')
plt.scatter(c[:,0],c[:,1],s = 100,marker = '^',cmap = 'orange')
plt.scatter(X_sample[0,0],X_sample[0,1],marker = 'x',s = 100,cmap = 'cool')
for i in neighbors[0]:
    plt.plot([X[i][0],X_sample[0][0]],[X[i][1],X_sample[0][1]],'k--',linewidth = 0.6)

#knn算法拟合
import matplotlib.pyplot as plt
import numpy as np
n_dots = 50
X = np.random.rand(n_dots,1)
y = 5 * np.cos(X).ravel()
y += 0.2 * np.random.rand(n_dots) - 0.1
from sklearn.neighbors import KNeighborsRegressor
k = 5
knn = KNeighborsRegressor(n_neighbors = k)
knn.fit(X,y)
T = np.linspace(0,1,500)[:,np.newaxis]
y_pred = knn.predict(T)
knn.score(X,y)
plt.figure(figsize = (16,10))
plt.scatter(X,y,c = 'g',s = 100,label = 'data')
plt.plot(T,y_pred,c = 'k',label = 'prediction',lw = 4)
plt.title("KNeighborsRegressor (k = %i)" % k)
plt.axis('tight')
plt.show()

#糖尿病预测
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/pima-indians-diabetes/diabetes.csv')
print('dataset shape {}'.format(data.shape))
data.head()
data.groupby("Outcome").size()
X = data.iloc[:,0:8]
Y = data.iloc[:,8]
print('shape of X {}； shapae of Y {}'.format(X.shape,Y.shape))

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
models = []
models.append("KNN",KNeighborsClassifier(n_neighbors = 2))
models.append("KNN with weights",KNeighborsClassifier(n_neighbors=5,weights='distance'))
models.append("KNN neighbors",KNeighborsClassifier(n_neighbors=2,radius = 500.0))
results = []
for name,model in models:
    model.fit(Xtrain,Ytrain)
    results.append(name,model.score(Xtest,Ytest))
for i in range(len(results)):
    print("name: {};score:{}".format(results[i][0],results[i][1]))

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

results = []
for name,model in models:
    kFold = KFold(n_splits = 10)
    cv_result = cross_val_score(model,X,Y,cv = kFold)
    results.append(name,cv_result)
for i in range(len(results)):
    print("name: {};cross val score:{}".format(results[i][0,results[i][1]]).mean())

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(Xtrain,Ytrain)
train_score = knn.score(Xtrain,Ytrain)
test_score = knn.score(Xtest,Ytest)
print("train score:{};test score:{}".format(train_score,test_score))

from sklearn.model_selection import ShuffleSplit
from common.utils import plot_learning_curve
   
knn = KNeighborsClassifier(n_neighbors = 2)
cv = ShuffleSplit(n_splits=10, test_size=0.2,  random_state=0)
plt.figure(16,10)
plot_learning_curve(plt,knn ,"Learn Curve KNN for Diabetes",X, y, ylim=(0.0,0.1), cv=cv)

from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k = 2)
X_new = selector.fit_transform(X,Y)
X_new[0:5]

for name,model in models:
    kfold = KFold(n_splits = 10)
    cv_result = cross_val_score(model,X_new,Y,cv = kfold)
    results.append(name,cv_result)
for i in range(len(results)):
    print("name: {};cross val score: {}".format(results[i][0],results[i][1]).mean()) 

plt.figure(figsize=(10,6))
plt.ylabel("BMI")
plt.xlabel("Glucose")
plt.scatter(X_new[Y == 0][:,0],X_new[Y == 0][:,1])
plt.scatter(X_new[Y == 1][:,0],X_new[Y == 1][:,1])

#线性回归算法
import matplotlib.pyplot as plt
import numpy as np

n_dots = 200
