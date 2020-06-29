import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '../Dataset Generation/')
import surfaceload as sl
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from keras.layers import Dense,Input,Dropout,BatchNormalization,concatenate,GRU,Embedding,Flatten
from keras.models import Model
from keras import backend as B
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from numpy import linalg as la
from Metrics_Success import Accuracy, dist_thresh, F_value, precise, Utility
import normalization
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes as nb
from sklearn import svm
from sklearn import linear_model as lm
from sklearn import datasets
from sklearn.metrics import roc_curve
import sys
sys.path.insert(0,'/Users/connorrobertson/Dropbox/Sailsbery Robertson Group/MachineLearning/')
from Metrics_Success import F_value
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis

mode = np.zeros((10,8))

for i in range(1):
    data = pd.read_csv('../Dataset Generation/pipes.csv')
    pipes = data.drop(['endx','endy','startx','elevation',
                       'starty','springville','orem','breaks'],axis = 1)
    breaks = data.breaks
    breaks = -.5*(breaks*(breaks-3))
    pipes.head()
    #Last minute feature engineering/transforming
    pipes['surface_load'] = normalization.rankgauss(pipes.surface_load)
    #Splitting Datasets
    train_sp,test,ans_sp,sols = train_test_split(pipes[data.orem ==0],
                                                        breaks[data.orem ==0],test_size = .3)
    train = np.array(pd.concat((pipes[data.orem ==1],train_sp)))
    ans = pd.concat((breaks[data.orem ==1],ans_sp)).values.astype(int)
    test = np.array(test)
    sols = np.array(sols).astype(int)

    model1 = nb.BernoulliNB()
    model1.fit(train,ans)
    preds1 = model1.predict_proba(test)[:,1]

    model2 = lm.LinearRegression()
    model2.fit(train,ans)
    preds2 = model2.predict(test)

    model3 = lm.ElasticNetCV()
    model3.fit(train,ans)
    preds3 = model3.predict(test)

    model4 = xgb.XGBRegressor(n_estimators=1000)
    model4.fit(train,ans)
    preds4 = model4.predict(test)

    tra = lgb.Dataset(train,label = ans)
    param = {'application' : 'regression','metric':'quantile'
             ,'nthread':4,'learning_rate':0.3,'max_depth':4,'num_leaves':20}
    model5 = lgb.train(param,train_set=tra,num_boost_round=1000)
    preds5 = model5.predict(test)
    
    model6 = svm.SVC(kernel='poly',C=10)
    model6.fit(train,ans)
    preds6 = model6.predict(test)
    
    model7 = RandomForestClassifier(n_estimators=1000)
    model7.fit(train,ans)
    preds7 = model7.predict_proba(test)[:,1]
    
    model8 = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', lm.ElasticNetCV())])
    model8.fit(train,ans)
    preds8 = model8.predict(test)

    x1,y1,t1 = roc_curve(sols,preds1)
    x2,y2,t2 = roc_curve(sols,preds2)
    x3,y3,t3 = roc_curve(sols,preds3)
    x4,y4,t4 = roc_curve(sols,preds4)
    x5,y5,t5 = roc_curve(sols,preds5)
    x6,y6,t6 = roc_curve(sols,preds6)
    x7,y7,t7 = roc_curve(sols,preds7)
    x8,y8,t8 = roc_curve(sols,preds8)
    P = np.sum(sols)
    N = len(sols)-P
    mode[i,:] = np.array([np.max(y1*P/(.5*(P-y1*P+x1*N) + y1*P)),np.max(y2*P/(.5*(P-y2*P+x2*N) + y2*P)),
                            np.max(y3*P/(.5*(P-y3*P+x3*N) + y3*P)),np.max(y4*P/(.5*(P-y4*P+x4*N) + y4*P)),
                            np.max(y5*P/(.5*(P-y5*P+x5*N) + y5*P)),np.max(y6*P/(.5*(P-y6*P+x6*N) + y6*P)),
                            np.max(y7*P/(.5*(P-y7*P+x7*N) + y7*P)),np.max(y8*P/(.5*(P-y8*P+x8*N) + y8*P))])

    

    plt.figure(figsize = (12,6))
    plt.plot(t1,y1*P/(.5*(P-y1*P+x1*N) + y1*P),label="Naive Bayes")
    plt.plot(t2,y2*P/(.5*(P-y2*P+x2*N) + y2*P),label="Linear Regression")
    plt.plot(t3,y3*P/(.5*(P-y3*P+x3*N) + y3*P),label="Elastic Net")
    plt.plot(t4,y4*P/(.5*(P-y4*P+x4*N) + y4*P),'-.',label="Gradient Boost")
    plt.plot(t5,y5*P/(.5*(P-y5*P+x5*N) + y5*P),label="LGB")
    plt.plot(t6,y6*P/(.5*(P-y6*P+x6*N) + y6*P),label="SVM")
    plt.plot(t7,y7*P/(.5*(P-y7*P+x7*N) + y7*P),label="Random Forest")
    plt.plot(t8,y8*P/(.5*(P-y8*P+x8*N) + y8*P),'-',label="Pipe Elastic")
    plt.xlim(0,.5)
    plt.ylabel('F1 Score')
    plt.xlabel('τ')
    plt.title('Change in F1 Score for Varied Probability Cutoffs')
    plt.legend()
    plt.savefig("ROCsample.pdf",bbox_inches="tight")
    #plt.show()
    
    