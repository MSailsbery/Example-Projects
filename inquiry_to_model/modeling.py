import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import xgboost as xgb
import sklearn.svm as svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

def interp(ts,val):
    if val < max(ts):
        return np.argmax(ts>val)
    else:
        return -1

class modeling:
    
    def __init__(self,package,data,pipes,breaks,springville = True,
                 orem = False,classifier = False,ftimp = True, coef = False,lgb=False,**kwargs):
        self.package = package
        self.data = data
        self.pipes = pipes
        self.breaks = breaks
        self.springville = springville
        self.orem = orem
        self.classifier = classifier
        self.ftimp = ftimp
        self.coef = coef
        self.lgb = lgb
        self.kwargs = kwargs
        self.model = self.package(**self.kwargs)
        
    def _split(self,size = .3):
        if self.springville:
            train_sp,test,ans_sp,sols = train_test_split(self.pipes[self.data.orem ==0],
                                                         self.breaks[self.data.orem ==0],test_size = size)
            train = np.array(pd.concat((self.pipes[self.data.orem ==1],train_sp)))
            ans = pd.concat((self.breaks[self.data.orem ==1],ans_sp)).values.astype(int)
            test = np.array(test)
            sols = np.array(sols).astype(int)
            
        elif self.orem:
            train_or,test,ans_or,sols = train_test_split(self.pipes[self.data.orem ==1],
                                                    self.breaks[self.data.orem ==1],test_size = size)
            train = np.array(pd.concat((self.pipes[self.data.orem ==0],train_or)))
            ans = pd.concat((self.breaks[self.data.orem ==0],ans_or)).values.astype(int)
            test = np.array(test)
            sols = np.array(sols).astype(int)
            
        else:
            train,test,ans,sols = train_test_split(self.pipes,self.breaks,test_size = size)
            
        return train,test,ans,sols
    
    def _pred(self,size = .3,cutoff = None):
        train,test,ans,sols = self._split(size)
        
        self.model = self.package(**self.kwargs)
        self.model.fit(train,ans)
        if self.classifier:
            output = 1-self.model.predict_proba(test)[:,0]
        else:
            output = self.model.predict(test)

        if cutoff is not None:
            output = np.ones(output.shape)*(output >= cutoff)

        return output, sols

    
    def ens_pred(self,train,test,ans,cutoff = None):
        self.model = self.package(**self.kwargs)
        self.model.fit(train,ans)
        if self.classifier:
            output = 1-self.model.predict_proba(test)[:,0]
        else:
            output = self.model.predict(test)

        if cutoff is not None:
            output = np.ones(output.shape)*(output >= cutoff)
        
        return output

    def change_pred(self,train,test,ans,cutoff = None):
        self.model.fit(train,ans)
        if self.classifier:
            output = 1-self.model.predict_proba(test)[:,0]
        else:
            output = self.model.predict(test)

        if cutoff is not None:
            output = np.ones(output.shape)*(output >= cutoff)
        
        return output
        
    
    def accuracy(self,size = .3, n = 1,single=True):
        
        scores = []

        for i in range(n):
            p,s = self._pred(size)
            scores.append(1-Accuracy(p,s))
        if single:
            return np.mean(scores)
        else:
            return scores
        
    def f_score(self,size = .3, n = 1,single=True,cutoff = None):
        
        scores = []

        for i in range(n):
            p,s = self._pred(size,cutoff = cutoff)
            scores.append(F_value(p,s))
        if single:
            return np.mean(scores)
        else:
            return scores
        
    def rock(self):
        pred,sols = self._pred()
        
        P = np.sum(sols)
        N = len(sols)-P
        
        x,y,t = roc_curve(sols,pred)
        t = t[::-1]
        y = y[::-1]
        x = x[::-1]
        
        return t,y*P/(.5*(P-y*P+x*N) + y*P)
    
    def rock_accum(self,num_tau = 1000,n=10):
        heights = np.zeros(num_tau)
        taus = np.linspace(.001,.999,num_tau)
        for i in range(n):
            t,f = self.rock()
            inds = [interp(t,tau) for tau in taus]
            heights += np.array([f[ind] for ind in inds])
        self.model = self.package(**self.kwargs)
        return heights/n

    def _cutoff(self,num_tau = 1000, n =10, max_val=False):
        heights = self.rock_accum(num_tau=num_tau,n=n)
        taus = np.linspace(.001,.999,num_tau)
        if max_val:
            return taus[np.argmax(heights)], np.max(heights)
        else:
            return taus[np.argmax(heights)]

    def reinit(self):
        self.model = self.package(**self.kwargs)
                      
    def feature_importances(self,size=.3,n = 10):
        importances = np.zeros(len(list(self.pipes.columns)))
        
        if self.ftimp:
            for i in range(n):
                self._pred(size)
                importances += self.model.feature_importances_
                
        elif self.coef:
            for i in range(n):
                self._pred(size)
                importances += self.model.coef_
                
        elif self.lgb:
            for i in range(n):
                self._pred(size)
                importances += self.model.feature_importance('gain')
                
        return dict(zip(list(self.pipes.columns),importances/n))

    
    
    
    
class RFC_wrapper(BaseEstimator,ClassifierMixin):

    def __init__(self,cutoff=.5,n_estimators=100,criterion=0,max_features = None,max_depth = None,min_impurity_decrease = 0):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.cutoff = cutoff
        self.min_samples_leaf = 1
        
    def fit(self,train,labels):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,criterion=self.criterion,max_features=self.max_features,max_depth=self.max_depth,min_impurity_decrease=self.min_impurity_decrease,min_samples_leaf=self.min_samples_leaf)
        self.model.fit(train,labels)
        return self

    def predict(self,test):
        pred = 1-self.model.predict_proba(test)[:,0]
        pred = np.ones(pred.shape)*(pred>=self.cutoff)
        return pred

class RFR_wrapper(BaseEstimator,RegressorMixin):

    def __init__(self,cutoff=.5,n_estimators=0,criterion=0,max_features = 0,max_depth = 0,min_impurity_decrease = 0):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = 1
        self.cutoff = cutoff

    def fit(self,train,labels):
        self.model = RandomForestRegressor(n_estimators=self.n_estimators,criterion=self.criterion,max_features=self.max_features,max_depth=self.max_depth,min_impurity_decrease=self.min_impurity_decrease,min_samples_leaf=self.min_samples_leaf)
        self.model.fit(train,labels)
        return self

    def predict(self,test):
        pred = self.model.predict(test)
        pred = np.ones(pred.shape)*(pred>=self.cutoff)
        return pred

class SVC_wrapper(BaseEstimator,ClassifierMixin):

    def __init__(self,cutoff=.5,C=0,kernel='linear'):
        self.C = 0
        self.cutoff = cutoff
        self.kernel = kernel
        
    def fit(self,train,labels):
        self.model = svm.SVC(C=self.C,kernel=self.kernel)
        self.model.fit(train,labels)
        return self

    def predict(self,test):
        pred = 1-self.model.predict_proba(test)[:,0]
        pred = np.ones(pred.shape)*(pred>=self.cutoff)
        return pred
    
class SVR_wrapper(BaseEstimator,RegressorMixin):

    def __init__(self,cutoff=.5,C=0,kernel='linear'):
        self.C = 0
        self.cutoff = cutoff
        self.kernel = kernel
        
    def fit(self,train,labels):
        self.model = svm.SVR(C=self.C,kernel=self.kernel)
        self.model.fit(train,labels)
        return self

    def predict(self,test):
        pred = self.model.predict(test)
        pred = np.ones(pred.shape)*(pred>=self.cutoff)
        return pred
    
class QDA_wrapper(BaseEstimator,ClassifierMixin):

    def __init__(self,cutoff=.5,reg_param=.01):
        self.cutoff= cutoff
        self.reg_param = reg_param
        
    def fit(self,train,labels):
        self.model = QuadraticDiscriminantAnalysis(reg_param=self.reg_param)
        self.model.fit(train,labels)
        return self

    def predict(self,test):
        pred = 1-self.model.predict_proba(test)[:,0]
        pred = np.ones(pred.shape)*(pred>=self.cutoff)
        return pred
    
class XGB_wrapper(BaseEstimator,ClassifierMixin):

    def __init__(self,cutoff=.5,gamma=0,learning_rate=.1,n_estimators=100):
        self.cutoff = cutoff
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        
    def fit(self,train,labels):
        self.model = xgb.XGBClassifier(gamma=self.gamma,learning_rate=self.learning_rate,n_estimators=self.n_estimators)
        self.model.fit(train,labels)
        return self

    def predict(self,test):
        pred = 1-self.model.predict_proba(test)[:,0]
        pred = np.ones(pred.shape)*(pred>=self.cutoff)
        return pred