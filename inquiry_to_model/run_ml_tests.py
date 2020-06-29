import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from numpy import linalg as la
from Metrics_Success import Accuracy, dist_thresh, F_value, precise, Utility, TP, FP, FN
import sklearn.naive_bayes as nb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
import modeling as md
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
import warnings
warnings.filterwarnings('ignore')

spbool = False
orbool = True

data = pd.read_csv('pipes.csv')
pipes = data.drop(['endx','endy','startx',
                   'starty','springville','orem','breaks','elevation','mukey'],axis = 1)
breaks = data.breaks
breaks = -.5*(breaks-1)*(breaks-2)+1

#Cutoffs
model1 = md.modeling(QuadraticDiscriminantAnalysis,data,pipes,breaks,springville=spbool,orem=orbool,
	classifier=True,ftimp = True, coef=False,reg_param=.1)
cutoff1,max1 = model1._cutoff(n=1000,max_val=True)

model2 = md.modeling(LinearDiscriminantAnalysis,data,pipes,breaks,springville=spbool,orem=orbool,classifier=True,
                    ftimp = True, coef=False)
cutoff2,max2 = model2._cutoff(n=1000,max_val=True)

model3 = md.modeling(svm.SVC,data,pipes,breaks,springville=spbool,orem=orbool,classifier = True,
                    ftimp = True, coef=False,kernel='poly',C = 10,probability=True,class_weight = 'balanced')
cutoff3,max3 = model3._cutoff(n=20,max_val=True)

model4 = md.modeling(nb.BernoulliNB,data,pipes,breaks,springville=spbool,orem=orbool,classifier=True,
                    ftimp = True, coef=False)
cutoff4,max4 = model4._cutoff(n=1000,max_val=True)

model5 = md.modeling(xgb.XGBClassifier,data,pipes,breaks,springville=spbool,orem=orbool,classifier=True,
                    ftimp = True, coef=False,n_estimators = 1000,scale_pos_weight = .1)
cutoff5,max5 = model5._cutoff(n=100,max_val=True)

model6 = md.modeling(RandomForestClassifier,data,pipes,breaks,springville=spbool,orem=orbool,classifier=True,
                    ftimp = True, coef=False,n_estimators = 1000,class_weight = 'balanced_subsample')
cutoff6,max6 = model6._cutoff(n=100,max_val=True)

model7 = md.modeling(Pipeline,data,pipes,breaks,springville=spbool,orem=orbool,coef=True,
                     steps=[('poly', PolynomialFeatures(degree=3)),('linear', lm.ElasticNetCV())])
cutoff7,max7 = model7._cutoff(n=20,max_val=True)

model8 = md.modeling(Pipeline,data,pipes,breaks,springville=spbool,orem=orbool,classifier=True,
                    ftimp = True, coef=False,steps=[('poly', PolynomialFeatures(degree=3)),('QDA',QuadraticDiscriminantAnalysis(reg_param=.00075))])
cutoff8,max8 = model8._cutoff(n=1000,max_val=True)

model9 = md.modeling(Pipeline,data,pipes,breaks,springville=spbool,orem=orbool,classifier=True,
                    ftimp = True, coef=False,steps=[('poly', PolynomialFeatures(degree=3)),('LDA',LinearDiscriminantAnalysis())])
cutoff9,max9 = model9._cutoff(n=1000,max_val=True)

model10 = md.modeling(Pipeline,data,pipes,breaks,springville=spbool,orem=orbool,classifier=True,
                    ftimp = True, coef=False,steps=[('poly', PolynomialFeatures(degree=3)),('SVM',svm.SVC(kernel='poly',C = 10,probability=True,class_weight = 'balanced'))])
cutoff10,max10 = model10._cutoff(n=20,max_val=True)

model11 = md.modeling(Pipeline,data,pipes,breaks,springville=spbool,orem=orbool,classifier=True,
                    ftimp = True, coef=False,steps=[('poly', PolynomialFeatures(degree=3)),('NB',nb.BernoulliNB())])
cutoff11,max11 = model11._cutoff(n=1000,max_val=True)

model12 = md.modeling(Pipeline,data,pipes,breaks,springville=spbool,orem=orbool,classifier=True,
                    ftimp = True, coef=False,steps=[('poly', PolynomialFeatures(degree=3)),('XGB',xgb.XGBClassifier(n_estimators=1000,scale_pos_weight = .1))])
cutoff12,max12 = model12._cutoff(n=10,max_val=True)

model13 = md.modeling(Pipeline,data,pipes,breaks,springville=spbool,orem=orbool,classifier=True,
                    ftimp = True, coef=False,steps=[('poly', PolynomialFeatures(degree=3)),('RF',RandomForestClassifier(n_estimators=1000,class_weight = 'balanced_subsample'))])
cutoff13,max13 = model13._cutoff(n=10,max_val=True)

model14 = md.modeling(MLPClassifier,data,pipes,breaks,springville=spbool,orem=orbool,ftimp=True,coef=False,classifier=True,max_iter=5000)
cutoff14,max14 = model14._cutoff(n=100,max_val=True)

model15 = md.modeling(Pipeline,data,pipes,breaks,springville=spbool,orem=orbool,classifier=True,
                    ftimp = True, coef=False,steps=[('poly', PolynomialFeatures(degree=3)),('NN',MLPClassifier(max_iter=5000))])
cutoff15,max15 = model15._cutoff(n=100,max_val=True)

model16 = md.modeling(lm.LinearRegression,data,pipes,breaks,springville=spbool,orem=orbool,ftimp=True,coef=False)
cutoff16,max16 = model16._cutoff(n=1000,max_val=True)

model17 = md.modeling(KNeighborsClassifier,data,pipes,breaks,springville=spbool,orem=orbool,ftimp=True,coef=False,classifier=True)
cutoff17,max17 = model17._cutoff(n=100,max_val=True)

model18 = md.modeling(LabelPropagation,data,pipes,breaks,springville=spbool,orem=orbool,classifier=True)
cutoff18,max18 = model18._cutoff(n=20,max_val=True)

np.save('cutoffs.npy',np.array([cutoff1,cutoff2,cutoff3,cutoff4,cutoff5,cutoff6,cutoff7,cutoff8,cutoff9,cutoff10,cutoff11,cutoff12,cutoff13,cutoff14,cutoff15,cutoff16,cutoff17,cutoff18]))

max_sum = max1+max2+max3+max4+max5+max6+max7+max8+max9+max10+max11+max12+max13+max14+max15+max16+max17+max18
ensemble_fscore = np.array([0.,0.,0.,0.,0.,0.,0.])
pure_fscore = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
ensemble_TP = np.array([0.,0.,0.,0.,0.,0.,0.])
pure_TP = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
ensemble_FP = np.array([0.,0.,0.,0.,0.,0.,0.])
pure_FP = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
ensemble_FN = np.array([0.,0.,0.,0.,0.,0.,0.])
pure_FN = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

n = 200
cov_matrices = []
for i in range(n):
    train,test,ans,sols = model1._split()
    
    p1 = model1.ens_pred(train,test,ans,cutoff = cutoff1)
    p2 = model2.ens_pred(train,test,ans,cutoff = cutoff2)
    p3 = model3.ens_pred(train,test,ans,cutoff = cutoff3)
    p4 = model4.ens_pred(train,test,ans,cutoff = cutoff4)
    p5 = model5.ens_pred(train,test,ans,cutoff = cutoff5)
    p6 = model6.ens_pred(train,test,ans,cutoff = cutoff6)
    p7 = model7.ens_pred(train,test,ans,cutoff = cutoff7)
    p8 = model8.ens_pred(train,test,ans,cutoff = cutoff8)
    p9 = model9.ens_pred(train,test,ans,cutoff = cutoff9)
    p10 = model10.ens_pred(train,test,ans,cutoff = cutoff10)
    p11 = model11.ens_pred(train,test,ans,cutoff = cutoff11)
    p12 = model12.ens_pred(train,test,ans,cutoff = cutoff12)
    p13 = model13.ens_pred(train,test,ans,cutoff = cutoff13)
    p14 = model14.ens_pred(train,test,ans,cutoff = cutoff14)
    p15 = model15.ens_pred(train,test,ans,cutoff = cutoff15)
    p16 = model16.ens_pred(train,test,ans,cutoff = cutoff16)
    p17 = model17.ens_pred(train,test,ans,cutoff = cutoff17)
    p18 = model18.ens_pred(train,test,ans,cutoff = cutoff18)
    
    
    p01 = ((p1+p2+p3+p4+p5+p6+p7) > 3)
    p02 = ((p2+p4+p5+p7) > 2)
    p03 = ((p2+p4+p5+p7) >=2)
    p04 = ((p2+p3+p4+p5+p6+p7) > 3)
    p05 = ((p2+p3+p4+p5+p6+p7) >= 3)
    p06 = ((.1*p1+.3*p2+.2*p3+.4*p4+.3*p5+.2*p6+.3*p7) >=.9)
    p07 = ((max1*p1+max2*p2+max3*p3+max4*p4+max5*p5+max6*p6+max7*p7+max8*p8+max9*p9+max10*p10+max11*p11+max12*p12+max13*p13+max14*p14+max15*p15+max16*p16+max17*p17+max18*p18)/max_sum >= .5*max_sum)
    
    ensemble_fscore += np.array([F_value(p01,sols),F_value(p02,sols),
                            F_value(p03,sols),F_value(p04,sols),
                            F_value(p05,sols),F_value(p06,sols),
                            F_value(p07,sols)])/n
    
    pure_fscore += np.array([F_value(p1,sols),F_value(p2,sols),
                            F_value(p3,sols),F_value(p4,sols),
                            F_value(p5,sols),F_value(p6,sols),
                            F_value(p7,sols),F_value(p8,sols),
                            F_value(p9,sols),F_value(p10,sols),
                            F_value(p11,sols),F_value(p12,sols),
                            F_value(p13,sols),F_value(p14,sols),
                            F_value(p15,sols),F_value(p16,sols),
                            F_value(p17,sols),F_value(p18,sols)])/n

    ensemble_TP += np.array([TP(p01,sols),TP(p02,sols),
                            TP(p03,sols),TP(p04,sols),
                            TP(p05,sols),TP(p06,sols),
                            TP(p07,sols)])/n
    
    pure_TP += np.array([TP(p1,sols),TP(p2,sols),
                            TP(p3,sols),TP(p4,sols),
                            TP(p5,sols),TP(p6,sols),
                            TP(p7,sols),TP(p8,sols),
                            TP(p9,sols),TP(p10,sols),
                            TP(p11,sols),TP(p12,sols),
                            TP(p13,sols),TP(p14,sols),
                            TP(p15,sols),TP(p16,sols),
                            TP(p17,sols),TP(p18,sols)])/n

    ensemble_FP += np.array([FP(p01,sols),FP(p02,sols),
                            FP(p03,sols),FP(p04,sols),
                            FP(p05,sols),FP(p06,sols),
                            FP(p07,sols)])/n
    
    pure_FP += np.array([FP(p1,sols),FP(p2,sols),
                            FP(p3,sols),FP(p4,sols),
                            FP(p5,sols),FP(p6,sols),
                            FP(p7,sols),FP(p8,sols),
                            FP(p9,sols),FP(p10,sols),
                            FP(p11,sols),FP(p12,sols),
                            FP(p13,sols),FP(p14,sols),
                            FP(p15,sols),FP(p16,sols),
                            FP(p17,sols),FP(p18,sols)])/n

    ensemble_FN += np.array([FN(p01,sols),FN(p02,sols),
                            FN(p03,sols),FN(p04,sols),
                            FN(p05,sols),FN(p06,sols),
                            FN(p07,sols)])/n
    
    pure_FN += np.array([FN(p1,sols),FN(p2,sols),
                            FN(p3,sols),FN(p4,sols),
                            FN(p5,sols),FN(p6,sols),
                            FN(p7,sols),FN(p8,sols),
                            FN(p9,sols),FN(p10,sols),
                            FN(p11,sols),FN(p12,sols),
                            FN(p13,sols),FN(p14,sols),
                            FN(p15,sols),FN(p16,sols),
                            FN(p17,sols),FN(p18,sols)])/n

    covs = np.vstack((p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18))
    np.save('cov{}.npy'.format(i),np.cov(covs))

np.save("Ensemble_F_values",ensemble_fscore)
np.save("Pure_F_values",pure_fscore)
np.save("Ensemble_TP",ensemble_TP)
np.save("Pure_TP",pure_TP)
np.save("Ensemble_FP",ensemble_FP)
np.save("Pure_FP",pure_FP)
np.save("Ensemble_FN",ensemble_FN)
np.save("Pure_FN",pure_FN)
