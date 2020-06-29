import numpy as np
from numpy import linalg as la
import pandas as pd
import sys
from scipy import sparse as sp
#sys.path.append('/Users/mitchellsailsbery/Dropbox/Sailsbery Robertson Group/Springville')
#sys.path.append('/Users/connorrobertson/Dropbox/Sailsbery_Robertson_Group/Springville')
#import appendix

def F_value(pred,actual):
    '''pred,actual - numpy array, break = 1, not break = 0
    '''
    TP = np.sum((pred>0)*(actual>0))
    FP = np.sum((pred>0)*(actual<1))
    FN = np.sum((pred<1)*(actual>0))
    if TP+FP+FN>0:
        return 2*TP/(2*TP+FP+FN)
    else:
        return 0

def F_back(actual,pred):
    '''pred,actual - numpy array, break = 1, not break = 0
    '''
    TP = np.sum((pred>0)*(actual>0))
    FP = np.sum((pred>0)*(actual<1))
    FN = np.sum((pred<1)*(actual>0))
    if TP+FP+FN>0:
        return 2*TP/(2*TP+FP+FN)
    else:
        return 0

def TP(pred,actual):
    return np.sum((pred>0)*(actual>0))

def FP(pred,actual):
    return np.sum((pred>0)*(actual<1))

def FN(pred,actual):
    return np.sum((pred<1)*(actual>0))

def Accuracy(pred,actual):
    return np.sum(np.abs(pred-actual))/len(pred)

def Utility(pred,actual,budget = 100,a = 5):
    P = np.sum(pred>0)
    FN = np.sum((pred<1)*(actual>0))
    return min(-P+budget,0)-a*FN

def precise(pred,actual):
    return actual

def dist_thresh(pred,actual,test,thresh = 500):
    '''Note that test must include the breaks column'''
    output = np.copy(actual)
    
    for i in range(pred.shape[0]):
        
        if pred[i] > 0:
            
            
            row = test.iloc[i]
            x1,y1,x2,y2 = row.startx,row.starty,row.endx,row.endy
            start_coords = np.array([x1,y1])
            end_coords = np.array([x2,y2])
            newdf = test[(np.abs(test.startx-x1)<thresh) | (np.abs(test.starty-y1)<thresh) | (np.abs(test.endx-x2)<thresh) | (np.abs(test.endy-y2)<thresh)]
            
            newdf['dist1'] = la.norm(newdf[['startx','starty']]-start_coords, axis = 1)
            newdf['dist2'] = la.norm(newdf[['startx','starty']]-end_coords, axis = 1)
            newdf['dist3'] = la.norm(newdf[['endx','endy']]-start_coords, axis = 1)
            newdf['dist4'] = la.norm(newdf[['endx','endy']]-end_coords, axis = 1)
            
            close_break = newdf[newdf.dist1 < thresh].breaks.sum() + newdf[newdf.dist2 < thresh].breaks.sum() + newdf[newdf.dist3 < thresh].breaks.sum() + newdf[newdf.dist4 < thresh].breaks.sum()
            
            if close_break > 0:
                
                output[i] = 1
                
    return output
    
def metric_thresh(pred,actual,test,thresh = 500):
    '''Note that test must include the breaks column'''
    output = np.copy(actual)
    
    for i in range(pred.shape[0]):
        
        if pred[i] > 0:
            row1 = test.iloc[i]
            obj1 = row1.objectID
            dist_col = []
            
            for j in range(pred.shape[0]):
                row2 = test.iloc[j]
                dist_col.append(appendix.d(obj1,row2.objectID,test))
                
            
            test['dist'] = np.array(dist_col)
            
            close_break = np.sum(test[test.dist < thresh].breaks.values)
                        
            if close_break > 0:
                
                output[i] = 1
                
    return output
    
def type_success(pred,actual,test):
    used_ids = set()
    output = np.copy(actual)
    
    for i in range(pred.shape[0]):
        
        if pred[i] > 0:
            
            row = test.iloc[i]
            dia, mat, yr, leng = row.diameter, row.material, row.installyear, row.length
            narrowed = test[(test.diameter == dia) & (test.material == mat) & (test.installyear == yr)]
            narrowed_new = narrowed[np.invert(narrowed.objectID.isin(used_ids))]
            narrower = narrowed_new[np.abs(narrowed_new.length-leng) < narrowed_new.length/10]
            
            close_break = np.sum(narrower.breaks.values)
            
            if close_break > 0:
                used_index = np.argmax(narrower.breaks > 0)
                used_id = narrower.objectID[used_index]
                used_ids.add(used_id)
                output[i] = 1
                
    return output

def in_component(pred,actual,test_inds,n=3):
    """Input: pred (dict of pipe numbers and guesses)
    actual (dict of pipe numbers and truth)
    test_inds (indices in the order in which they appear in the test set)"""
    predc = pred
    data = np.load('springville_adj_matrix.npz')
    adj = sp.csr_matrix((data['data'],data['indices'],data['indptr']),shape = data['shape'])
    adj = adj.expm1()
    print('ready')
    
    def get_paths(pipe_num):
        """Input pipe_num (int):
        Output associated_pipes (dict of pipes and distances)"""
        level = adj.getrow(pipe_num)
        dists = [level.toarray().ravel()]
        for j in range(adj.shape[1]):
            dists.append(level.multiply(adj.getcol(j).transpose()).toarray().ravel())
        dists = np.concatenate(dists)
        zeros = np.argmax(np.sort(dists)>0.1)
        print(np.argsort(dists)[::-1][:20])
        print(np.sort(dists)[::-1][:20])

        best = np.argsort(dists)[zeros:zeros+n]
        bestr = [i%8469 for i in best]
        c = 1

        while len(np.unique(np.array(bestr))) < n:
            print('c',c,len(np.unique(np.array(bestr))),best,bestr)
            best = np.argsort(dists)[zeros:zeros+n+c]
            bestr = [i%8469 for i in best]
            c += 1

        bdists = dict()
        for j in best:
            k = j%8469
            if k in bdists.keys():
                if dists[j] < bdists[k]:
                    bdists[k] = np.log1p(dists[j])
            else:
                bdists[k] = np.log1p(dists[j])


        bestr = np.unique(bestr)

        return bdists
    
    guess_inds = np.argsort(predc)[-n:]
    print('set')
    
    for ind in guess_inds:
        print('go')
        associated = get_paths(ind).keys()
        for ind2 in associated:
            predc[ind2] = 1
    
    
    
    tp1 = TPd(predc,actual)
    fn1 = FNd(predc,actual)
    
    tp = TPd(predc[test_inds],actual[test_inds])
    fn = FNd(predc[test_inds],actual[test_inds])
    return tp/(tp+fn), tp1/(tp1+fn1)

def TPd(pred,actual):
    ind = list(pred.keys())
    p = pred[ind]
    a = actual[ind]
    return np.sum((pred>0)*(actual>0))

def FNd(pred,actual):
    ind = list(pred.keys())
    p = pred[ind]
    a = actual[ind]
    return np.sum((pred<1)*(actual>0))