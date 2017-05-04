# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:22:08 2017

@author: gs
"""

# ------------------------------------------------------------
# results
# ------------------------------------------------------------
#
# eta = 0.3  -> 0.87892 LB (predicted 0.88 ?)
# eta = 0.05 -> (predicted 0.8916)

# ------------------------------------------------------------
# imports and constants
# ------------------------------------------------------------
from skimage import io
#from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import xgboost as xgb
from sklearn import model_selection
from sklearn.metrics import fbeta_score


path1 = '/home/gs/DataScientist/planet'
trainPath = '/train-tif'
testPath = '/test-tif'

MODEL_NUMBER = 41

VERBOSE_INTERVAL = 5000

NUM_BINS = 64
MAX_PIX_VAL = 65535



# ------------------------------------------------------------
# definitions
# ------------------------------------------------------------
def getImageHistograms (filePath):
    try:
        img = io.imread(filePath)
        r, g, b, nir = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
        n = np.true_divide(nir-r, nir+r)
        hr, bins = np.histogram(r,NUM_BINS,[0, MAX_PIX_VAL])
        hg, bins = np.histogram(g,NUM_BINS,[0, MAX_PIX_VAL])
        hb, bins = np.histogram(b,NUM_BINS,[0, MAX_PIX_VAL])
        hnir, bins = np.histogram(nir,NUM_BINS,[0, MAX_PIX_VAL])
        ndvi, bins = np.histogram(n,NUM_BINS,[-1, +1])
    except Exception as e:
        print ('  error {} reading file {}'.format(e, filePath))
        hr = np.zeros(NUM_BINS)
        hg = np.zeros(NUM_BINS)
        hb = np.zeros(NUM_BINS)
        hnir = np.zeros(NUM_BINS)
        ndvi = np.zeros(NUM_BINS)
        
    return hr, hg, hb, hnir, ndvi

#getImageHistograms('/home/gs/DataScientist/planet/train-tif/train_3.tif')

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=2000):
    br = 0
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.05
    param['max_depth'] = 8
    param['silent'] = 1
    param['eval_metric'] = "logloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20, verbose_eval = 50)
        br = model.best_iteration
        #print ('best iteration for DICT: {}'.format(br))
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds, verbose_eval = 50)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model, br

def mapf (arr):
    res = ''
    for i in range(0,17):
        if arr[i] == 1:
            res += inv_label_map[i] + ' '
    res = res.rstrip()
    return res
    
def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*17
  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= float(resolution)
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x, best_score


# ------------------------------------------------------------
# read train y
# ------------------------------------------------------------
print ('read train y...')

try:
    Y_train = pd.read_csv(path1+'/train.csv')
except:
    path1 = '/home/ec2-user/DataScientist/planet'
    Y_train = pd.read_csv(path1+'/train.csv')

print ('Y_train lines read: {}'.format(len(Y_train)))

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in Y_train['tags'].values])))
label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}
#print(label_map)
#print
#print(inv_label_map)

Y_trainDict = {}
for i, row in Y_train.iterrows():
    name = row['image_name']
    tags = row['tags']
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    Y_trainDict[name] = targets


# ------------------------------------------------------------
# read train
# ------------------------------------------------------------
X_train = [] # arrays

Y_trainAll = []
X_train_id = []
lines = 0

print('Read train images')
path = os.path.join(path1+trainPath, '*.tif')
print (path)
files = glob.glob(path)
for fl in files:
    lines += 1
    if lines % VERBOSE_INTERVAL == 0:
        print ('  files read: {}'.format(lines))
    flbase = os.path.basename(fl)
    a,b,c,d,e = getImageHistograms(fl)
    r = np.concatenate((a, b, c, d, e), axis = 0)
    X_train.append(r)
    name = flbase.replace('.tif', '')
    X_train_id.append(name)
    Y_trainAll.append(Y_trainDict[name])


Y_trainAll = pd.DataFrame(Y_trainAll)

X_train = pd.DataFrame(X_train)
print ('X_train shape   : {}'.format(X_train.shape))
print ('Y_trainAll shape: {}'.format(Y_trainAll.shape))


# ------------------------------------------------------------
# create validation set
# ------------------------------------------------------------
VALSIZE = 35000
xTrain = X_train.iloc[:VALSIZE,:]
xValidation = X_train.iloc[VALSIZE:,:]
yTrain = Y_trainAll.iloc[:VALSIZE,:]
yValidation = Y_trainAll.iloc[VALSIZE:,:]

# ------------------------------------------------------------
# cross validation validation set
# ------------------------------------------------------------
brDict = {}
print ('cross validation validation set')
for i in range(0,17):
    print ('  target: {} {}'.format(i, inv_label_map[i]))
    Y_train = yTrain.ix[:,i]

    kf = model_selection.KFold(n_splits=3, shuffle=True, random_state=2016)
    for dev_index, val_index in kf.split(range(xTrain.shape[0])):
        dev_X, val_X = xTrain.iloc[dev_index], xTrain.iloc[val_index]
        dev_y, val_y = Y_train.iloc[dev_index], Y_train.iloc[val_index]
        preds, model, br = runXGB(dev_X, dev_y, val_X, val_y)
        brDict[i] = br

print ('bestRounds:')
print (brDict)




# ------------------------------------------------------------
# predict validation
# ------------------------------------------------------------
print ('predict validation')
predsDF = pd.DataFrame()
for i in range(0,17):
    print ('  predicting feature ' + str(i))
    Y_train = yTrain.ix[:,i]
    #print (Y_train.shape)
    preds, model, br = runXGB(xTrain, Y_train, xValidation, num_rounds=int(brDict[i]*1.33))
    predsDF[i] = preds

print (predsDF.shape)


# ------------------------------------------------------------
# find best thresholds and store / score model!
# ------------------------------------------------------------
thresholds, predictedScore = optimise_f2_thresholds(np.array(yValidation), np.array(predsDF))
print (thresholds, predictedScore)


# ------------------------------------------------------------
# cross validation whole train
# ------------------------------------------------------------
# save number of rounds!
print ('cross validation whole train...')
brDict = {}

for i in range(0,17):
    print ('  target: {} {}'.format(i, inv_label_map[i]))
    Y_train = Y_trainAll.ix[:,i]

    kf = model_selection.KFold(n_splits=3, shuffle=True, random_state=2016)
    for dev_index, val_index in kf.split(range(X_train.shape[0])):
        dev_X, val_X = X_train.iloc[dev_index], X_train.iloc[val_index]
        dev_y, val_y = Y_train.iloc[dev_index], Y_train.iloc[val_index]
        preds, model, br = runXGB(dev_X, dev_y, val_X, val_y)
        brDict[i] = br

print ('bestRounds:')
print (brDict)



# ------------------------------------------------------------
# read test
# ------------------------------------------------------------
X_test = [] # arrays
X_test_id = []
lines = 0 

print('Read test images')
path = os.path.join(path1+testPath, '*.tif')
print (path)
files = glob.glob(path)
for fl in files:
    lines += 1
    if lines % VERBOSE_INTERVAL == 0:
        print ('  files read: {}'.format(lines))
    flbase = os.path.basename(fl)
    a,b,c,d,e = getImageHistograms(fl)
    r = np.concatenate((a, b, c, d, e), axis = 0)
    X_test.append(r)
    name = flbase.replace('.tif', '')
    X_test_id.append(name)

X_test = pd.DataFrame(X_test)
print (X_test.shape)



# ------------------------------------------------------------
# predict test
# ------------------------------------------------------------
predsDF = pd.DataFrame()
for i in range(0,17):
    print ('predicting feature ' + str(i))
    Y_train = Y_trainAll.ix[:,i]
    #print (Y_train.shape)
    preds, model, br = runXGB(X_train, Y_train, X_test, num_rounds=int(brDict[i]*1.33))
    predsDF[i] = preds

print (predsDF.shape)



# ------------------------------------------------------------
# apply thresholds
# ------------------------------------------------------------
print (predsDF.head())
temp = predsDF.copy()
for i in range (0,17):
    temp[i] = temp[i] > thresholds[i]
print (temp.head())


# ------------------------------------------------------------
# create and save prediction file
# ------------------------------------------------------------
textResults = []

for i, row in temp.iterrows():
    #print (i)
    #print (list(row))
    textResults.append ( mapf( list (row)))
    
print (textResults[0:5])

res = pd.DataFrame()
res['image_name'] = X_test_id
res['tags'] = textResults

print (res.head())

res.to_csv(path1+'SUB_'+ MODEL_NUMBER +'.csv', index=False)


# ------------------------------------------------------------
# save raw file
# ------------------------------------------------------------
predsDF['id'] = X_test_id
predsDF.to_csv(path1+'RAW_'+ MODEL_NUMBER +'.csv', index=False)


# ------------------------------------------------------------
# save thresholds
# ------------------------------------------------------------

thrDF = pd.DataFrame(thresholds)
thrDF.to_csv(path1+'THR_'+ MODEL_NUMBER +'.csv', index=False)

print ('predicted score: {}'.format(predictedScore))

