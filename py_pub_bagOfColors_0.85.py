from multiprocessing import Pool, cpu_count
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
import numpy as np
import glob
import cv2

ftwo_scorer = make_scorer(fbeta_score, beta=2, average='samples')

def get_im_cv2(path):
    img = cv2.imread(path)
    m, s = cv2.meanStdDev(img)
    img = cv2.resize(img, (20, 20), cv2.INTER_LINEAR)
    img = np.append(img.flatten(), m.flatten())
    img = np.append(img, s.flatten())
    return [path, img]

def normalize_image_features(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_im_cv2, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    return fdata

train_jpg = glob.glob('../input/train-jpg/*')
train_tags = pd.read_csv('../input/train.csv')
train = pd.DataFrame([[p.split('/')[3].replace('.jpg',''),p] for p in train_jpg], columns = ['image_name','path'])
train_tags = pd.concat([train_tags, train_tags['tags'].str.get_dummies(sep=' ')], axis=1)
train = pd.merge(train_tags, train, how='left', on='image_name')
y = train[[c for c in train if c not in ['image_name','path','tags']]]
xtrain = normalize_image_features(train['path']); print('train...')

test_jpg = glob.glob('../input/test-jpg/*')
test = pd.DataFrame([[p.split('/')[3].replace('.jpg',''),p] for p in test_jpg], columns = ['image_name','path'])
xtest = normalize_image_features(test['path']); print('test...')

etr = ExtraTreesRegressor(n_estimators=15, max_depth=15, n_jobs=-1, random_state=1)
etr.fit(xtrain, y); print('fit...')
train_pred = etr.predict(xtrain)
train_pred2 = etr.predict(xtrain)
best_score=0.0
cutoff=-1
for i in range(0,100):
    num=(i/100.0)
    train_pred2[train_pred > num] = 1
    train_pred2[train_pred <= num] = 0
    x=fbeta_score(y,train_pred2,beta=2, average='samples')
    if(x>best_score):
        best_score=x
        cutoff=num
    print(x, i)

pred = etr.predict(xtest); print('predict...')

print (pred)

tags = []
for r in pred:
    r = list(r)
    tags.append(' '.join([j[1] for j in sorted([[r[i],y.columns[i]] for i in range(len(y.columns)) if r[i]>cutoff], reverse=True)]))

test['tags'] = tags
test[['image_name','tags']].to_csv('submission_bag_of_colors.csv', index=False)
test.head()