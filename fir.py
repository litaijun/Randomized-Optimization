
# coding: utf-8

# In[ ]:


import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sklearn.model_selection as ms
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

d = pd.read_csv('loan_data.csv')
ldX = d.iloc[:,:11]
ldY = d.iloc[:,11]
Y[0:5]

ld_trgX, ld_tstX, ld_trgY, ld_tstY = ms.train_test_split(ldX, ldY, test_size=0.3, random_state=0,stratify=ldY)
pipe = Pipeline([('Scale',StandardScaler())])
trgX = pipe.fit_transform(ld_trgX,ld_trgY)
trgY = np.atleast_2d(ld_trgY).T
tstX = pipe.transform(ld_tstX)
tstY = np.atleast_2d(ld_tstY).T
trgX, valX, trgY, valY = ms.train_test_split(trgX, trgY, test_size=0.2, random_state=1,stratify=trgY)
tst = pd.DataFrame(np.hstack((tstX,tstY)))
trg = pd.DataFrame(np.hstack((trgX,trgY)))
val = pd.DataFrame(np.hstack((valX,valY)))

tst.to_csv('ld_test.csv',index=False,header=False)
trg.to_csv('ld_trg.csv',index=False,header=False)
val.to_csv('ld_val.csv',index=False,header=False)

