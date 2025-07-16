import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier

import yaml
prmtr= yaml.safe_load(open('Params.yaml','r'))['model_building']

train_data = pd.read_csv('data/features/train_tfidf.csv')

X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

# Deine and train the model
clf = GradientBoostingClassifier(n_estimators=prmtr['n_estimators'],learning_rate=prmtr['learning_rate'], random_state=42)
clf.fit(X_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))

