import pickle 
import pandas as pd
import json 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

clf = pickle.load(open('model.pkl','rb'))
test_data = pd.read_csv('data/features/test_tfidf.csv')

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values


y_perd = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_perd)
precision = precision_score(y_test, y_perd)
recall = recall_score(y_test, y_perd)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'auc': auc
}       

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)
