import pandas as pd
# pd.options.display.max_columns = 500
import numpy as np

import datetime
import warnings
from tqdm import tqdm
from multiprocessing import Pool

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

print('Libraries loaded')

headers = ['age',
            'workclass',
            'fnlwgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country',
            'above-50k']

ADULT = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                   names=headers, index_col=False, na_values=' ?')

ADULT.dropna(inplace=True)

print(f'ADULT shape: {ADULT.shape}')

to_dummy = []
for col in range(ADULT.shape[1]):
    if ADULT.iloc[:, col].dtype != int:
        to_dummy.append(col)
        u = ADULT.iloc[:, col].unique()
        ADULT.iloc[:, col] = ADULT.iloc[:, col].replace(u, np.arange(len(u)))
        
to_dummy = to_dummy[:-1] # don't include the last one, our target
        
def log(message):
    t = datetime.datetime.now()
    
    with open('ADULT_cv.log', 'a') as logf:
        if isinstance(message, str):
            logf.write(f'[{t}] {message}\n')
        elif isinstance(message, list):
            for i, m in enumerate(message):
                logf.write(f'[{t}] ({i}) {m}\n')
        

def log_results(partition, classifier, training, validation, testing, params):
    res = []
    for key in params:
        res.append(f'{key}: {params[key]}')
        
    log([f'RESULTS: clf: {classifier}, p: {partition}, train: {training:.4f}, val: {validation:.4f}, test: {testing:.4f}',
         f'BEST_PARAMS: {", ".join(res)}'])

# Initialize KFold    

folds = 5

kf = KFold(n_splits=folds, shuffle=True)

# Initialize Classifiers

rfc = RandomForestClassifier(n_estimators=1024, n_jobs=-1, criterion='entropy')
rfc_params = {'max_features': [1, 2, 4, 6, 8, 12, 16, 20]}

svc = SVC()
svc_params = {'kernel': ['linear', 'poly', 'rbf'], 
              'degree': [2, 3],
              'gamma': [0.001,0.01,0.1,1,2],
              'C': [10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 1.0]}

knn = KNeighborsClassifier(n_jobs=-1)
knn_params = {'weights': ['uniform', 'distance']}

boost = xgb.XGBClassifier()
boost_params = {'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]}

classifier_names = ['boost', 'knn', 'random forest', 'svc']
classifiers = [boost, knn, rfc, svc]
classifier_params = [boost_params, knn_params, rfc_params, svc_params]
trials = 3
partitions = [0.2, 0.5, 0.8]
total = ADULT.shape[0]

def grid_search(classifier, params):
    return GridSearchCV(classifier, params, cv=kf, return_train_score=True, scoring='f1')

def train_val(classifier, params, X_train, y_train):
    gs = grid_search(classifier, params)
    gs.fit(X_train, y_train)
        
    return gs

def test(estimator, X_test, y_test):
    return estimator.score(X_test, y_test)

def run_partitioned_data(partition):
        
    shuffled = ADULT.sample(frac=1)
    split = int(total * partition)

    obj_cols = shuffled.iloc[:, to_dummy].astype(object)
    shuffled_ohe = pd.get_dummies(obj_cols, prefix=np.array(headers)[to_dummy])
    shuffled_ohe = pd.concat([shuffled.drop(shuffled.columns[to_dummy], axis=1).iloc[:, :-1], shuffled_ohe], axis=1)

    X_train_val = shuffled.iloc[:split, :-1]
    X_train_val_ohe = shuffled_ohe.iloc[:split, :]
    y_train_val = shuffled.iloc[:split, -1]

    X_test = shuffled.iloc[split:, :-1]
    X_test_ohe = shuffled_ohe.iloc[split:, :]
    y_test = shuffled.iloc[split:, -1]
        
    for classifier, params, name in zip(classifiers, classifier_params, classifier_names):
        
        log(f'TRAIN: clf: {name}, size: {partition}/{split}, testing: {total - split}')
        
        training = []
        validation = []
        testing = []
        best_params = []
        
        if name == 'knn':
            params['n_neighbors'] = np.linspace(1, int(split * (folds-1)/folds), 10).astype(int)
        
        for trial in range(trials):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if name in ['svc', 'knn']:
                    gs = train_val(classifier, params, X_train_val_ohe, y_train_val)
                    t_score = test(gs.best_estimator_, X_test_ohe, y_test)
                elif name in ['random forest', 'boost']:
                    gs = train_val(classifier, params, X_train_val, y_train_val)
                    t_score = test(gs.best_estimator_, X_test, y_test)
                
            testing.append(t_score)         
            
            log(f'TRAINED: clf: {name}, partition: {partition}, trial: {trial + 1}, test: {t_score:.4f}')
                                    
            training.append(gs.cv_results_['mean_train_score'])
            validation.append(gs.cv_results_['mean_test_score'])
            best_params.append(gs.best_params_)
            
        training = max(np.array(training).mean(axis=0))
        validation = max(np.array(validation).mean(axis=0))
        testing = np.mean(testing)
        
        log_results(partition, name, training, validation, testing, best_params[np.argmax(testing)])
        
with Pool(processes=4) as pool:
    log('==== ADULT CV ====')
    pool.map(run_partitioned_data, partitions)
    print('==== CV Complete ====')
    log('==== CV Complete ====')