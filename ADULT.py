import pandas as pd
# pd.options.display.max_columns = 500
import numpy as np
import os

import datetime, time
import warnings
import multiprocessing as mp
from multiprocessing import Pool

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from skopt import BayesSearchCV
from skopt.space import Real, Integer

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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

samples, features = ADULT.shape

to_dummy = []
for col in range(features):
    if ADULT.iloc[:, col].dtype != int:
        to_dummy.append(col)
        u = ADULT.iloc[:, col].unique()
        ADULT.iloc[:, col] = ADULT.iloc[:, col].replace(u, np.arange(len(u)))
        
to_dummy = to_dummy[:-1] # don't include the last one, our target

obj_cols = ADULT.iloc[:, to_dummy].astype(object)
ADULT_ohe = pd.get_dummies(obj_cols, prefix=np.array(headers)[to_dummy])
ADULT_ohe = pd.concat([ADULT.drop(ADULT.columns[to_dummy], axis=1).iloc[:, :-1], ADULT_ohe], axis=1)

ss = StandardScaler()

ADULT_ohe = pd.DataFrame(ss.fit_transform(ADULT_ohe), columns=ADULT_ohe.columns)
ADULT_ohe.iloc[:, -1] = (ADULT_ohe.iloc[:, -1] > 0).astype(int)

def log(message):
    t = datetime.datetime.now()
    
    with open('ADULT.log', 'a') as logf:
        if isinstance(message, str):
            logf.write(f'[{t}] {message}\n')
        elif isinstance(message, list):
            for i, m in enumerate(message):
                logf.write(f'[{t}] ({i}) {m}\n')
        

def log_results(partition, classifier, training, validation, testing, params, time_delta):
    
    dur = str(datetime.timedelta(seconds=time_delta))
    
    res = []
    for key in params:
        res.append(f'{key}: {params[key]}')
        
    log([f'RESULTS [{dur}]: clf: {classifier}, p: {partition}, train: {training:.4f}, val: {validation:.4f}, test: {testing:.4f}',
         f'BEST_PARAMS: {", ".join(res)}'])

# Initialize KFold    

folds = 5

kf = KFold(n_splits=folds, shuffle=True)

# Initialize Classifiers

models = [
    ('xgb', xgb.XGBClassifier()),
    ('rfc', RandomForestClassifier(n_estimators=1024, n_jobs=-1, criterion='entropy')),
    ('ada', AdaBoostClassifier())
]

param_space = [
    ({'n_estimators': Integer(2, 1024)}, 16),
    ({'max_features': Integer(1, features-1)}, 20),
    ({'n_estimators': Integer(2, 1024), 'learning_rate': Real(1e-6, 100, prior='log-uniform')}, 16)
]

models_ohe = [
    ('svc_lin', LinearSVC()),
    ('svc_rbf', SVC(kernel='rbf')),
    ('logreg', LogisticRegression(penalty='l2', n_jobs=3)),
    ('knn', KNeighborsClassifier(n_jobs=2)),
    ('logreg', LogisticRegression(penalty='l2', n_jobs=-1)),
]

param_space_ohe = [
    ({'C': Real(1e-7, 1000, prior='log-uniform')}, 16),
    ({'gamma': Real(0.001, 2, prior='log-uniform'), 'C': Real(1e-7, 1000, prior='log-uniform')}, 12),
    ({'C': Real(1e-8, 1e+4, prior='log-uniform')}, 16),
    ({'weights': ['uniform', 'distance']}, 12),
    ({'C': Real(1e-8, 1e+4, prior='log-uniform')}, 16),
]

trials = 3
partitions = [0.2,0.5,0.8]
early_stop_tol = 1e-5
stop_after_iters = 7

def train_val(classifier, name, partition, trial, params, n_iter, X_train, y_train):
    
    bscv = BayesSearchCV(classifier, params, n_iter=n_iter, cv=kf, scoring='f1', return_train_score=True, n_jobs=3)
        
    log(f'Making {bscv.total_iterations} iterations on {name}_{trial+1} ({partition})')
    
    total_iters = [0]
    prior_scores = []
    
    def on_step(optim_result):
        total_iters[0] += 1
        prior_scores.append(bscv.best_score_)
        
        log(f'{name}{total_iters}[{trial+1}] current best score: {bscv.best_score_:.4f}')
        
        if total_iters[0] >= stop_after_iters:
            if bscv.best_score_ == 1.0 or (np.mean(prior_scores[-stop_after_iters:]) - np.mean(prior_scores[-(stop_after_iters+1):-1])) < early_stop_tol:
                log(f'{name}{total_iters}[{trial+1}] stopped early')
                return True
    
    bscv.fit(X_train, y_train, callback=on_step)
        
    return bscv

def test(estimator, X_test, y_test):
    return estimator.score(X_test, y_test)

def run_trial(trial_num):
    
    for partition in partitions:
        
        # since random can't play nicely with multiprocessing
        # https://stackoverflow.com/questions/24345637/why-doesnt-numpy-random-and-multiprocessing-play-nice
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        
        ixs = np.arange(samples)
        np.random.shuffle(ixs)
        
        shuffled = ADULT.iloc[ixs]
        shuffled_ohe = ADULT_ohe.iloc[ixs]
        
        split = int(samples * partition)

        X_train_val = shuffled.iloc[:split, :-1]
        X_train_val_ohe = shuffled_ohe.iloc[:split, :]
        y_train_val = shuffled.iloc[:split, -1]

        X_test = shuffled.iloc[split:, :-1]
        X_test_ohe = shuffled_ohe.iloc[split:, :]
        y_test = shuffled.iloc[split:, -1]
                
        log(f'TRIAL #{trial_num}, TRAIN SZ: {split}, TEST SZ: {samples-split}')

        for (name, clf), (params, n_iter) in zip(models, param_space):
                            
            start_time = time.time()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                gs = train_val(clf, name, partition, trial_num, params, n_iter, X_train_val, y_train_val)
                t_score = test(gs.best_estimator_, X_test, y_test)
                
            end_time = time.time()

            log_results(partition, name, 
                        max(gs.cv_results_['mean_train_score']), 
                        max(gs.cv_results_['mean_test_score']),
                        t_score,
                        gs.best_params_,
                        end_time - start_time)
            
        for (name, clf), (params, n_iter) in zip(models_ohe, param_space_ohe):
            
            if name == 'knn':
                params['n_neighbors'] = Integer(1,int(split*(folds-1)/folds))
                            
            start_time = time.time()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                gs = train_val(clf, name, partition, trial_num, params, n_iter, X_train_val_ohe, y_train_val)
                t_score = test(gs.best_estimator_, X_test_ohe, y_test)
                
            end_time = time.time()

            log_results(partition, name, 
                        max(gs.cv_results_['mean_train_score']),
                        max(gs.cv_results_['mean_test_score']),
                        t_score,
                        gs.best_params_,
                        end_time - start_time)
        
with Pool(processes=4) as pool:
    log('==== ADULT CV ====')
    start_time = time.time()
    pool.map(run_trial, range(trials))
    dur = str(datetime.timedelta(seconds=(time.time() - start_time)))
    print('==== CV Complete ====')
    log([f'Total Time: {dur}', '==== CV Complete ===='])
