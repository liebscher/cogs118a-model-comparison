import pandas as pd
import numpy as np

# get shapes for all datasets. exclude the targets in calculations

#### ADULT

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
                   names=headers, index_col=False, na_values=' ?').iloc[:, :-1]

ADULT.dropna(inplace=True)

to_dummy = []
for col in range(ADULT.shape[1]):
    if ADULT.iloc[:, col].dtype != int:
        to_dummy.append(col)
        u = ADULT.iloc[:, col].unique()
        ADULT.iloc[:, col] = ADULT.iloc[:, col].replace(u, np.arange(len(u)))
        
obj_cols = ADULT.iloc[:, to_dummy].astype(object)
ADULT_ohe = pd.get_dummies(obj_cols, prefix=np.array(headers)[to_dummy])
ADULT_ohe = pd.concat([ADULT.drop(ADULT.columns[to_dummy], axis=1), ADULT_ohe], axis=1)

print('ADULT')
print(f'shape (w/o target): {ADULT.shape}')
print(f'ohe shape (w/o target): {ADULT_ohe.shape}')
print()

#### CELL

CELL_data = np.load('/Volumes/microSD/COGS118a/CELL_data.npy')

print('CELL')
print(f'shape (w/o target): {CELL_data.shape}')
print()

#### COD

COD = pd.DataFrame()

colspecs = [(i, i+1) for i in range(17,32)]
colspecs.append((37,50))

for i in range(1,5):
    _cod = pd.read_fwf(f'coding.{i}.data', colspecs=colspecs, header=None)
    COD = pd.concat([COD, _cod], axis=0)
    
COD = COD.iloc[:, :-1]
    
to_dummy = []
for col in range(COD.shape[1]):
    if COD.iloc[:, col].dtype != int:
        to_dummy.append(col)
        u = COD.iloc[:, col].unique()
        COD.iloc[:, col] = COD.iloc[:, col].replace(u, np.arange(len(u)))
        
obj_cols = COD.iloc[:, to_dummy].astype(object)
shuffled_ohe = pd.get_dummies(obj_cols, prefix=np.arange(16)[to_dummy])
shuffled_ohe = pd.concat([COD.drop(COD.columns[to_dummy], axis=1), shuffled_ohe], axis=1)

print('COD')
print(f'shape (w/o target): {COD.shape}')
print(f'ohe shape (w/o target): {shuffled_ohe.shape}')
print()

### MUSH

headers = ['edibility',
                'cap-shape',
                'cap-surface',
                'cap-color',
                'bruises',
                'odor',
                'gill-attachment',
                'gill-spacing',
                'gill-size',
                'gill-color',
                'stalk-shape',
                'stalk-root',
                'stalk-surface-above-ring',
                'stalk-surface-below-ring',
                'stalk-color-above-ring',
                'stalk-color-below-ring',
                'veil-type',
                'veil-color',
                'ring-number',
                'ring-type',
                'spore-print-color',
                'population',
                'habitat']

MUSH = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',
                  index_col=False, names=headers).iloc[:, 1:]

MUSH.dropna(inplace=True)

to_dummy = []
for col in range(MUSH.shape[1]):
    if MUSH.iloc[:, col].dtype != int:
        to_dummy.append(col)
        u = MUSH.iloc[:, col].unique()
        MUSH.iloc[:, col] = MUSH.iloc[:, col].replace(u, np.arange(len(u)))
        
obj_cols = MUSH.iloc[:, to_dummy].astype(object)
shuffled_ohe = pd.get_dummies(obj_cols, prefix=np.array(headers)[to_dummy])
shuffled_ohe = pd.concat([MUSH.drop(MUSH.columns[to_dummy], axis=1), shuffled_ohe], axis=1)

print('MUSH')
print(f'shape (w/o target): {MUSH.shape}')
print(f'ohe shape (w/o target): {shuffled_ohe.shape}')