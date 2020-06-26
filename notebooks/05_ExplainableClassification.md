https://eli5.readthedocs.io/en/latest/_modules/eli5/sklearn/explain_weights.html
https://eli5.readthedocs.io/en/latest/tutorials/xgboost-titanic.html#explaining-predictions

Prøv at tage, f. eks., Vinterraps mod alle andre (dvs. binær klassifikation), og se hvilke datoer der gør at vinterraps er let at få øje på.

Prøv også at kigge på https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html.

```python
import os
import random
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

from pathlib import Path
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split         # Split data into train and test set

from utils import evaluate_classifier, get_sklearn_df 

# Allow more rows to be printed to investigate feature importance
pd.set_option('display.max_rows', 300)

# Automatically prints execution time for the individual cells
%load_ext autotime

# Automatically reloads functions defined in external files
%load_ext autoreload
%autoreload 2

# Set xarray to use html as display_style
xr.set_options(display_style="html")

# Tell matplotlib to plot directly in the notebook
%matplotlib inline  

# The path to the project (so absoute file paths can be used throughout the notebook)
PROJ_PATH = Path.cwd().parent

# Mapping dict
mapping_dict_crop_types = {
    'Kartofler, stivelses-': 'Potato',
    'Kartofler, lægge- (egen opformering)': 'Potato',
    'Kartofler, andre': 'Potato',
    'Kartofler, spise-': 'Potato',
    'Kartofler, lægge- (certificerede)': 'Potato',
    'Vårbyg': 'Spring barley',
    'Vinterbyg': 'Winter barley',
    'Vårhvede': 'Spring wheat',
    'Vinterhvede': 'Winter wheat',
    'Vinterrug': 'Winter rye',
    'Vårhavre': 'Spring oat',
    'Silomajs': 'Maize',
    'Vinterraps': 'Rapeseed',
    'Permanent græs, normalt udbytte': 'Permanent grass',
    'Pil': 'Willow',
    'Skovdrift, alm.': 'Forest'
}


# Set seed for random generators
RANDOM_SEED = 42

# Seed the random generators
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
```

```python
netcdf_path = (PROJ_PATH / 'data' / 'processed' / 'FieldPolygons2019_stats').with_suffix('.nc') 
ds = xr.open_dataset(netcdf_path, engine="h5netcdf")
ds  # Remember to close the dataset before the netcdf file can be rewritten in cells above
```

```python
ds.close()
```

```python
# Convert the xarray dataset to pandas dataframe
df = ds.to_dataframe()
df = df.reset_index()  # Removes MultiIndex
df = df.drop(columns=['cvr', 'gb', 'gbanmeldt', 'journalnr', 'marknr', 'pass_mode', 'relative_orbit'])
df = df.dropna()
```

```python
df_sklearn = get_sklearn_df(polygons_year=2019, 
                            satellite_dates=slice('2018-01-01', '2019-12-31'), 
                            fields='all', 
                            satellite='S1B', 
                            polarization='all',
                            crop_type='all',
                            netcdf_path=netcdf_path)
    
df_sklearn_remapped = df_sklearn.copy()
df_sklearn_remapped.insert(3, 'Crop type', '')
df_sklearn_remapped.insert(4, 'Label ID', 0)
mapping_dict = {}
class_names = [] 
i = 0
for key, value in mapping_dict_crop_types.items():
    df_sklearn_remapped.loc[df_sklearn_remapped['afgroede'] == key, 'Crop type'] = value 
    if value not in class_names:
        class_names.append(value)
        mapping_dict[value] = i
        i += 1

for key, value in mapping_dict.items():
    df_sklearn_remapped.loc[df_sklearn_remapped['Crop type'] == key, 'Label ID'] = value 
#print(f"Crop types: {class_names}")

# Get values as numpy array
array = df_sklearn_remapped.values

# Define the independent variables as features.
X = np.float32(array[:,5:])  # The features 

# Define the target (dependent) variable as labels.
y = np.int8(array[:,4])  # The column 'afgkode'

# Create a train/test split using 30% test size.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

# Instantiate and evaluate classifier
from sklearn.linear_model import LogisticRegression          
clf = LogisticRegression(solver='lbfgs', multi_class='auto', n_jobs=32, max_iter=1000)
clf_trained, _, accuracy_test, results_report = evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, 
                                                                    feature_scale=True, plot_conf_matrix=False,
                                                                    print_classification_report=True)
```

```python
# Get classfication report as pandas df
df_results = pd.DataFrame(results_report).transpose()  

# Round the values to 2 decimals
df_results = df_results.round(2).astype({'support': 'int32'})  

# Correct error when creating df (acc. is copied into 'precision')
df_results.loc[df_results.index == 'accuracy', 'precision'] = ''  

# Correct error when creating df (acc. is copied into 'recall')
df_results.loc[df_results.index == 'accuracy', 'recall'] = ''  

# Correct error when creating df (acc. is copied into 'recall')
df_results.loc[df_results.index == 'accuracy', 'recall'] = ''  

# The number of samples ('support') was incorrectly parsed in to dataframe
df_results.loc[df_results.index == 'accuracy', 'support'] = df_results.loc[df_results.index == 'macro avg', 'support'].values

# Print df in latex format (I normally add a /midrule above accuracy manually)
print(df_results.to_latex(index=True))  
```

```python
df_results.loc[df_results.index == 'accuracy', 'precision'] = ''
df_results.loc[df_results.index == 'accuracy', 'recall'] = ''
df_results.loc[df_results.index == 'accuracy', 'support'] = df_results.loc[df_results.index == 'macro avg', 'support'].values
df_results
```

```python
try:
    from eli5 import show_weights
except:
    !conda install -y eli5
```

```python
import eli5
from eli5.sklearn import PermutationImportance

feature_names = df_sklearn.columns[3:]
perm = PermutationImportance(clf, random_state=1).fit(X_test, y_test)
eli5.explain_weights(clf, feature_names=list(feature_names), target_names=class_names)
# Look at https://eli5.readthedocs.io/en/latest/autodocs/formatters.html#eli5.formatters.html.format_as_html

# IMPORTANT: LOOK HERE TO FIND IMPORTANCE FOR INDIVIDUAL CLASSES:
#            https://stackoverflow.com/questions/59245580/eli5-explain-weights-does-not-returns-feature-importance-for-each-class-with-skl
```

```python
df_explanation = eli5.formatters.as_dataframe.explain_weights_df(perm, feature_names=list(feature_names))
df_explanation = df_explanation.sort_values(by=['feature'])
df_explanation['polarization'] = ''
features = df_explanation['feature'].unique()
for feature in features:
    if feature[-5:] == 'VV-VH':
        df_explanation.loc[df_explanation['feature'] == feature, 'polarization'] = 'VV-VH'
        df_explanation = df_explanation.replace(feature, feature[:-6])
    elif feature[-2:] == 'VV':
        df_explanation.loc[df_explanation['feature'] == feature, 'polarization'] = 'VV'
        df_explanation = df_explanation.replace(feature, feature[:-3])
    else:
        df_explanation.loc[df_explanation['feature'] == feature, 'polarization'] = 'VH'
        df_explanation = df_explanation.replace(feature, feature[:-3])
        
# OLD CODE:
#df_explanation_vh = df_explanation.iloc[::3]
#df_explanation_vh['polarization'] = 'VH'
#df_explanation_vh['feature'] = df_explanation_vh['feature'].map(lambda x: str(x)[:-3])
#df_explanation_vv = df_explanation.iloc[1::3]
#df_explanation_vv['polarization'] = 'VV'
#df_explanation_vv['feature'] = df_explanation_vv['feature'].map(lambda x: str(x)[:-3])
#df_explanation_vvvh = df_explanation.iloc[2::3]
#df_explanation_vvvh['polarization'] = 'VV-VH'
#df_explanation_vvvh['feature'] = df_explanation_vvvh['feature'].map(lambda x: str(x)[:-6])
#df_explanation = pd.concat([df_explanation_vh, df_explanation_vv, df_explanation_vvvh])
```

```python
df_explanation.head(3)
```

```python
plt.figure(figsize=(24, 8))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='feature', y='weight', hue='polarization', data=df_explanation, ci='sd')
```

```python
#df_explanation = eli5.formatters.as_dataframe.explain_prediction_df(perm, feature_names=list(feature_names))
```

```python
# Show the calculated stds in the df as confidence interval on the plot
# https://stackoverflow.com/questions/58399030/make-a-seaborn-lineplot-with-standard-deviation-confidence-interval-specified-f
#lower_bound = [M_new_vec[i] - Sigma_new_vec[i] for i in range(len(M_new_vec))]
#upper_bound = [M_new_vec[i] + Sigma_new_vec[i] for i in range(len(M_new_vec))]
#plt.fill_between(x_axis, lower_bound, upper_bound, alpha=.3)
```

```python
df_explanation = eli5.formatters.as_dataframe.explain_weights_df(clf, feature_names=list(feature_names), target_names=class_names)
df_explanation = df_explanation.sort_values(by=['feature', 'target'])
df_explanation['polarization'] = ''
features = df_explanation['feature'].unique()
features = features[:-1]  # The last features are the bias values
df_bias_values = df_explanation[df_explanation['feature'] == '<BIAS>']

df_explanation = df_explanation[df_explanation['feature'] != '<BIAS>']
for feature in features:
    if feature[-5:] == 'VV-VH':
        df_explanation.loc[df_explanation['feature'] == feature, 'polarization'] = 'VV-VH'
        df_explanation = df_explanation.replace(feature, feature[:-6])
    elif feature[-2:] == 'VV':
        df_explanation.loc[df_explanation['feature'] == feature, 'polarization'] = 'VV'
        df_explanation = df_explanation.replace(feature, feature[:-3])
    else:
        df_explanation.loc[df_explanation['feature'] == feature, 'polarization'] = 'VH'
        df_explanation = df_explanation.replace(feature, feature[:-3])
```

```python
df_bias_values
```

```python
data = df_explanation[df_explanation['polarization'] == 'VH']
#data = data.loc[data['target'].isin(['Forest', 'Maize', 'Rapeseed'])]
plt.figure(figsize=(24, 8))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='feature', y='weight', hue='target', data=data, ci='sd')
```

```python
data = df_explanation[df_explanation['polarization'] == 'VH']
data = data.loc[data['target'].isin(['Barley', 'Wheat', 'Forest', 'Potato'])]
plt.figure(figsize=(24, 8))
plt.xticks(rotation=90, horizontalalignment='center')
ax = sns.lineplot(x='feature', y='weight', hue='target', data=data, ci='sd')
```

```python

```
