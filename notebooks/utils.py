import itertools
import multiprocessing

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
import pandas as pd

sns.set_style('ticks')
from time import time
from pycm import ConfusionMatrix
from pathlib import Path
from matplotlib import cm  # For waterfall plot
from matplotlib.ticker import LinearLocator, FormatStrFormatter  # For waterfall plot
from mpl_toolkits.mplot3d import Axes3D
from tqdm.autonotebook import tqdm
from rasterstats import zonal_stats, gen_zonal_stats
from sklearn.preprocessing import StandardScaler             # Feature scaling
from sklearn.metrics import classification_report            # Summary of classifier performance
from sklearn.metrics import confusion_matrix                 # Confusion matrix
from sklearn.metrics import accuracy_score


class RasterstatsMultiProc(object):
    """
    Inspired by https://github.com/perrygeo/python-rasterstats/blob/master/examples/multiproc.py
    """
    def __init__(self, tif=None, band=1, df=None, shp=None, stats=['min', 'max', 'median', 'mean', 'std'], all_touched=False):
            self.all_touched = all_touched
            self.band = band
            self.df = df
            self.shp = shp
            self.stats = stats
            self.tif = tif
            
    def calc_zonal_stats(self, prog_bar=True):
        gen = gen_zonal_stats(self.df, self.tif, band=self.band, all_touched=self.all_touched, stats=self.stats, geojson_out=True)
        length = self.df.shape[0]
        results = []
        
        if prog_bar:
            for result in tqdm(gen, total=length):
                results.append(result)
        else:
            for result in gen:
                results.append(result)
            
        results_df = geopandas.GeoDataFrame.from_features(results)
        results_df.crs = self.df.crs
        
        # Move the 'geometry' column to be the last column (https://stackoverflow.com/a/56479671/12045808)
        results_df = results_df[ [ col for col in results_df.columns if col != 'geometry' ] + ['geometry'] ]
        
        return results_df

    def calc_zonal_stats_multiproc(self, features, crs):
        # Create a process pool using all cores
        cores = 8# multiprocessing.cpu_count()
        with multiprocessing.Pool(cores) as pool:
            # parallel map
            results_lists = pool.map(self.zonal_stats_partial, self.chunks(features, cores))

        # flatten to a single list
        results = list(itertools.chain(*results_lists))
        assert len(results) == len(features)
        
        # Create geodataframe from the results
        results_df = geopandas.GeoDataFrame.from_features(results, crs=crs)
        
        # Move the 'geometry' column to be the last column (https://stackoverflow.com/a/56479671/12045808)
        results_df = results_df[ [ col for col in results_df.columns if col != 'geometry' ] + ['geometry'] ]
        
        return results_df 
    
    @staticmethod
    def chunks(data, n):
        """Yield successive n-sized chunks from a slice-able iterable."""
        for i in range(0, len(data), n):
            yield data[i:i+n]

    def zonal_stats_partial(self, feats):
        """Wrapper for zonal stats, takes a list of features"""
        return zonal_stats(feats, self.tif, self.band, all_touched=self.all_touched, stats=self.stats, geojson_out=True)

def get_df(polygons_year=2019, 
           satellite_dates=slice('2019-01-01', '2019-12-31'), 
           fields='all', 
           satellite='all', 
           polarization='all',
           crop_type='all',
           netcdf_path=None):
    # TODO: Perhaps it would be an idea to have field centroids (as lat, lon) to find fields within geographic area
    # Load the xarray dataset
    #netcdf_name = 'FieldPolygons{}_stats'.format(polygons_year)
    #netcdf_path = (PROJ_PATH / 'data' / 'processed' / netcdf_name).with_suffix('.nc')
    with xr.open_dataset(netcdf_path, engine="h5netcdf") as ds:
        # Select dates, fields, and polarizations
        ds = ds.sel(date=satellite_dates)
        if not fields == 'all':  # Must be 'all' or array of integers (eg. [1, 2, 3, 4]) of field_ids
            ds = ds.isel(field_id=fields) 
        if not polarization == 'all':  # Must be 'all', 'VV', 'VH', or 'VV-VH'
            ds = ds.sel(polarization=polarization) 

        # Convert ds to dataframe
        df = ds.to_dataframe()
        df = df.reset_index()  # Removes MultiIndex
        df = df.drop(columns=['cvr', 'gb', 'gbanmeldt', 'journalnr', 'marknr'])

        # Select satellites
        if not satellite == 'all':  # Must be 'all', 'S1A', or 'S1B'
            df = df[df['satellite']==satellite]
            
        # Select crop types
        if not crop_type == 'all':  # Must be 'all' or name of crop type
            df = df[df['afgroede']==crop_type]
        
    return df

def get_plot_df(polygons_year=2019, 
                satellite_dates=slice('2019-01-01', '2019-12-31'), 
                fields='all', 
                satellite='all', 
                polarization='all',
                crop_type='all',
                netcdf_path=None):
    
    df = get_df(polygons_year=polygons_year, 
                satellite_dates=satellite_dates, 
                fields=fields, 
                satellite=satellite, 
                polarization=polarization,
                crop_type=crop_type,
                netcdf_path=netcdf_path)
    
    # Format the dataframe to work well with Seaborn for plotting
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df['field_id'] = ['$%s$' % x for x in df['field_id']]  # https://github.com/mwaskom/seaborn/issues/1653
    df['afgkode'] = ['$%s$' % x for x in df['afgkode']]  # https://github.com/mwaskom/seaborn/issues/1653
    df['relative_orbit'] = ['$%s$' % x for x in df['relative_orbit']]  # https://github.com/mwaskom/seaborn/issues/1653
        
    return df


def get_sklearn_df(polygons_year=2019, 
                   satellite_dates=slice('2019-01-01', '2019-12-31'), 
                   fields='all', 
                   satellite='all', 
                   polarization='all',
                   crop_type='all',
                   netcdf_path=None):
    
    if polarization == 'all':
        polarizations = ['VV', 'VH', 'VV-VH']

    # Create the df format to be used by scikit-learn
    for i, polarization_val in enumerate(polarizations):
        df_polarization = get_df(polygons_year=polygons_year, 
                                 satellite_dates=satellite_dates, 
                                 fields=fields, 
                                 satellite=satellite, 
                                 polarization=polarization_val,
                                 netcdf_path=netcdf_path)

        # Extract a mapping of field_ids to crop type
        if i == 0:
            df_sklearn = df_polarization[['field_id', 'afgkode', 'afgroede']]

        # Pivot the df (https://stackoverflow.com/a/37790707/12045808)
        df_polarization = df_polarization.pivot(index='field_id', columns='date', values='stats_mean')

        # Add polarization to column names
        df_polarization.columns = [str(col)[:10]+f'_{polarization_val}' for col in df_polarization.columns]  

        # Merge the polarization dataframes into one dataframe
        df_polarization = df_polarization.reset_index()  # Creates new indices and a 'field_id' column (field id was used as indices before)
        df_sklearn = pd.merge(df_sklearn, df_polarization, on='field_id') 

    # Drop fields having nan values
    df_sklearn = df_sklearn.dropna()

    # The merge operation for some reason made duplicates (there was a bug reported on this earlier), so drop duplicates and re-index the df
    df_sklearn = df_sklearn.drop_duplicates().reset_index(drop=True)
    
    return df_sklearn
    
    
def plot_waterfall_all_polarizations(crop_type = 'Vinterraps', satellite_dates=slice('2019-01-01', '2019-12-31'), num_fields=128, satellite='all', sort_rows=True, netcdf_path=None):
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'})
    #fig.suptitle(f"Temporal evolution of {crop_type}", fontsize=16)
    fig.set_figheight(8)
    fig.set_figwidth(24) 
    
    polarizations = ['VV', 'VH', 'VV-VH']
    for i, polarization in enumerate(polarizations):
        df = get_plot_df(polygons_year=2019, 
                         satellite_dates=satellite_dates, 
                         fields='all', 
                         satellite=satellite, 
                         polarization=polarization,
                         crop_type=crop_type,
                         netcdf_path=netcdf_path)

        df = df.dropna()
        
        # Get the dates (needed later for plotting)
        dates = df['date'].unique()
        num_dates = len(dates)

        # Pivot the df (https://stackoverflow.com/a/37790707/12045808)
        df = df.pivot(index='field_id', columns='date', values='stats_mean')
        
        # Drop fields having any date with a nan value, and pick num_fields from the remainder
        df = df.dropna().sample(n=num_fields, random_state=1)
        
        if sort_rows:
            # Sort by sum of each row
            df = df.reset_index()
            df = df.drop(columns=['field_id'])
            idx = df.sum(axis=1).sort_values(ascending=False).index
            df = df.iloc[idx]

            # Sort by specific column
            #col = 0
            #df = df.sort_values(by=df.columns.tolist()[col], ascending=False) 

        # Get the min and max values depending on polarization
        if polarization == 'VV':
            vmin, vmax = -20, 5
        elif polarization == 'VH':
            vmin, vmax = -30, -5
        elif polarization == 'VV-VH':
            vmin, vmax = 0, 25
        else:
            raise ValueError("Polarization not supporten (must be VV, VH, or VV-VH)")
            
        # Make data.
        x = np.linspace(1, num_dates, num_dates)  # Dates
        y = np.linspace(1, num_fields, num_fields)  # Fields
        X,Y = np.meshgrid(x, y)
        Z = df.to_numpy()

        # Plot the surface.
        surf = axs[i].plot_surface(X, Y, Z, cmap=cm.coolwarm, vmin=vmin, vmax=vmax,
                               linewidth=0, antialiased=False)
        
        # Set title 
        axs[i].set_title(f"Temporal evolution of: {crop_type}, {polarization}")
        
        # Set angle (https://stackoverflow.com/a/47610615/12045808)
        axs[i].view_init(25, 280)
        
        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, ax=axs[i], shrink=0.5, aspect=10)
        
        # Customize the z axis (backscattering value)
        axs[i].zaxis.set_major_locator(LinearLocator(6))
        axs[i].zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        axs[i].set_zlim(vmin, vmax)
        for tick in axs[i].zaxis.get_major_ticks():
            tick.label1.set_horizontalalignment('left')
        
        # Customize the x axis (dates)
        ticks_divider = int(np.ceil(num_dates/10))  # If more than 10 dates, skip every second tick, if more than 20 dates, skip every third ...
        xticks = range(1, num_dates+1)[::ticks_divider]  # Array must be starting at 1
        xticklabels = dates[::ticks_divider]
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels(xticklabels, rotation=75, horizontalalignment='right')
        
        # Customize the y axis (field ids)
        for tick in axs[i].yaxis.get_major_ticks():
            tick.label1.set_horizontalalignment('left')
            tick.label1.set_verticalalignment('bottom')
            tick.label1.set_rotation(-5)

        # Set viewing distance (important to not cut off labels)
        axs[i].dist = 11   

    fig.tight_layout()
    fig.show()

    
def plot_heatmap_all_polarizations(crop_type = 'Vinterraps', satellite_dates=slice('2019-01-01', '2019-12-31'), num_fields=128, satellite='all', sort_rows=False, netcdf_path=None):
    fig, axs = plt.subplots(1, 3)
    fig.suptitle(f"Temporal evolution of: {crop_type}", fontsize=16)
    fig.set_figheight(8)
    fig.set_figwidth(24) 
    
    polarizations = ['VV', 'VH', 'VV-VH']
    for i, polarization in enumerate(polarizations):
        df = get_plot_df(polygons_year=2019, 
                         satellite_dates=satellite_dates, 
                         fields='all', 
                         satellite=satellite, 
                         polarization=polarization,
                         crop_type=crop_type,
                         netcdf_path=netcdf_path)

        # Pivot the df (https://stackoverflow.com/a/37790707/12045808)
        df = df.pivot(index='field_id', columns='date', values='stats_mean')
        
        # Drop fields having any date with a nan value, and pick num_fields from the remainder
        df = df.dropna()
        if num_fields > df.shape[0]:
            num_fields = df.shape[0]
            print(f"Only {num_fields} fields were available for plotting")
        df = df.sample(n=num_fields, random_state=1)

        if sort_rows:
            # Sort by sum of each row
            df = df.reset_index()
            df = df.drop(columns=['field_id'])
            idx = df.sum(axis=1).sort_values(ascending=False).index
            df = df.iloc[idx]

            # Sort by specific column
            #col = 0
            #df = df.sort_values(by=df.columns.tolist()[col], ascending=False) 

        # Get the min and max values depending on polarization
        if polarization == 'VV':
            vmin, vmax = -15, 0
        elif polarization == 'VH':
            vmin, vmax = -25, -10 
        elif polarization == 'VV-VH':
            vmin, vmax = 5, 20
        else:
            raise ValueError("Polarization not supporten (must be VV, VH, or VV-VH)")

        sns.heatmap(df, ax=axs[i], linewidths=0, linecolor=None, vmin=vmin, vmax=vmax, yticklabels=False, cmap=cm.coolwarm, cbar_kws={'label': "{}, stats_mean".format(polarization)})

    fig.show()

def plot_confusion_matrix(cm, classes):
    # Modified form of https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Create figure
    if len(classes) < 5:
        plt.figure(figsize=(7, 7))
    else: 
        plt.figure(figsize=(20, 20))

    # Plot non-normalized confusion matrix
    plt.subplot(121)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout(w_pad=8)
    
    # Plot normalized confusion matrix
    plt.subplot(122)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout(w_pad=8)
        
def save_confusion_matrix_fig(cm, classes, save_path=None, normalized=False, fontsize=11):
    # Modified form of https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Create figure
    if len(classes) < 5:
        plt.figure(figsize=(7, 7))
    else: 
        plt.figure(figsize=(8, 8))

    # Plot non-normalized confusion matrix
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.title('Confusion matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=fontsize)
    plt.yticks(tick_marks, classes, fontsize=fontsize)

    fmt = '.2f' if normalized else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=fontsize)
    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.tight_layout(w_pad=8)
    if save_path:
        if normalized:
            save_path = Path(f'{str(save_path)}_normalized')
        plt.savefig(save_path.with_suffix('.pdf'))
    
def evaluate_classifier(clf, X_train, X_test, y_train, y_test, class_names, feature_scale=False, plot_conf_matrix=True, auto_sklearn_crossvalidation=False, print_classification_report=True):
    """
    This function evaluates a classifier. It measures training and prediction time, and 
    prints performance metrics and a confustion matrix. The returned classifier and 
    scaler are fitted to the training data, and can be used to predict new samples.
    """
    
    # Perform feature scaling
    scaler = StandardScaler()  # Scale to mean = 0 and std_dev = 1
    if feature_scale:
        # ====================== YOUR CODE HERE =======================

        X_train = scaler.fit_transform(X_train)  # Fit to training data and then scale training data
        X_test = scaler.transform(X_test)  # Scale test data based on the scaler fitted to the training data

        # =============================================================
        
    # Store the time so we can calculate training time later
    t0 = time()

    # Fit the classifier on the training features and labels
    if not auto_sklearn_crossvalidation:
        clf.fit(X_train, y_train)
    else:
        # From https://automl.github.io/auto-sklearn/master/examples/example_crossvalidation.html
        # fit() changes the data in place, but refit needs the original data. We
        # therefore copy the data. In practice, one should reload the data
        clf.fit(X_train.copy(), y_train.copy())
        # During fit(), models are fit on individual cross-validation folds. To use
        # all available data, we call refit() which trains all models in the
        # final ensemble on the whole dataset.
        clf.refit(X_train.copy(), y_train.copy()) 
    
    # Calculate and print training time
    print("Training time:", round(time()-t0, 4), "s")

    # Store the time so we can calculate prediction time later
    t1 = time()
    
    # Use the trained classifier to classify the test data
    predictions = clf.predict(X_test)
    
    # Calculate and print prediction time
    print("Prediction time:", round(time()-t1, 4), "s")

    # Evaluate the model
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    results_report_dict = classification_report(y_test, predictions, target_names=class_names, output_dict=True)

    # Print the reports
    print("\nReport:\n")
    print("Train accuracy: {}".format(round(train_accuracy, 4)))
    print("Test accuracy: {}".format(round(test_accuracy, 4)))
    if print_classification_report:
        results_report = classification_report(y_test, predictions, target_names=class_names)
        print("\n", results_report)
    
    # Plot confusion matrices
    cnf_matrix = confusion_matrix(y_test, predictions)
    if plot_conf_matrix:
        plot_confusion_matrix(cnf_matrix, classes=class_names)
    
    # Return the trained classifier to be used on future predictions
    return clf, scaler, test_accuracy, results_report_dict, cnf_matrix


def plot_svm_hparams_search():
    pass
    """
    # Idea: Maybe make a utils folder, with a plotting module, evaluation module etc.. The below here should 
    #       then be put in the plotting module. 
    mean_test_scores = grid_trained.cv_results_['mean_test_score']
    mean_fit_times = grid_trained.cv_results_['mean_fit_time']
    param_columns = list(grid_trained.cv_results_['params'][0].keys())
    result_columns = ['mean_fit_time', 'mean_test_score']
    num_fits = len(grid_trained.cv_results_['params'])

    df_cv_results = pd.DataFrame(0, index=range(num_fits), columns=param_columns+result_columns)
    for i, param_set in enumerate(grid_trained.cv_results_['params']):
        for param, value in param_set.items():
            df_cv_results.loc[i, param] = value 
        df_cv_results.loc[i, 'mean_test_score'] = mean_test_scores[i]
        df_cv_results.loc[i, 'mean_fit_time'] = mean_fit_times[i]

    df_heatmap_mean_score = df_cv_results.pivot(index='C', columns='gamma', values='mean_test_score')
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(df_heatmap_mean_score, annot=True, cmap=plt.cm.Blues)

    df_heatmap_fit_time = df_cv_results.pivot(index='C', columns='gamma', values='mean_fit_time')
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(df_heatmap_fit_time.astype('int64'), annot=True, fmt='d', cmap=plt.cm.Blues_r)
    """
    
    
def numpy_confusion_matrix_to_pycm(confusion_matrix_numpy, labels=None):
    """Create a pycm confusion matrix from a NumPy confusion matrix
    Creates a confusion matrix object with the pycm library based on a confusion matrix as 2D NumPy array (such as
    the one generated by the sklearn confusion matrix function).
    See more about pycm confusion matrices at `pycm`_, and see more
    about sklearn confusion matrices at `sklearn confusion matrix`_.
    Args:
        confusion_matrix_numpy (np.array((num_classes, num_classes)) :
        labels (list) :
    Returns:
        confusion_matrix_pycm (pycm.ConfusionMatrix) :
    .. _`pycm`: https://github.com/sepandhaghighi/pycm
    .. _`sklearn confusion matrix`:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # Create empty dict to be used as input for pycm (see https://github.com/sepandhaghighi/pycm#direct-cm)
    confusion_matrix_dict = {}

    # Find number and classes and check labels
    num_classes = np.shape(confusion_matrix_numpy)[0]
    if not labels:  # If no labels are provided just use [0, 1, ..., num_classes]
        labels = range(num_classes)
    elif len(labels) != num_classes:
        raise AttributeError("Number of provided labels does not match number of classes.")

    # Fill the dict in the format required by pycm with values from the sklearn confusion matrix
    for row in range(num_classes):
        row_dict = {}
        for col in range(num_classes):
            row_dict[str(labels[col])] = int(confusion_matrix_numpy[row, col])
        confusion_matrix_dict[str(labels[row])] = row_dict

    # Instantiate the pycm confusion matrix from the dict
    confusion_matrix_pycm = ConfusionMatrix(matrix=confusion_matrix_dict)

    return confusion_matrix_pycm