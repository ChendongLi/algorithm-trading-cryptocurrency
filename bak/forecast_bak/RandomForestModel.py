import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
)
from imblearn.under_sampling import (
RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
)
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, ADASYN)

from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
from collections import Counter

class RandomForecastTrendForecasting:
    def __init__(self):
        self.random_state = 42
        self.n_jobs = 1

    def rf_model(self, 
                    X_train, 
                    X_test, 
                    y_train, 
                    y_test, 
                    n_estimators, 
                    max_depth, 
                    feature_names 
                    ):
        
        model = RandomForestClassifier(
            n_estimators= n_estimators,
            n_jobs= self.n_jobs,
            random_state= self.random_state,
            class_weight = 'balanced_subsample',
            criterion = 'entropy',  
            max_depth= max_depth
            )

        model.fit(X_train, y_train)

        # print ('Features sorted by their score:')
        # feature_importances = pd.DataFrame(model.feature_importances_,
        #         index = feature_names,  
        #         columns=['importance']).sort_values('importance',ascending=False)                                                    
        # print(tabulate(feature_importances, headers = 'keys', tablefmt="psql"))

        print("Train Outputting Metrics...")
        print("Hit-Rate: %s" % model.score(X_train, y_train))
        print("%s\n" % confusion_matrix(y_train, model.predict(X_train)))

        #model.fit(X, y)
        print("Test Outputting metrics...")
        print("Hit-Rate: %s" % model.score(X_test, y_test))
        print("%s\n" % confusion_matrix( y_test, model.predict(X_test)))

    def rf_undersample(self, 
                X_train, 
                X_test, 
                y_train, 
                y_test, 
                n_estimators, 
                max_depth, 
                ):

        techniques = [RandomOverSampler(), 
                    SMOTE(), 
                    ADASYN(),
                    RandomUnderSampler(),
                    NearMiss(version=1)
                    # NearMiss(version=2),
                    # TomekLinks(),
                    # EditedNearestNeighbours()
                    ]
    
        model = RandomForestClassifier(
        n_estimators= n_estimators,
        n_jobs= self.n_jobs,
        random_state= self.random_state,
        class_weight = 'balanced_subsample',
        criterion = 'entropy',  
        max_depth= max_depth
        )

        results = {
            'undersample': {}}

        for sampler in techniques:
            technique = sampler.__class__.__name__
            if technique == 'NearMiss': technique+=str(sampler.version)
            print(f'Technique: {technique}')
            print(f'Before resampling: {sorted(Counter(y_train).items())}')
            X_resampled, y_resampled = sampler.fit_sample(X_train, y_train)
            print(f'After resampling: {sorted(Counter(y_resampled).items())}')

            model.fit(X_resampled, y_resampled)
            predictions = model.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, predictions)
            precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions)
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
            fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            results['undersample'][technique] = {'accuracy': accuracy, 
                                                'precision': precision, 
                                                'recall': recall,
                                                'fscore': fscore, 
                                                'n_occurences': support,
                                                'predictions_count': Counter(predictions),
                                                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                                                'auc': auc}
            

        return results
