import time
from random import randint

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

classifiers = {
    'Voting Classifier': VotingClassifier(
        estimators=[('first', GradientBoostingClassifier(n_estimators=1000)),
                    ('second', KNeighborsClassifier(3)),
                    ('third', AdaBoostClassifier())],
        voting='soft'),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Nearest Neighbors': KNeighborsClassifier(3),
    'Perceptron': Perceptron(),
    'Multi Layer Perceptron': MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), activation="relu", random_state=1),
    'Linear SVM': SVC(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=500),
    'CatBoost': CatBoostClassifier(verbose=False),
    'Naive Bayes': GaussianNB(),
    'Ada Boost': AdaBoostClassifier(),
    'Gaussian Process': GaussianProcessClassifier(),
}


def trainclassifiers(data, split, verbose, balance_data=True, normalization=True):
    if (normalization):
        data = normalize(data)
    X = data
    Y = X.pop('class')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=split, random_state=0)

    if balance_data:
        smote = SMOTE('not majority', random_state=1)
        x_train, y_train = smote.fit_resample(x_train, y_train)
        x_test, y_test = smote.fit_resample(x_test, y_test)

    models = {}
    for classifier_name, classifier in list(classifiers.items()):
        t_start = time.process_time()
        classifier.fit(x_train, y_train)
        t_end = time.process_time()
        train_time = t_end - t_start

        y_pred = classifier.predict(x_test)
        cmatrix = confusion_matrix(y_test, y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        dict_metrics = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        models[classifier_name] = {
            'model': classifier,
            'metrics': dict_metrics,
            'train_time': train_time,
        }
        if verbose:
            print(f"###################  {classifier_name}  ##################")
            print(f'Trained in : {train_time} ')
            print(cmatrix)
            print(classification_report(y_test, y_pred))
    return models


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def select_features(data,featureclusters,criteria='random'):

    corr=data.corr()
    features_selected=[]

    for i in range(0,featureclusters['Id'].max()):

        cluster=featureclusters.loc[featureclusters["Id"]==i+1,"Name"]

        if criteria=='random' :
            features_selected.append(cluster.iloc[randint(0,cluster.shape[0]-1)])

        if criteria=='class' :
            featuretoadd=cluster.iloc[0]
            for feature in cluster:
                if corr.loc[feature]['class']>corr.loc[featuretoadd]['class']:
                    featuretoadd=feature
            features_selected.append(featuretoadd)

    if 'class' not in features_selected:
        features_selected.append('class')
    print(features_selected)
    print(features_selected.__len__())
    new_data = data[features_selected]
    new_data.pop('id')
    return new_data

def rankclassifiers(modellist, verbose=True):
    data = {'Classifier name': classifiers.keys(), 'Accuracy': []}
    for name in data['Classifier name']:
        print(name)
        accuracy = modellist[name]['metrics']['accuracy']
        data['Accuracy'].append(accuracy)

    df = pd.DataFrame(data)
    print(df.sort_values(by='Accuracy', ascending=False))
    return df

    # Encoding dei tipi


def trainwithfeatureclusters(clustersfiles, data):

    clss = {'threshold': [], 'classifiers': []}
    for clustersfile in clustersfiles:
        featurecluster = pd.read_csv(clustersfile)
        print('##########################################################################################\n'
              fr'############  Results using file {clustersfile}  ###################'
              '\n##########################################################################################')
        preprocessed_data = select_features(data, featurecluster)
        classifiers = trainclassifiers(preprocessed_data, split=0.25, verbose=True)
        rankclassifiers(classifiers)
        clss['threshold'].append(clustersfile)
        clss['classifiers'].append(classifiers)
    return clss

def main():
    # lettura del dataset e delle feature clusterizzate tramite il dendrogramma
    data = pd.read_csv("../Data/pd_speech_features_firstrowdropped.csv")
    clustersfiles=[]

    clustersfiles.append("../Features clusters/features09.csv")
    clustersfiles.append("../Features clusters/features07.csv")
    clustersfiles.append("../Features clusters/features05.csv")
    clustersfiles.append("../Features clusters/features03.csv")
    clustersfiles.append("../Features clusters/features02.csv")
    clustersfiles.append("../Features clusters/features01.csv")

    trainwithfeatureclusters(clustersfiles,data)


if __name__ == '__main__':
    main()
