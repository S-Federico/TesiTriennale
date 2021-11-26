import time
from random import randint

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

classifiers = {
    #'Logistic Regression': LogisticRegression(max_iter=1000),
    'Nearest Neighbors': KNeighborsClassifier(),
    #'Perceptron': Perceptron(),
    #'Multi Layer Perceptron': MLPClassifier(),
    #'Linear SVM': SVC(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=1000),
    'Naive Bayes': GaussianNB(),
    'Ada Boost': AdaBoostClassifier(),
    # 'QDA': QuadraticDiscriminantAnalysis(),
    # 'Gaussian Process': GaussianProcessClassifier(),
    #'Dummy Classifier uniform': DummyClassifier(strategy='uniform'),
}


def trainclassifiers(data, split, verbose):
    X = data
    Y = X.pop('class')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=split, random_state=0)
    models = {}
    for classifier_name, classifier in list(classifiers.items()):

        t_start = time.process_time()

        classifier.fit(x_train, y_train)

        t_end = time.process_time()
        train_time = t_end - t_start

        y_pred = classifier.predict(x_test)
        cmatrix = confusion_matrix(y_test, y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        dict_metrics = classification_report(y_test, y_pred, output_dict=True,zero_division=0)

        # dizionario che contiene i modelli di classificatori
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


def select_features(data, featureclusters, criteria='random'):
    corr = data.corr()
    features_selected = []

    for i in range(0, featureclusters['Id'].max()):

        cluster = featureclusters.loc[featureclusters["Id"] == i + 1, "Name"]

        if criteria == 'random':
            features_selected.append(cluster.iloc[randint(0, cluster.shape[0] - 1)])

    return data[features_selected]


def main():
    # lettura del dataset e delle feature clusterizzate tramite il dendrogramma
    data = pd.read_csv("../Data/pd_speech_features_firstrowdropped.csv")
    featureclusters09 = pd.read_csv("../Features clusters/features09.csv")
    featureclusters07 = pd.read_csv("../Features clusters/features07.csv")
    featureclusters05 = pd.read_csv("../Features clusters/features05.csv")
    featureclusters03 = pd.read_csv("../Features clusters/features03.csv")
    featureclusters02 = pd.read_csv("../Features clusters/features02.csv")
    featureclusters01 = pd.read_csv("../Features clusters/features01.csv")
    featureclusters005 = pd.read_csv(r"C:\Users\fonde\PycharmProjects\Tesi triennale\Features clusters\features005.csv")
    featureclusters001 = pd.read_csv(r"C:\Users\fonde\PycharmProjects\Tesi triennale\Features clusters\features001.csv")

    print(
        '##########################\n############  Results using threshold 0.9  ###################\n###########################')
    preprocessed_data_09 = select_features(data, featureclusters09)
    classifiers09 = trainclassifiers(preprocessed_data_09, split=0.2, verbose=True)

    print(
        '##########################\n############  Results using threshold 0.7  ###################\n###########################')
    preprocessed_data_07 = select_features(data, featureclusters07)
    classifiers07 = trainclassifiers(preprocessed_data_07, split=0.2, verbose=True)

    print(
        '##########################\n############  Results using threshold 0.5  ###################\n###########################')
    preprocessed_data_05 = select_features(data, featureclusters05)
    classifiers05 = trainclassifiers(preprocessed_data_05, split=0.2, verbose=True)

    print(
        '##########################\n############  Results using threshold 0.3  ###################\n###########################')
    preprocessed_data_03 = select_features(data, featureclusters03)
    classifiers03 = trainclassifiers(preprocessed_data_03, split=0.2, verbose=True)

    print(
        '##########################\n############  Results using threshold 0.2  ###################\n###########################')
    preprocessed_data_02 = select_features(data, featureclusters02)
    classifiers02 = trainclassifiers(preprocessed_data_02, split=0.2, verbose=True)

    print(
        '##########################\n############  Results using threshold 0.1  ###################\n###########################')
    preprocessed_data_01 = select_features(data, featureclusters01)
    classifiers01 = trainclassifiers(preprocessed_data_01, split=0.2, verbose=True)

    print(
        '##########################\n############  Results using threshold 0.05  ###################\n###########################')
    preprocessed_data_005 = select_features(data, featureclusters005)
    classifiers005 = trainclassifiers(preprocessed_data_005, split=0.2, verbose=True)

    print(
        '##########################\n############  Results using threshold 0.01  ###################\n###########################')
    preprocessed_data_001 = select_features(data, featureclusters001)
    classifiers001 = trainclassifiers(preprocessed_data_001, split=0.2, verbose=True)


if __name__ == '__main__':
    main()
