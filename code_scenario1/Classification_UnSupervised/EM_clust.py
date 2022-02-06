#Code source: Crystalor SAH
import csv, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections, numpy
from time import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report,\
    hamming_loss,  matthews_corrcoef, zero_one_loss

attack = {'BENIGN': 0,'PortScan': 1}

def import_data():
    os.chdir('../../data/MachineLearning')
    with open('Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv') as csvfile:
        data = pd.read_csv(csvfile, header= 0, sep=',')
        train_features, test_features , train_labels, test_labels  = split(data)
        train_features, train_labels = convertAndClean(train_features, train_labels)
        test_features, test_labels = convertAndClean(test_features,test_labels)
        return (train_features, train_labels, test_features,test_labels)

#50% train , 50%test
def split(data):
    #permute the data, to avoid obtaining just normal evet
    data = np.random.permutation(data)
    features = data[:,:-1].astype(float)
    #print('number of rows -> #{}'.format(len(features)))
    labels = data[:,-1]
    train_features, test_features, train_labels,test_labels = train_test_split(features, labels,
    test_size=0.5, random_state=0)
    return(train_features, test_features, train_labels,test_labels)

def convertAndClean(features,labels):
    rowsNotNull = ~np.isnan(features).any(axis=1)
    #print('number of rows without null (true)-> #{}'.format(collections.Counter(rowsNotNull)))
    features = features[rowsNotNull]
    #print('There are #{} rows without null variable'.format(len(features)))
    labels = labels[rowsNotNull]
    rows_finite = np.isfinite(features).all(axis=1)
    features = features[rows_finite]
    labels = labels[rows_finite]
    #convert labels to numbers- supervised_learning does not support string values
    checkData(features)
    labels_numbers= [attack[labels_str] for labels_str  in labels]
    return(features,labels_numbers)

def checkData(data):
    nan_indices = 0
    inf_indices = 0
    nan_indices = sum(np.isnan(data).all(axis=1))
    inf_indices = sum(np.isinf(data).all(axis=1))

    if nan_indices > 0 or inf_indices > 0:
        raise Exception("Sorry, there is an infinitive or nan value in data")

def process():
    train_features, train_labels, test_features,test_labels = import_data()
    #training the data
    t0 = time()
    clf = GaussianMixture(n_components=2, reg_covar=1)
    clf.fit(train_features)
    tt = time()-t0
    print("Classified in {} seconds".format(round(tt,3)))
    label_prediction = clf.predict(test_features)
    #testing the data
    print(label_prediction)

    #Score computing
    TP, FP, TN, FN = perf_measure(test_labels, label_prediction)
    DR = dr(TP, FP)
    FAR = far(FP, TN)
    accuracy = accuracy_score(test_labels, label_prediction)
    confusion = confusion_matrix(test_labels, label_prediction)
    classification_r = classification_report(test_labels, label_prediction)
    hamming = hamming_loss(test_labels, label_prediction)
    matthews = matthews_corrcoef(test_labels, label_prediction)
    zero_one = zero_one_loss(test_labels, label_prediction)
    saveScore(DR, FAR, TP, FP, TN, FN, accuracy, confusion, classification_r, hamming, matthews, zero_one,
              tt, len(train_features), len(test_features))
    return (test_features, label_prediction)

def perf_measure(y_true, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]==1:
           TP += 1
        if y_true[i]==1 and y_true[i]!=y_pred[i]:
           FP += 1
        if y_true[i]==y_true[i]==0:
           TN += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
           FN += 1

    return(TP, FP, TN, FN)

def dr(TP, FP):
    return (TP/(FP+TP))*100

def far(FP, TN):
    return (FP/(FP+TN))*100

def saveScore(DR, FAR, TP, FP, TN, FN, accuracy, confusion, classification_r, hamming,matthews, zero_one, tt, size_train, size_test):
    os.chdir('../../result/text')
    filename  = "EM_Clust.txt"
    with open(filename, 'w') as f:
        print('Results: \n', file=f)
        print('time: '+str(tt)+' seconds \n', file=f)
        print('size of tested data: '+str(size_test) ,file=f)
        print('size of trained data: '+str(size_train)+'\n', file=f)
        print('True Positive: ' + str(TP), file=f)
        print('False Positive: ' + str(FP), file=f)
        print('True Negative: ' + str(TN), file=f)
        print('False Negative: ' + str(FN), file=f)
        print('Detection Rate: ' + str(DR) + ' %', file=f)
        print('False Alarm Rate: ' + str(FAR) + ' %\n', file=f)
        print("\n\nAccuracy {} %".format(round(accuracy * 100, 3)), file=f)
        print("\n\nConfusion Matrix: \n\n {}".format(confusion), file=f)
        print("\n\nClassification Scores: \n\n {}".format(classification_r), file=f)
        print("\n\nHamming Loss {}".format(hamming), file=f)
        print("\n\nMatthews corrcoef {}".format(matthews), file=f)
        print("\n\nZero-One Loss {}".format(zero_one), file=f)


def plot(features, labels):
    notAttack = labels!=1
    attack = labels==1
    plt.scatter(features[notAttack, 0], features[notAttack, 1],c=labels[notAttack],cmap="tab10", label='normal')
    plt.scatter(features[attack, 0], features[attack, 1],c=labels[attack],cmap="Pastel1", label='attack')
    plt.xlabel('Destination Port')
    plt.ylabel('Flow Duration')
    plt.legend()
    plt.title("EM_Clust")
    os.chdir('../figure')
    finalfig = "EM_Clust.png"
    plt.savefig(finalfig)
    plt.show()

if __name__ == '__main__':
    (features, labels) = process()
    plot(features, labels)