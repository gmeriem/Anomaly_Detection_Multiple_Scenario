#code source : Meriem GHALI
import csv, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections, numpy
from time import time
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing

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
    clf = svm.LinearSVC()
    clf.fit(train_features,train_labels)
    tt = time()-t0
    print("Classified in {} seconds".format(round(tt,3)))

    #testing the data
    label_prediction = clf.predict(test_features)

    #Score fonction
    accuracy, detectionRate, far = results(label_prediction, test_labels)
    print('accuracy score:', accuracy)
    print('detectionRate score:', detectionRate)
    print('far score:', far)
    saveScore(accuracy, detectionRate, far, tt, len(train_features), len(test_features))
    return (test_features, label_prediction)

def results(prediction, test):
    testEQualPrediction = prediction == test
    accuracy = np.sum(testEQualPrediction)/len(prediction)
    i = 0
    Tp = 0
    Fp = 0
    from collections import Counter
    count = Counter(test)
    #compute True Positive && False Positive
    while i < len(prediction):
        if prediction[i] == 1:
            if test[i] == 1: #if real attack true positive
                Tp +=1
            else :
                Fp +=1 #if fake attack false positive
        i += 1
    #count[1] = number of  attack, count[0] number of normal
    detectionRate = Tp/count[1]
    far = Fp/count[0]
    return (accuracy, detectionRate, far)

def saveScore(accuracy, detectionRate, far, tt, size_train, size_test):
    os.chdir('../../result/text')
    filename  = "Classification_SVM.txt"
    with open(filename, 'w') as f:
        print('Results: \n', file=f)
        print('accuracy score:'+str(accuracy*100)+ '%\n', file=f)
        print('detectionRate score:'+str(detectionRate*100)+ '%\n', file=f)
        print('far score:'+str(far*100)+ '%\n', file=f)
        print('time: '+str(tt)+' seconds \n', file=f)
        print('size of tested data: '+str(size_test) ,file=f)
        print('size of trained data: '+str(size_train), file=f)

def plot(features, labels):
    notAttack = labels!=1
    attack = labels==1
    plt.scatter(features[notAttack, 0], features[notAttack, 1],c=labels[notAttack],cmap="Set3", label='normal')
    plt.scatter(features[attack, 0], features[attack, 1],c=labels[attack],cmap="Pastel1", label='attack')
    plt.xlabel('Destination Port')
    plt.ylabel('Flow Duration')
    plt.legend()
    os.chdir('../figure')
    finalfig = "Classification_SVM.png"
    plt.savefig(finalfig)
    plt.show()

if __name__ == '__main__':
    (features, labels) = process()
    plot(features, labels)
