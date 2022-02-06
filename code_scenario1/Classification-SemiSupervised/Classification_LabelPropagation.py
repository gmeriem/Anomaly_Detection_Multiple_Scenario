#code source : Meriem GHALI
import csv, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections, numpy
from collections import Counter
from time import time
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

attack = {'BENIGN': 0,'PortScan': 1}
prTrain = 5
prTest = 0.10
def import_testing_data():
    with open('Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv') as csvfile:
        data = pd.read_csv(csvfile, header= 0, sep=',')
        test_features, test_labels  = split(data)
        test_features, test_labels = convertAndClean(test_features,test_labels,'test')
        return (test_features,test_labels)

def import_training_data():
    os.chdir('../../data/MachineLearning')
    with open('Monday-WorkingHours.pcap_ISCX.csv') as csvfile:
        data = pd.read_csv(csvfile, header= 0, sep=',')
        train_features, train_labels  = split_divide(data)
        convertAndClean(train_features,train_labels,'train')
        train_features, train_labels = convertAndClean(train_features,train_labels,'train')
        return (train_features, train_labels)

#monday contains a lot of data -> lack of memory
#we are going to use N% of the data
def split_divide(data):
    data = np.random.permutation(data)
    #print('number of rows -> #{}'.format(len(features)))
    features = data[:,:-1]
    labels = data[:,-1]
    N= len(features)
    num_max = int(N*prTrain/100)
    features = features[:num_max]
    labels = labels[:num_max]
    return (features, labels)

#seperate labels -> lack of memory
def split(data):
    #permute the data, to avoid obtaining just normal evet
    data = np.random.permutation(data)
    features = data[:,:-1].astype(float)
    #print('number of rows -> #{}'.format(len(features)))
    labels = data[:,-1]
    #split cause probleme
    features1, features2, labels1,labels2 = train_test_split(features, labels,
    test_size=prTest, random_state=0)
    return(features2,labels2)

def convertAndClean(features,labels, test):
    rowsNotNull = ~pd.isnull(features).any(axis=1)
    #print('number of rows without null (true)-> #{}'.format(collections.Counter(rowsNotNull)))
    features = np.array(features[rowsNotNull], dtype=float)
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
    train_features, train_labels = import_training_data()
    test_features,test_labels = import_testing_data()
    #training the data
    t0 = time()
    label_prop_model = LabelPropagation()
    label_prop_model.fit(train_features,train_labels)
    tt = time()-t0
    print("Classified in {} seconds".format(round(tt,3)))

    #testing import_testing_data()the data
    label_prediction = label_prop_model.predict(test_features)
    #print(label_prediction)

    #Score fonction
    accuracy, detectionRate, far = results(label_prediction, test_labels)
    print('accuracy score:', accuracy)
    print('detectionRate score:', detectionRate)
    print('far score:', far)
    saveScore(accuracy, detectionRate, far, tt, len(train_features), len(test_features))
    return (test_features, label_prediction)

def results(prediction, test):
    testEQualPrediction = prediction == test
    counter2 = Counter(prediction)
    count = Counter(test)
    #print(counter2, count)
    accuracy = np.sum(testEQualPrediction)/len(prediction)
    i = 0
    Tp = 0
    Fp = 0
    #compute True Positive && False Positive
    while i < len(prediction):
        if prediction[i] == 1:
            if test[i] == 1: #if real attack true positive
                Tp +=1
            else :
                Fp +=1 #if fake attack false positive
        i += 1
    #count[1] = number of  attack, count[0] number of normal
    if count[1] != 0:
        detectionRate = Tp/count[1]
    else:
        detectionRate = 0
        print("can not compute detection rate, there is no attack")
    far = Fp/count[0]
    return (accuracy, detectionRate, far)

def saveScore(accuracy, detectionRate, far, tt, size_train, size_test):
    os.chdir('../../result/text')
    filename  = "Classification_LabelPropagation.txt"
    with open(filename, 'w') as f:
        print('Results: \n', file=f)
        print('accuracy score:'+str(accuracy*100)+ '%\n', file=f)
        print('detectionRate score:'+str(detectionRate*100)+ '%\n', file=f)
        print('far score:'+str(far*100)+ '%\n', file=f)
        print('time: '+str(tt)+' seconds \n', file=f)
        print('size of tested data: '+str(size_test) ,file=f)
        print('size of trained data: '+str(size_train), file=f)

def plot(features, labels):
    notAttack = labels==0
    attack = labels != 0
    plt.scatter(features[notAttack, 0], features[notAttack, 1],c=labels[notAttack],cmap="Set3", label='normal')
    plt.scatter(features[attack, 0], features[attack, 1],c=labels[attack],cmap="Pastel1", label='attack')
    plt.xlabel('Destination Port')
    plt.ylabel('Flow Duration')
    plt.legend()
    os.chdir('../figure')
    finalfig = "Classification_LabelPropagation.png"
    plt.savefig(finalfig)
    plt.show()

if __name__ == '__main__':
    (features, labels) = process()
    plot(features, labels)
