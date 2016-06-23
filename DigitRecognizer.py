import scipy
import numpy as np
import operator
import csv
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def read_data(f, header=True, test=False, rows=0):
    data = []
    labels = []

    csv_reader = csv.reader(open(f, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index = index + 1
        if rows > 0 & index > rows:
            break
        if header and index == 1:
            continue

        if not test:
            labels.append(int(row[0]))
            row = row[1:]

        data.append(np.array(np.int64(row)))
    return (data, labels)

def predictSVC(train, labels, test):
    print 'starting SVC'
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, labels, test_size=0.3, random_state=0)
    clf = SVC(probability=True)
    print 'Training..'
    clf.fit(X_train, y_train)
    acc_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test, sample_weight=None)
    print("Accuracy of SVC: %f \n", accuracy)
    print 'Testing..'
    svc_predictions = clf.predict(test)
    svc_probs = clf.predict_proba(test)
    svc_bestProbs = svc_probs.max(axis=1)
    print 'done with SVC'
    return svc_predictions, svc_bestProbs

def predictRFC(train, labels, test):
    print 'starting RFC'
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, labels, test_size=0.2, random_state=1)
    clf = RandomForestClassifier(n_estimators=400, criterion='entropy')
    print 'Training..'
    clf.fit(X_train, y_train)
    acc_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test, sample_weight=None)
    print 'Accuracy of RFC: ', accuracy*100
    print 'Testing..'
    rfc_predictions = clf.predict(test)
    rfc_probs = clf.predict_proba(test)
    rfc_bestProbs = rfc_probs.max(axis=1)
    print 'done with RFC'
    return rfc_predictions, rfc_bestProbs

class PredScore:
    def __init__(self,prediction,score):
        self.prediction = prediction
        self.score = score
    prediction = -1
    score = 0

if __name__ == '__main__':
    print 'reading data..'

    train, labels = read_data("train.csv", rows=1000)
    np.save('train_small.npy', train)
    np.save('labels_small.npy', labels)

    train = np.load('train_small.npy')
    labels = np.load('labels_small.npy')

    print 'done reading train'

    test, tmpl = read_data("test.csv", test=True, rows=1000)
    np.save('test_small.npy', test)
    np.save('tmpl_small.npy', tmpl)

    test = np.load('test_small.npy')
    tmpl = np.load('tmpl_small.npy')

    print 'done reading test'

    #svcPredictions, svcScore = predictSVC(train, labels, test)
    rfcPredictions, rfcScore = predictRFC(train, labels, test)

    retArray = []
    index = 0
    #for sf in sfScore:
    for rf in rfcScore:
        #svcPredScore = PredScore(svcPredictions[index],svcScore[index])
        rfcPredScore = PredScore(rfcPredictions[index],rfcScore[index])

        options = []
        #options.append(svcPredScore)
        options.append(rfcPredScore)

        maxObj = max(options,key=operator.attrgetter('score'))
        retArray.append(maxObj.prediction)
        index = index + 1
    np.savetxt('submission.csv', retArray, delimiter=',',fmt='%i')
    print 'done'
