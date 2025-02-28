from sklearn.naive_bayes import MultinomialNB
from utils import toNumpyArray

# You may add more classifier methods replicating this function
def applyNaiveBayes(X_train, y_train, X_test):
    '''
    Task: Given some features train a Naive Bayes classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict


"""
from sklearn.neighbors import KNeighborsClassifier

def applyKNN(X_train, y_train, X_test, k=5):
    '''
    Task: Train a k-Nearest Neighbors classifier and return predictions.
    Input: 
        X_train -> Train features
        y_train -> Train labels
        X_test -> Test features
        k -> Number of neighbors to consider (default=5)
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    
    return y_predict

from sklearn.svm import SVC

def applySVM(X_train, y_train, X_test, kernel='linear'):
    '''
    Task: Train a Support Vector Machine (SVM) classifier and return predictions.
    Input: 
        X_train -> Train features
        y_train -> Train labels
        X_test -> Test features
        kernel -> Kernel type (default='linear', can be 'rbf', 'poly', etc.)
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = SVC(kernel=kernel)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    
    return y_predict

from sklearn.ensemble import RandomForestClassifier

def applyRandomForest(X_train, y_train, X_test, n_estimators=100):
    '''
    Task: Train a Random Forest classifier and return predictions.
    Input: 
        X_train -> Train features
        y_train -> Train labels
        X_test -> Test features
        n_estimators -> Number of trees in the forest (default=100)
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    
    return y_predict

from sklearn.linear_model import LogisticRegression

def applyLogisticRegression(X_train, y_train, X_test):
    '''
    Task: Train a Logistic Regression classifier and return predictions.
    Input: 
        X_train -> Train features
        y_train -> Train labels
        X_test -> Test features
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = LogisticRegression(max_iter=200)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    
    return y_predict

"""