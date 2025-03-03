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

