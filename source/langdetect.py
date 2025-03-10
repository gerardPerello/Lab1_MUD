import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from utils import *
from classifiers import *
from preprocess import preprocess
import csv  # Add this import

seed = 42
random.seed(seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", 
                        help="Input data in csv format", type=str)
    parser.add_argument("-v", "--voc_size", 
                        help="Vocabulary size", type=int)
    parser.add_argument("-a", "--analyzer",
                         help="Tokenization level: {word, char}", 
                        type=str, choices=['word','char'])
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    
    # Languages
    languages = set(raw['language'])
    print('========')
    print('Languages', languages)
    print('========')

    # Split Train and Test sets
    X=raw['Text']
    y=raw['language']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    print('========')
    print('Split sizes:')
    print('Train:', len(X_train))
    print('Test:', len(X_test))
    print('========')
    

    # Preprocess text (Word granularity only)
    if args.analyzer == 'word':
        X_train = X_train.apply(lambda sentence: preprocess(sentence, y_train[X_train.index[X_train == sentence].tolist()[0]]))
        X_test = X_test.apply(lambda sentence: preprocess(sentence, y_test[X_test.index[X_test == sentence].tolist()[0]]))

   
    #Compute text features
    features, X_train_raw, X_test_raw = compute_features(X_train, 
                                                            X_test, 
                                                            analyzer=args.analyzer, 
                                                            max_features=args.voc_size)

    print('========')
    print('Number of tokens in the vocabulary:', len(features))
    coverage = compute_coverage(features, X_test.values, analyzer=args.analyzer)
    print('Coverage: ', coverage)
    print('========')


    #Apply Classifier  
    X_train, X_test = normalizeData(X_train_raw, X_test_raw)
    y_predict = applyNaiveBayes(X_train, y_train, X_test)
    y_predict2 = applySVM(X_train, y_train, X_test)
    
    print('========')
    print('Prediction Results:')    
    f1,f2,f3 = plot_F_Scores(y_test, y_predict)
    f11,f22,f33 = plot_F_Scores(y_test, y_predict2)
    print('========')
    
    plot_Confusion_Matrix(y_test, y_predict, color="Greens", analyzer=args.analyzer, voc_size=args.voc_size, classifier="NaiveBayes") 
    plot_Confusion_Matrix(y_test, y_predict2, color="Greens", analyzer=args.analyzer, voc_size=args.voc_size, classifier="SVM") 


    #Plot PCA
    print('========')
    print('PCA and Explained Variance:') 
    pca = plotPCA(X_train, X_test,y_test, languages, args.voc_size, args.analyzer) 
    print('========')

    # Save results to CSV
    setResults(args.voc_size, args.analyzer, 'NaiveBayes', f1, f2, f3, coverage, pca)
    setResults(args.voc_size, args.analyzer, 'SVM', f11, f22, f33, coverage, pca)
