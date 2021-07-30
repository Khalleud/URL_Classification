import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from FeaturesExtractor import FeaturesExtractor
from classifiers import clf_log, clf_knn
from evaluateModel import evaluate_model
import configparser
import os
import sys



if __name__ == "__main__":



    print('Read Data...')
    df = pd.read_csv('Data.csv')

    df = df[:1100]

    y = df[['target']]
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print("Extracting Features ...")
    featureExtractor = FeaturesExtractor(X_train, X_test)

    config = configparser.ConfigParser()
    config.read('settings.conf')

    max_df = int(config.get('vectorizers','max_df')) if config.get('vectorizers','max_df').isdigit() else  float(config.get('vectorizers','max_df'))
    min_df = int(config.get('vectorizers','min_df')) if config.get('vectorizers','min_df').isdigit() else  float(config.get('vectorizers','min_df'))
    max_features = int(config.get('vectorizers','numberOfFeatures')) if config.get('vectorizers','numberOfFeatures').isdigit() else None




    X_train, X_test = featureExtractor.getBagOfWordsFeatures(max_df, min_df, max_features)

    classes = df['target'].unique()

    print("Training...")
    clf_log.fit(X_train, y_train)


    y_pred = clf_log.predict(X_test)


    print("Evaluation...")

    if not os.path.exists('results'):
        os.mkdir('results')

    all_report, accuracy, weighted_avg = evaluate_model(y_test, y_pred)
    print(accuracy, weighted_avg)

    with open(os.path.join('results', 'report.txt'), 'w') as f:


        sys.stdout = f # Change the standard output to the file we created.
        print(all_report)
        print('accuracy : {}'.format(accuracy))
        print('weighted_Precision : {}'.format(weighted_avg['precision']))
        print('weighted_Recall    : {}'.format(weighted_avg['recall']))
        print('weighted_F-score   : {}'.format(weighted_avg['f1-score']))
        print('weighted_Support   : {}'.format(weighted_avg['support']))
        sys.stdout = sys.__stdout__
