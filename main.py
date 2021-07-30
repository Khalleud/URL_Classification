import pandas as pd
from sklearn.model_selection import train_test_split
from FeaturesExtractor import FeaturesExtractor
from classifiers import clf_log
from evaluateModel import evaluate_model

if __name__ == "__main__":

    df = pd.read_csv('Data.csv')
    print(df.head())

    df = df[:12000]

    y = df[['target']]
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    featureExtractor = FeaturesExtractor(X_train, X_test)

    #X_train, X_test = featureExtractor.getLexicalFeatures()

    max_df = 1.0
    min_df = 1
    max_features = 1000
    X_train, X_test = featureExtractor.getBagOfWordsFeatures(max_df, min_df, max_features)

    print(y)
    clf_log.fit(X_train, y_train)


    y_pred = clf_log.predict(X_test)

    accuracy, weighted_avg = evaluate_model(y_test, y_pred)
    print(accuracy, weighted_avg)
