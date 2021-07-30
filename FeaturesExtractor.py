import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utils import UrlFeaturesExtractor, urlToCsventry, getTokens


class FeaturesExtractor:

    def __init__(self, X_train, X_test):


        self.X_train = X_train

        self.X_test = X_test




    def getLexicalFeatures(self):


        column_names = ["url", "domain", "entropy", "numDigits", "numParameters", "urlLength","numFragments", "subDomains", "domainExtension", "hasHttp", "hasHttps", "countDots "]

        X_train = pd.DataFrame(columns = column_names)
        X_test = pd.DataFrame(columns = column_names)

        for index, row in self.X_train.iterrows():

            X_train.loc[index] = urlToCsventry(row['url'])

        for index, row in self.X_test.iterrows():

            X_test.loc[index] = urlToCsventry(row['url'])


        return X_train, X_test



    def getBagOfWordsFeatures(self, max_df  , min_df  , max_features ):

        X_train = self.X_train['url'].to_list()
        X_test = self.X_test['url'].to_list()

        vectorizer = CountVectorizer( tokenizer=getTokens, max_features= max_features, max_df = max_df, min_df = min_df)

        vectorizer.fit(X_train)

        count_X_train = pd.DataFrame(vectorizer.transform(X_train).toarray(), columns=vectorizer.get_feature_names())

        count_X_test = pd.DataFrame(vectorizer.transform(X_test).toarray(), columns=vectorizer.get_feature_names())

        return count_X_train, count_X_test


    def getSmoothTfIDF(self, max_df  , min_df  , max_features):



        X_train = self.X_train['url'].to_list()
        X_test = self.X_test['url'].to_list()

        vectorizer = TfidfVectorizer( tokenizer=getTokens, use_idf=True, smooth_idf=True, sublinear_tf=False, max_df = max_df, min_df = min_df, max_features = max_features)

        vectorizer.fit(X_train)

        count_X_train = pd.DataFrame(vectorizer.transform(X_train).toarray(), columns=vectorizer.get_feature_names())

        count_X_test = pd.DataFrame(vectorizer.transform(X_test).toarray(), columns=vectorizer.get_feature_names())

        return count_X_train, count_X_test



    def getTfIdf(self, max_df  , min_df  , max_features):

        X_train = self.X_train['url'].to_list()
        X_test = self.X_test['url'].to_list()

        vectorizer = TfidfVectorizer( tokenizer=getTokens, max_features= max_features, max_df = max_df, min_df = min_df)
        vectorizer.fit(X_train)

        count_X_train = pd.DataFrame(vectorizer.transform(X_train).toarray(), columns=vectorizer.get_feature_names())

        count_X_test = pd.DataFrame(vectorizer.transform(X_test).toarray(), columns=vectorizer.get_feature_names())

        return count_X_train, count_X_test


    def getBertEmedding():

        pass
