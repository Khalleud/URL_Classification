import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utils import UrlFeaturesExtractor, urlToCsventry, getTokens
import torch
import transformers as ppb
from transformers import BertTokenizer



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

        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        model_class_bert, tokenizer_class_bert, pretrained_weights_bert = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        model = model_class_bert.from_pretrained(pretrained_weights_bert)


        urls = self.X_train['url']
        tokenized = urls.apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length = 512, truncation = True)))


        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

        attention_mask = np.where(padded != 0, 1, 0)

        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        X_train = last_hidden_states[0][:,0,:].numpy()

        pass
