# URL_Classification


Instructions:

1. Run : mkdir data
2. Download [Data](https://drive.google.com/file/d/1q4EYndbegewI6wc59CiJSY6t9YitnHD4/view) then unzip it in 'data' repository
3. Run : pip install -r requirements.txt
4. Run : python prepareData.py
5. Set paramets
6. Run : python main.py
7. Results are saved in the 'result' repository

+ Parameters could be changed in setting.conf file



## Todo : 
+ Study similarities ( cosine, jaccard...) with the features (lexical, bag of words...) already done between the urls and the classes of the training set
+ Optimize hyperparameters
+ Preprocessing data : preprocess target, use better tokenizer
+ Use Embeddings Techniques: Bert, CNN for features extraction
