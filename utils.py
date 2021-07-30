
import numpy as np
import whois
from pyquery import PyQuery
from requests import get
import math




class UrlFeaturesExtractor(object):

    def __init__(self, url):

        self.domain = url.split('//')[-1].split('/')[0]
        self.url = url

        #try:
        #    self.whois = whois.query(self.domain).__dict__
        #except:
        #    self.whois = None

        #try:
        #    self.response = get(self.url)
        #    self.pq = PyQuery(self.response.text)
        #except:
        #    self.response = None
        #    self.pq = None



        string = self.url.strip()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        self.entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])


        digits = [i for i in self.url if i.isdigit()]
        self.numDigits = len(digits)

        self.urlLength =  len(self.url)


        self.numParameters = len(self.url.split('&')) - 1


        self.numFragments = len(self.url.split('#')) - 1

        self.numSubDomains = len(self.url.split('http')[-1].split('//')[-1].split('/')) -1

        self.domainExtension = self.url.split('.')[-1].split('/')[0]

        self.hasHttp = 'http:' in self.url

        self.hasHttps = 'https:' in self.url

        self.countDots = self.url.count('.')



        #Scrapping
        '''
        if self.pq is not None:
            self.bodyLength =  len(self.pq('html').text())
            titles = ['h{}'.format(i) for i in range(7)]
            titles = [self.pq(i).items() for i in titles]
            self.numTitles =  len([item for s in titles for item in s])
            self.numImages =  len([i for i in self.pq('img').items()])
            self.scriptLength = len(self.pq('script').text())
            self.numLinks =  len([i for i in self.pq('a').items()])
        else:
            self.numTitles = 0
            self.bodyLength = 0
            self.numLinks = 0
            self.numImages = 0
            self.scriptLength = 0


        '''



def urlToCsventry(url):
    url_extractor = UrlFeaturesExtractor(url)
    return url_extractor.url, url_extractor.domain, url_extractor.entropy, url_extractor.numDigits, url_extractor.numParameters, url_extractor.urlLength, url_extractor.numFragments, url_extractor.numSubDomains, url_extractor.domainExtension, url_extractor.hasHttp, url_extractor.hasHttps, url_extractor.countDots



def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')
    allTokens=[]
    for i in tokensBySlash:
        tokens = str(i).split('-')
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokentsByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    if 'com' in allTokens:
        allTokens.remove('com')
    return allTokens
