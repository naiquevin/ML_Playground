from __future__ import division

import json
import operator
import string
# from pprint import pprint

import numpy as np
import pandas as pd
from lookupy import Collection, Q
from textmining import TermDocumentMatrix


def stopwords():
    with open('stopwords.txt') as f:
        return set([x.strip() for x in f.read().split(',')])


STOPWORDS = stopwords()

## The 4 terms, 'software', 'setup', 'hardware', 'dream' appear in all
## documents as these are a part of the question asked. So we ignore these
STOPWORDS.update(set(['software', 'setup', 'hardware', 'dream']))
STOPWORDS.update(set(['time', 'work', 'new', 'two', 'make', 'years']))

DEV_CATEGORIES = ['developer', 'hacker', 'software', 'sysadmin', 'linux']


def depunctuate(text):
    table = dict((ord(char), None) for char in string.punctuation)
    return text.translate(table)


def build_tdm1(doc_list, cutoff=2):
    texts = (x['body'] for x in doc_list)
    # doc1 = u'John and Bob are brothers. Pink'
    # doc2 = u'John went to the store. The store was closed. Pink'
    # doc3 = u'Bob went to the store too. And Pink Floyd'
    # doc_list = [doc1, doc2, doc3]
    tdm = TermDocumentMatrix()
    for text in texts:
        doc = text.lower()
        # remove stopwords
        doc = ' '.join([w for w in doc.split() if w not in STOPWORDS])
        doc = depunctuate(doc)
        doc = ' '.join([w for w in doc.split() if w not in STOPWORDS])
        # remove punctuation
        tdm.add_doc(doc)

    tdm_as_list = list(tdm.rows(cutoff=cutoff))
    words, docfreq = tdm_as_list[0], tdm_as_list[1:]
    data = dict(zip(words, np.array(docfreq).T))
    tdm_df = pd.DataFrame(data)
    return tdm_df


def build_tdm2(doc_list):
    keyterms = list(set(reduce(lambda x, y: x + y, [x['keyterms'] for x in doc_list], [])))
    matrix = [[int(k in d['keyterms']) for d in doc_list] for k in keyterms]
    data = dict(zip(keyterms, matrix))
    tdm_df = pd.DataFrame(data)
    return tdm_df


## choose the tdm/dataframe builder here
build_tdm = build_tdm2


def training_set(vec):
    tdm_df = build_tdm(vec)
    term_freq = tdm_df.sum().order(ascending=False)
    df = pd.DataFrame(term_freq, columns=['frequency'])
    df['occurrence'] = tdm_df.apply(lambda x: x > 0).sum() / len(tdm_df)
    df['density'] = df['frequency'] / df['frequency'].sum()
    df = df.sort(['occurrence'], ascending=[0])
    return df


def classifier(doc, training_df, prior=0.5, c=1e-6):
    tdm_df = build_tdm([doc])
    word_freq = tdm_df.sum()
    common_words = set(word_freq.index).intersection(set(training_df.index))
    if len(common_words) < 1:
        return prior * (c ** len(word_freq))
    else:
        common_words_probs = training_df.select(lambda i: i in common_words)['occurrence']
        return prior * common_words_probs.prod() * (c ** (len(word_freq) - len(common_words)))


def classify(doc, dev_df, nondev_df, priors):
    dev_prob = classifier(doc, dev_df, prior=priors[0])
    nondev_prob = classifier(doc, nondev_df, prior=priors[1])
    return TestClassification(doc, dev_prob, nondev_prob)


class TestClassification(object):

    def __init__(self, doc, dev_prob, nondev_prob):
        self.doc = doc
        self.dev_prob = dev_prob
        self.nondev_prob = nondev_prob
    
    @property
    def expected(self):
        is_dev = any([c in DEV_CATEGORIES for c in self.doc['categories']])
        return 'D' if is_dev else 'N'
        
    @property
    def computed(self):
        return 'D' if self.dev_prob > self.nondev_prob else 'N'


if __name__ == '__main__':
    with open('interviews.json') as f:
        # all interviews
        interviews = json.load(f)

    num_training = 300

    # training data
    training = interviews[:num_training] 

    # testing data
    testing = interviews[num_training:]

    c = Collection(training)

    ## get training data of type 'developer'
    dev_lookups = [Q(categories__contains=x) for x in DEV_CATEGORIES]
    training_developer = list(c.filter(reduce(operator.or_, dev_lookups)))

    ## get training data of type 'non-developer'
    nondev_lookups = map(operator.invert, dev_lookups)
    training_nondeveloper = list(c.filter(reduce(operator.and_, nondev_lookups)))

    priors = (len(training_developer)/num_training, len(training_nondeveloper)/num_training)

    # get training data as a dataframe
    dev_df = training_set(training_developer)
    nondev_df = training_set(training_nondeveloper)

    # print dev_df.select(lambda i: i == 'Python')
    # print priors
    # exit(0)

    results = [classify(interview, dev_df, nondev_df, priors) for interview in testing]

    print
    print '==== Summary ===='
    print 
    print 'True-Positives', sum([int(c.expected == 'D' and c.computed == 'D') for c in results])
    print 'True-Negatives', sum([int(c.expected == 'N' and c.computed == 'N') for c in results])
    print 'False-Positives', sum([int(c.expected == 'N' and c.computed == 'D') for c in results])
    print 'False-Negatives', sum([int(c.expected == 'D' and c.computed == 'N') for c in results])
    print '================='

    print
    print '==== Details ===='
    print
    for result in results:
        print result.doc['name'], ':', result.expected, result.computed
    print '================='

