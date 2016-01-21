import nltk 
from nltk.corpus import brown
from nltk.probability import LidstoneProbDist, WittenBellProbDist
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from ngram import NgramModel
import numpy as np
import operator
import pp
from time import time, sleep
estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
alphabet = 'abcdefghijklmnopqrstuvwxyz'
unigram = nltk.FreqDist(brown.words()) 
import copy_reg
import types
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
        return func.__get__(obj, cls)


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
        return func.__get__(obj, cls)



copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
class HMM(object):
    def __init__(self, matrixE):
        self.matrixE = matrixE
        self.M = 20
        self.bigram =  NgramModel(2, [ word.lower() for word in brown.words()], estimator) 
        self.word_dict_path  = '../data/word_by_len'
        self.word_dict = {}
        self.load_word_dict()
        self.ppservers=("*",)
        self.job_server = pp.Server(ppservers= self.ppservers) 
        print "Starting pp with", self.job_server.get_ncpus(), "workers"
        #self.trigram =  NgramModel(3, brown.words(), estimator)

    def load_word_dict(self):
        for i in xrange(1,24):
            with open("%s/%d.txt"%(self.word_dict_path, i), 'r') as fd:
                self.word_dict[i] = [line.strip().lower() for line in fd.readlines()]

    def hmmFirstOrder(self):
        self.viterbi()

    def viterbi(self,sentence):
        tokenizer = RegexpTokenizer(r'[\w\']+')
        self.token_words = tokenizer.tokenize(sentence)
        #self.token_words = word_tokenize(sentence)
        self.N = len(self.token_words)
        self.viterbiM = np.zeros((self.N, self.M+2), dtype = 'double')
        self.words = np.zeros((self.N , self.M + 2), dtype = 'str')
        self.backpointer = np.zeros( ( self.N , self.M + 2), dtype = 'int32')
        for i, word in enumerate(self.token_words):
            starttime = int(time())
            states = self.most_simlilar_words(word)
            find_time = int(time()) - starttime
            for j, (state, prob) in enumerate(states):
                if j < 10:
                    print state, prob
                self.words[i][j+1] = state
                if i == 0:
                    self.viterbiM[i][j+1] = self.bigram.prob(state, [u' '])*prob
                    self.backpointer[i][j+1] = 0
                else:
                    self.backpointer[i][j+1] , self.viterbiM[i][j+1] = max(enumerate([self.viterbiM[i-1][k+1]*self.bigram.prob(state, [ str(self.words[i-1][k+1]) ])* prob for k in xrange(self.M)]), key = operator.itemgetter(1))
            print "Eclapse %d s matching most possible word (%s), eclapse %d s for viterbi..."%(find_time, word , int(time()) - starttime - find_time)
        l = [self.viterbiM[self.N - 1][k+1]*self.endOfSentence(self.bigram, [ self.words[self.N - 1][k+1] ]) for k in xrange(self.M)]
        self.backpointer[self.N - 1][self.M+1], self.viterbiM[self.N - 1][self.M + 1] = max(enumerate(l), key = operator.itemgetter(1))
        path=[]
        end = self.backpointer[self.N - 1][self.M+1]
        for i in xrange(self.N - 1, 0, -1):
            path.append(end)
            end = self.backpointer[i][end+1]
        path.append(end)
        word_vector = []
        print self.N, len(path)
        for i in xrange(self.N-1, -1, -1):
            word_vector.append(self.words[self.N -1 - i][path[i]+1])
        return word_vector 

    def endOfSentence(self, lm, word):
        prob = 0
        for separator in [',', '.']:
            prob += lm.prob(separator, word) 
        return prob

       
    def most_simlilar_words(self, word):
        prob_list = {}
        jobs = self.paralize(word)
        prob_list.update(jobs)
        sorted_prob = sorted(prob_list.items(), key = operator.itemgetter(1), reverse=True)
        return sorted_prob[:self.M]

    def paralize(self, word):
        parts = 21
        jobs = []
        for index in xrange(parts):
            for i in xrange(3):
                if i == 1:
                    punish = 1
                else:
                    punish = 0.05
                length =  len(word) + i - 1
                if length == 0:
                    _list = []
                else:
                    _list = split_dict(self.word_dict[length], length, parts - 1, index) 
                jobs.append(self.job_server.submit(populate_prob, (_list, self.matrixE, unigram, length, word, {}, substitue, punish,), (substitue, insert, delete, probaWord, weightedPopularity, charToNum,),("nltk",) ))
        self.job_server.wait()
        stats = {}
        for job in jobs:
            stats.update(job())
        #self.job_server.print_stats()
        return stats

def split_dict( _list, len_w, parts, piece):
    avg_len = len(_list)/parts
    return _list[avg_len*piece: min(avg_len*(piece+1), len(_list))] 

def charToNum(a):
    special_char = [',','.','\'', ' ', '\n']
    if (a >= 'a' and a <= 'z') or (a >= 'A' and a <= 'Z'):
        return ord(a.lower()) - ord('a')
    elif a not in special_char:
        raise ValueError
    else:
        return {
            ' ': 26,
            ',': 27,
            '.': 28,
            '\'': 29,
            '\n': 30
        }[a]

def substitue( word):
    return set([word])

def delete(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in splits if b]
    return set(deletes)

def insert(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    inserts    = [a + c + b for a, b in splits for c in alphabet]
    return set(inserts)
    

def probaWord(typed_word, recog_word, matrixE):
    prod = 1
    for i in xrange(len(typed_word)):
        try:
            x_num = charToNum(typed_word[i])
            y_num = charToNum(recog_word[i])
            prod *= matrixE[y_num][x_num] 
        except:
            return -1
    return prod


def weightedPopularity( word, prod, unigram):
    if len(word) <=5 :
        weight = 0.95
    else:
        weight = 0.9995
    return weight*prod + (1 - weight )*unigram.freq(word)

def populate_prob(word_dict, matrixE, unigram, word_len,recog_word, prob_list, func, punish):
    if word_len > 0:
        for typed_word in word_dict:
            for second_word in func(typed_word) :
                prod = probaWord(second_word, recog_word, matrixE)
                if prod == -1:
                    continue
                prob = weightedPopularity(typed_word, prod, unigram) * punish 
                if typed_word in prob_list.keys():
                    prob_list[typed_word] = max(prob_list[typed_word], prob)
                else:
                    prob_list[typed_word] = prob
    else:
        for second_word in func(''):
            prob = probaWord(second_word, recog_word, matrixE)
            if prob == -1:
                continue
            if '' in prob_list.keys():
                prob_list[''] = max(prob_list[''], prob)
            else:
                prob_list[''] = prob

    return prob_list

