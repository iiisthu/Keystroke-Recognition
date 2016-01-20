from nltk import FreqDist
from nltk.corpus import brown
from nltk.probability import LidstoneProbDist, WittenBellProbDist
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from ngram import NgramModel
import numpy as np
import operator
from time import time
estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
class HMM(object):
    def __init__(self, matrixE, charToNum):
        self.matrixE = matrixE
        self.M = 20
        self.charToNum = charToNum
        self.unigram = FreqDist(brown.words()) 
        self.bigram =  NgramModel(2, [ word.lower() for word in brown.words()], estimator) 
        self.word_dict_path  = '../data/word_by_len'
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.word_dict = {}
        self.load_word_dict()
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
                if j < 3:
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
    def weightedPopularity(self, word, prod):
        weight = 0.95
        return weight*prod + (1 - weight )*self.unigram.freq(word)
    def insert_sort(self, prob_list, sorted_word_list, new_word, new_prob):
        if len(sorted_word_list) < self.M:
            sorted_word_list = self._insert(prob_list, sorted_word_list, new_word, new_prob)
        else:
            if new_prob <= prob_list[sorted_word_list[-1]]:
                return sorted_word_list
            else:
                sorted_word_list = self._insert(prob_list, sorted_word_list, new_word, new_prob)
                del prob_list[sorted_word_list[-1]] 
                del sorted_word_list[-1] 
        return sorted_word_list 

    def _insert(self, prob_list, sorted_word_list, new_word, new_prob):
        if new_word in prob_list.keys():
            if new_prob <= prob_list[new_word]:
                return sorted_word_list
            else:
                sorted_word_list.remove(new_word)
        for i in reversed(xrange(len(sorted_word_list))): 
            if prob_list[sorted_word_list[i]] > new_prob:
                if i + 1 != len(sorted_word_list):
                    sorted_word_list = sorted_word_list[:(i+1)] + [new_word] + sorted_word_list[(i+1):]
                else:
                    sorted_word_list.append(new_word)
                prob_list[new_word] = new_prob
                return sorted_word_list
            else:
                continue
        sorted_word_list = [new_word] + sorted_word_list
        prob_list[new_word] = new_prob
        return sorted_word_list

    def populate_prob(self, word_len,recog_word, prob_list, func, punish, sorted_word_list):
        if word_len > 0:
            for typed_word in self.word_dict[word_len]:
                for second_word in func(typed_word) :
                    prod = self.probaWord(second_word, recog_word)
                    if prod == -1:
                        continue
                    prob = self.weightedPopularity(typed_word, prod) * punish 
                    sorted_word_list =  self.insert_sort(prob_list, sorted_word_list, typed_word, prob)   
        else:
            for second_word in func(''):
                prod = self.probaWord(second_word, recog_word)
                if prod == -1:
                    continue
                sorted_word_list = self.insert_sort(prob_list, sorted_word_list, '', prod)
        return sorted_word_list

        
    def most_simlilar_words(self, word):
        prob_list = {}
        sorted_word_list = []
        sorted_word_list = self.populate_prob(len(word), word, prob_list, self.substitue, 1 , sorted_word_list)
        sorted_word_list = self.populate_prob(len(word)+1, word, prob_list, self.delete, 0.05 , sorted_word_list)
        sorted_word_list = self.populate_prob(len(word)-1, word, prob_list, self.insert, 0.05 , sorted_word_list)

        for i in xrange(len(sorted_word_list)):
            yield (sorted_word_list[i], prob_list[sorted_word_list[i]])

    def substitue(self, word):
        return set([word])

    def delete(self,word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [a + b[1:] for a, b in splits if b]
        return set(deletes)

    def insert(self,word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        inserts    = [a + c + b for a, b in splits for c in self.alphabet]
        return set(inserts)
        

    def probaWord(self,typed_word, recog_word):
        prod = 1
        for i in xrange(len(typed_word)):
            try:
                x_num = self.charToNum(typed_word[i])
                y_num = self.charToNum(recog_word[i])
                prod *= self.matrixE[y_num][x_num] 
            except:
                return -1
        return prod


