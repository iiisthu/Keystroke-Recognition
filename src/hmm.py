import nltk 
from nltk.corpus import brown
from nltk.probability import LidstoneProbDist, WittenBellProbDist
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from ngram import NgramModel
import numpy as np
import operator
import pp
import sys
from time import time, sleep,gmtime, strftime
estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
alphabet = 'abcdefghijklmnopqrstuvwxyz'
unigram = nltk.FreqDist(brown.words()) 
log_file = '../data/hmm.%s.log'%(strftime('%Y-%m-%d-%H:%M:%S',gmtime()))
#log_file = sys.stdout 
import copy_reg
import types
def LOG(filename, string):
    with open(filename, 'a') as fd:
        fd.write(string)

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
        self.nngram = None
        self.word_dict_path  = '../data/word_by_len'
        self.word_dict = {}
        self.load_word_dict()
        self.ppservers=("localhost", )
        self.threshold  = 0.00000001
        self.job_server = pp.Server(ppservers= self.ppservers) 
        print "active nodes: ", self.job_server.get_active_nodes()
        #self.trigram =  NgramModel(3, brown.words(), estimator)

    def load_word_dict(self):
        for i in xrange(1,24):
            with open("%s/%d.txt"%(self.word_dict_path, i), 'r') as fd:
                self.word_dict[i] = [line.strip().lower() for line in fd.readlines()]

    def segment(self, word):
        prob_max = 0.0
        first, second = [],[] 
        special_char = [',','.','\'', ' ', '\n']
        for c in xrange(1,len(word)-1):
            first_cur = self.most_similar_words(word[:c])
            second_cur  = self.most_similar_words(word[(c+1):])
            for (key1, val1) in first_cur:
                for (key2, val2) in second_cur:
                    prob = val1 * val2 * max( [ self.matrixE[charToNum(sc)][charToNum(word[c])] for sc in special_char ])
                    if prob > prob_max:
                        prob_max = prob
                        LOG( log_file, '(%s, %d,  %f, %s, %s)\n' %(word, c, prob_max, key1, key2)) 
                        first = first_cur
                        second = second_cur
        return first, second, prob_max 
            
    def viterbi(self,sentence, order):
        tokenizer = RegexpTokenizer(r'[\w\']+')
        self.token_words = tokenizer.tokenize(sentence)
        #self.token_words = word_tokenize(sentence)
        self.nngram =  NgramModel(order, [ word.lower() for word in brown.words()], estimator) 
        self.N = len(self.token_words)
        MAX_OFFSET = 10
        if order == 2:
            viterbiM = np.zeros((self.N + MAX_OFFSET, self.M + 2), dtype = 'double')
            words = np.zeros((self.N + MAX_OFFSET, self.M + 2), dtype = object)
            backpointer = np.zeros( ( self.N + MAX_OFFSET, self.M + 2), dtype = 'int32')
            offset = np.zeros( self.N, dtype = 'int32' )
            return self.viterbi_first_order(viterbiM, words, backpointer, offset)
        if order == 3:
            viterbiM = np.zeros((self.N + MAX_OFFSET, self.M + 2, self.M + 2), dtype = 'double')
            words = np.zeros((self.N + MAX_OFFSET, self.M + 2), dtype = object)
            backpointer = np.zeros( ( self.N + MAX_OFFSET, self.M + 2, self.M + 2), dtype = 'int32')
            offset = np.zeros( self.N, dtype = 'int32' )
            return self.viterbi_second_order(viterbiM, words, backpointer, offset)
            
    def most_similar_words_with_split(self,word):
        states = self.most_similar_words(word)
        # If the probability is too small , it's possible that space is recognized as character
        states_tmp = []
        if states[0][1] < self.threshold:
            LOG(log_file, "Beyond bottom threshold, try to split %s\n"%word)
            states1, states2, prob  = self.segment(word)
            if prob > states[0][1]:
                states_tmp.append(states1)
                states_tmp.append(states2)
                splited = True
            else:
                states_tmp.append(states)
        else:
            states_tmp.append(states)
        return states_tmp

    def viterbi_first_order(self,viterbiM, words, backpointer, offset):
        print 'call_bigram'
        for i, word in enumerate(self.token_words):
            # Find matched word by probability
            splited = False
            starttime = int(time())
            print 'finding similar words(%s)'%word
            states_tmp = self.most_similar_words_with_split(word)
            print states_tmp
            find_time = int(time()) - starttime
            # initial current offset
            cur_offset = 0
            if i != 0:
                cur_offset =  offset[i-1]
            # recursion step
            print 'iterate states'
            for index, st in enumerate(states_tmp): 
                id_with_offset = cur_offset + index + i
                for j, (state, prob) in enumerate(st):
                    # print something out for debuging
                    if j < 10:
                        LOG(log_file, '%s, %f\n'%( state, prob ))
                        print state,prob
                    words[i][j+1] = state
                    if i == 0:
                        pref = [u' ']
                        viterbiM[id_with_offset][j+1] = self.nngram.prob(state, pref)*prob
                        backpointer[id_with_offset][j+1] = 0
                    else:
                        l_tmp =  max(enumerate([
                                                viterbiM[id_with_offset - 1 ][k+1]
                                                *self.nngram.prob(state, [ str(words[id_with_offset - 1 ][k+1]) ])
                                                * prob 
                                                for k in xrange(self.M)
                                            ]), 
                                                key = operator.itemgetter(1)
                                            )

                        backpointer[id_with_offset][j+1] , viterbiM[id_with_offset][j+1] = l_tmp 
            if splited:
                offset[i] = cur_offset + 1
            LOG( log_file, "Eclapse %d s matching most possible word (%s)," 
                            "eclapse %d s for viterbi...\n"
                            %(find_time, word , int(time()) - starttime - find_time))
        
        final_offset = offset[-1] 
        print 'final offset is %d'%final_offset
        # termination step
        l = [
                viterbiM[self.N + final_offset - 1][k+1]
                *self.endOfSentence(self.nngram, [ words[self.N + final_offset - 1][k+1] ]) 
                for k in xrange(self.M)
            ]
        backpointer[self.N + final_offset - 1][self.M+1], viterbiM[self.N + final_offset - 1][self.M + 1] = max(enumerate(l), key = operator.itemgetter(1))
        # backtrace
        path=[]
        end = backpointer[self.N + final_offset - 1][self.M+1]
        for i in xrange(self.N + final_offset - 1, 0, -1):
            path.append(end)
            end = backpointer[i][end+1]
        path.append(end)
        word_vector = []
        for i in xrange(self.N + final_offset -1, -1, -1):
            word_vector.append(words[self.N + final_offset -1 - i][path[i]+1])
        return word_vector 

    def viterbi_second_order(self,viterbiM, words, backpointer, offset):
        for i, word in enumerate(self.token_words):
            # Find matched word by probability
            starttime = int(time())
            splited = False
            states_tmp = self.most_similar_words_with_split(word)
            find_time = int(time()) - starttime
            # initial current offset
            cur_offset = 0
            if i != 0:
                cur_offset =  offset[i-1]
            # recursion step
            for index, st in enumerate(states_tmp): 
                id_with_offset = cur_offset + index + i
                for j, (state, prob) in enumerate(st):
                    # print something out for debuging
                    if j < 20:
                        LOG(log_file, '%s, %f\n'%( state, prob ))
                        print state,prob
                    words[i][j+1] = state
                    for l in xrange(self.M):
                        if i == 0:
                            pref = [u' ']
                            viterbiM[id_with_offset][l+1][j+1] = self.nngram.prob(state, pref)*prob
                            backpointer[id_with_offset][l+1][j+1] = 0
                        elif i == 1:
                            backpointer[id_with_offset][l+1][j+1] , viterbiM[id_with_offset][l+1][j+1] = max(enumerate([
                                                                        viterbiM[id_with_offset - 1 ][k+1][l+1]
                                                                        *self.nngram.prob(state, [
                                                                        ' ',
                                                                        str(words[id_with_offset - 1][l+1]) ])
                                                                        * prob 
                                                                        for k in xrange(self.M)
                                                                    ]), 
                                                                    key = operator.itemgetter(1)
                                                            )
                        else:
                            backpointer[id_with_offset][l+1][j+1] , viterbiM[id_with_offset][l+1][j+1] = max(enumerate([
                                                                        viterbiM[id_with_offset - 1 ][k+1][l+1]
                                                                        *self.nngram.prob(state, [
                                                                        str(words[id_with_offset - 2 ][k+1]), 
                                                                        str(words[id_with_offset - 1][l+1]) ])
                                                                        * prob 
                                                                        for k in xrange(self.M)
                                                                    ]), 
                                                                    key = operator.itemgetter(1)
                                                            )
            if splited:
                offset[i] = cur_offset + 1
            LOG( log_file, "Eclapse %d s matching most possible word (%s)," 
                            "eclapse %d s for viterbi...\n"
                            %(find_time, word , int(time()) - starttime - find_time))
        
        final_offset = offset[-1] 
        print 'final offset is %d'%final_offset
        # termination step
        for l in xrange(self.M):
            backpointer[self.N + final_offset - 1][l+1][self.M+1],viterbiM[self.N + final_offset - 1][l + 1][self.M+1] = max(enumerate([
                            viterbiM[self.N + final_offset - 1][k+1][l+1]
                            *self.endOfSentence(self.nngram, [ str(words[self.N + final_offset - 2][k+1]) 
                                                                , str(words[self.N + final_offset - 1][l+1]) ]) 
                            for k in xrange(self.M)
                            ]),
                            key = operator.itemgetter(1))
        backpointer[self.N + final_offset - 1][self.M+1][self.M+1],viterbiM[self.N + final_offset - 1][self.M+1][self.M+1] = max(enumerate([
                            viterbiM[self.N + final_offset - 1][k+1][self.M+1]
                            *self.endOfSentence(self.nngram, [ str(words[self.N + final_offset - 1][k+1]) ])
                            for k in xrange(self.M)
                            ]),
                            key = operator.itemgetter(1))
        # backtrace
        import pdb
        pdb.set_trace()
        path=[]
        end = backpointer[self.N + final_offset - 1][self.M+1][self.M+1]
        path.append(end)
        last_end = backpointer[self.N + final_offset - 1][end][self.M+1]
        path.append(last_end)
        for i in xrange(self.N + final_offset - 1, 1, -2):
            end = backpointer[i][last_end+1][end + 1]
            last_end = backpointer[i-1][end + 1][last_end + 1]
            path.append(end)
            path.append(last_end)

        path.append(end)
        print path
        word_vector = []
        for i in xrange(self.N + final_offset -1, -1, -1):
            word_vector.append(words[self.N + final_offset -1 - i][path[i]+1])
        print word_vector
        return word_vector 


    def endOfSentence(self, lm, word):
        prob = 0
        for separator in [',', '.']:
            prob += lm.prob(separator, word) 
        return prob

       
    def most_similar_words(self, word):
        prob_list = {}
        jobs = self.paralize(word)
        prob_list.update(jobs)
        sorted_prob = sorted(
                                prob_list.items(),
                                key = operator.itemgetter(1), 
                                reverse=True
                            )
        return sorted_prob[:self.M]

    def paralize(self, word):
        parts = 17
        jobs = []
        for index in xrange(parts):
            for i in xrange(3):
                if i == 1:
                    punish = 1
                else:
                    punish = 0.05
                length =  len(word) + i - 1
                if length <= 0:
                    _list = []
                else:
                    _list = split_dict(self.word_dict[length], length, parts - 1, index) 
                jobs.append(
                            self.job_server.submit(
                                                    populate_prob, 
                                                    (_list, self.matrixE, unigram, length, word, {}, substitue, punish,), 
                                                    (substitue, insert, delete, probaWord, weightedPopularity, charToNum,),
                                                    ("nltk",) 
                                                )
                            )
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
    weight = 1 - 0.1**len(word)
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

