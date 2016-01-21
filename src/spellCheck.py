#-*- coding: utf-8 -*-
#from googlecorrecter import correct 
import re
import pdb
import numpy as np
import operator
import hmm
import nltk
import sys
from nltk.tokenize import RegexpTokenizer
input_file = '../data/svm_output.txt'
output_file = '../data/svm_spell_check.txt'
perfect_file = '../data/source_1.txt'
bigram_file='../data/bigram.txt'
naive_file='../data/naive.txt'
def loadFile(filename):
    correct_label = ''
    predict_label = ''
    error_rate = 0.0
    with open(filename, 'r') as fd:
       for i, line in enumerate(fd):
            if i == 5:
                m = re.findall('\d+.\d+', line)
                error_rate = float(m[0])
            if i == 6:
                correct_label = line
            if i == 7:
                predict_label = line
    return error_rate, correct_label, predict_label

class SpellChecker(object):
    def __init__(self, goal, _input):
        self.goal_string = goal
        self._input_string = _input
        self.output = []
        self.output_nd = []
        self.word_error = 0.0
        self.ori_word_error = 0.0
        self.ori_char_error = 0.0
        self.char_error = 0.0
        self.special_char = [',','.','\'', ' ', '\n']
        self.charNum = 31
        self.M = 20
        self.separators = [',','.']
        self.tokenizer = RegexpTokenizer(r'[\w\']+')
        self.matrixE = np.zeros((self.charNum, self.charNum), dtype = 'double') 
        self.goal_nd = self.separate_delimiter(goal) 
        self._input_nd = self.separate_delimiter(_input)
        self.sentences = _input #nltk.sent_tokenize(_input)
        with open(perfect_file, 'r') as fd:
            self.perfect = self.separate_delimiter(fd.readlines()[0])
  
    def separate_delimiter(self, _input):
        return self.tokenizer.tokenize(_input)
        '''_input_tmp = [x for x in _input.split()]
        output = []
        output_no_separator = []
        for word in _input_tmp:
            flag = 0
            for delimeter in self.separators:
                if word.endswith(delimeter):
                    output.append(word[:-1])
                    output_no_separator.append(word[:-1])
                    output.append(delimeter)
                    flag = 1   
                elif word.startswith(delimeter):
                    output.append(delimeter)
                    output_no_separator.append(word[1:])
                    output.append(word[1:])
                    flag = 1   
            if flag == 0:
                output.append(word)
                output_no_separator.append(word)
        return output, output_no_separator'''
        
        
    def errorRate(self, goal, output):
        count1 = 0
        set_list = [output, goal]
        offset = [0 ,0]
        total = len(goal)
        total_char = 0
        output_char = 0
        print total
        for index in xrange(len(goal)):
            if index + offset[0] >= len(set_list[0]) :
                break
                ## 错位两位仍可纠正
            if set_list[0][index + offset[0]] != set_list[1][index]: 
                for j in xrange(1, 3):
                    if index + j < total and index+ offset[0]< len(set_list[0]) and set_list[0][index + offset[0]] == set_list[1][index + j]:
                        offset[0] -= j
                        break
                    if index - j >= 0 and index + offset[0] < len(set_list[0]) and set_list[0][index + offset[0]] == set_list[1][index - j]:
                        offset[0] += j
                        break
            if index + offset[0] >= len(set_list[0]):
                break
            x = set_list[0][ index + offset[0] ]
            y = set_list[1][ index + offset[1] ]
            print x, y
            if x == y:
                count1 = count1 + 1
            for i,j in zip(x, y):
                if i == j:
                    output_char += 1
            total_char += len(y)
        set_list_string = [ ' '.join(l) for l in set_list]
        self.word_error = 1 - 1.0 * count1 / total
        self.char_error = 1 - 1. * output_char / total_char
        print "correct %d words,total %d words"%(count1, total)
        print "correct %d characters,total %d characters"%(output_char, total_char)
        '''with open(output_file, 'w') as fd:
            fd.write('Origin word error:%.8lf \n Current word error: %.8lf\n'%(self.ori_word_error, self.word_error))
            ' '.join( spellchecker.output )
            fd.write('%s\n%s\n%s'%(' '.join( self.goal ), ' '.join( self._input ),' '.join( self.output )))
        print '%s generated .'%output_file 
'''
    def correctSentence(self):
        print "Start spell checking...\n"
        #self.output = [ correct(x) for x in  self._input ]
        print "Finish spell checking...\n"
    
    def naiveCorrect(self):
        output_nd = []
        _hmm = hmm.HMM(self.matrixE)
        for x in self._input_nd:
            y_list = _hmm.most_simlilar_words(x)
            print '%s -> %s'%(x, y_list[0][0])
            output_nd.append(y_list[0][0])
        with open(naive_file, 'w') as fd:
            fd.write(' '.join(output_nd))
        return  output_nd

    def bigram(self):
        output = []
        output_nd = []
        _hmm = hmm.HMM(self.matrixE)
        output_nd = _hmm.viterbi(self.sentences) 
        with open(bigram_file, 'r') as fd:
            fd.write(' '.join(output_nd))
        #    output_string = fd.readlines()
        #output_nd = self.separate_delimiter(output_string[0])
        return output_nd
        
    def computeMatrixE(self):
        uniCount = np.zeros(self.charNum, dtype='int32') 
        biCount = np.zeros((self.charNum, self.charNum), dtype='int32') 
        for x, y in zip(self.goal_string, self._input_string):
            x_num = self.charToNum(x)
            y_num = self.charToNum(y)
            uniCount[x_num] = uniCount[x_num] + 1
            biCount[y_num][x_num] += 1
        for i in xrange(self.charNum):
            self.matrixE[i] = [ 0.01 if value == 0 or biCount[i][j] == 0 else biCount[i][j]*1. / uniCount[j] for j, value in enumerate(uniCount) ]
    
    def charToNum(self,a):
        if a >= 'a' and a <= 'z' or a >= 'A' and a <= 'Z':
            return ord(a.lower()) - ord('a')
        elif a not in self.special_char:
            raise ValueError
        else:
            return {
                ' ': 26,
                ',': 27,
                '.': 28,
                '\'': 29,
                '\n': 30
            }[a]

if __name__ == '__main__':
    ori_error_rate, correct_label , predict_label = loadFile(input_file)
    spellchecker = SpellChecker(goal = correct_label, _input = predict_label)
    #spellchecker.correctSentence()
    #spellchecker.errorRate()
    spellchecker.computeMatrixE()
    #spellchecker.naiveCorrect()
    #bigram_out = spellchecker.bigram()
    #spellchecker.errorRate(spellchecker.goal_nd, bigram_out)
    #spellchecker.errorRate(spellchecker.perfect, bigram_out)
    arg = sys.argv[1].strip()
    print arg
    funcdict = {
    'naive' : spellchecker.naiveCorrect,
    'bigram': spellchecker.bigram
    }
    out = funcdict[arg]()
    spellchecker.errorRate(spellchecker.perfect, out)
    #spellchecker.errorRate(spellchecker.perfect, naive_out)
    #spellchecker.errorRate(spellchecker.goal_nd, spellchecker._input_nd)
    #print ' '.join(spellchecker.goal)
    #print ' '.join(spellchecker.output)
    #spellchecker.output_nd = spellchecker._input_nd
    #spellchecker.errorRate()
