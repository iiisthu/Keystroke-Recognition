#-*- coding: utf-8 -*-
#from googlecorrecter import correct 
import re
import pdb
import numpy as np
import operator

input_file = '../data/svm_output.txt'
output_file = '../data/svm_spell_check.txt'
word_dict_path = '../data/word_by_len'
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
        self.matrixE = np.zeros((self.charNum, self.charNum), dtype = 'double') 
        self.goal, self.goal_nd = self.separate_delimiter(goal) 
        self._input, self._input_nd = self.separate_delimiter(_input)

    def separate_delimiter(self, _input):
        _input_tmp = [x for x in _input.split()]
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
        return output, output_no_separator
        
    def errorRate(self):
        count1 = 0
        count2 = 0
        set_list = [self.output_nd, self.goal_nd, self._input_nd]
        offset = [0 ,0, 0]
        total = len(self.goal)
        total_char = 0
        output_char = 0
        input_char = 0
        for index in xrange(len(self.goal_nd)):
            if index + offset[0] >= len(set_list[0]) or index + offset[2] >= len(set_list[2]):
                break
            for i in [0, 2]:
                ## 错位两位仍可纠正
                if set_list[i][index + offset[i]] == set_list[1][index]:
                    break
                for j in xrange(1, 3):
                    print i,j
                    print index+j , total, index+ offset[i], len(set_list[i])
                    if index + j < total and index+ offset[i]< len(set_list[i]) and set_list[i][index + offset[i]] == set_list[1][index + j]:
                        offset[i] -= j
                        break
                
                    if index - j >= 0 and index + offset[i] < len(set_list[i]) and set_list[i][index + offset[i]] == set_list[1][index - j]:
                        offset[i] += j
                        break
            if index + offset[0] >= len(set_list[0]) or index + offset[2] >= len(set_list[2]):
                break
            x = set_list[0][ index + offset[0] ]
            y = set_list[1][ index + offset[1] ]
            z = set_list[2][ index + offset[2] ]
            print x, y, z
            if x == y:
                count1 = count1 + 1
            if y == z:
                count2 = count2 + 1
            for i,j,k in zip(x, y, z):
                if i == j:
                    output_char += 1
                if j == k:
                    input_char += 1
                total_char += 1
        set_list_string = [ ' '.join(l) for l in set_list]
        self.word_error = 1 - 1.0 * count1 / total
        self.ori_word_error = 1 - 1.0 * count2 / total
        self.char_error = 1 - 1. * output_char / total_char
        self.ori_char_error = 1 - 1. * input_char / total_char
        print "Origin correct %d words, now correct %d words,total %d words"%(count2, count1, total)
        print "Origin correct %d characters, now correct %d characters,total %d characters"%(input_char, output_char, total_char)
        with open(output_file, 'w') as fd:
            fd.write('Origin word error:%.8lf \n Current word error: %.8lf\n'%(self.ori_word_error, self.word_error))
            ' '.join( spellchecker.output )
            fd.write('%s\n%s\n%s'%(' '.join( self.goal ), ' '.join( self._input ),' '.join( self.output )))
        print '%s generated .'%output_file 

    def correctSentence(self):
        print "Start spell checking...\n"
        #self.output = [ correct(x) for x in  self._input ]
        print "Finish spell checking...\n"
    
    def naiveCorrect(self):
        output = []
        output_nd = []
        for x in self._input:
            if len(x) == 0:
                output.append('')
            elif len(x) == 1 and x in self.separators:
                output.append(x)
            else:
                y_list = self.most_simlilar_words(x)
                print '%s -> %s'%(x, y_list[0][0])
                output.append(y_list[0][0])
                output_nd.append(y_list[0][0])
        return output, output_nd
    def bigram(self):
        output = []
        output_nd = []
        
    def most_simlilar_words(self, word):
        prob_list = {}
        with open("%s/%d.txt"%(word_dict_path, len(word)), 'r') as fd:
            for line in fd:
                prod = self.probaWord(line.strip().lower(), word)
                if prod == -1:
                    continue
                prob_list[line.strip().lower()] = prod 
        sorted_prob = sorted(prob_list.items(), key = operator.itemgetter(1), reverse=True)
        return sorted_prob[:self.M]

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
    spellchecker.naiveCorrect()
    print ' '.join(spellchecker.goal)
    print ' '.join(spellchecker.output)
    #spellchecker.output_nd = spellchecker._input_nd
    spellchecker.errorRate()
