from googlecorrecter import correct 
import re
import pdb

input_file = '../data/svm_output.txt'
output_file = '../data/svm_spell_check.txt'
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
        self.goal = [ x for x in goal.split()]
        self._input = [ x for x in _input.split() ]
        self.output = []
        self.word_error = 0.0
        self.ori_word_error = 0.0
        self.ori_char_error = 0.0
        self.char_error = 0.0

    def errorRate(self):
        count1 = 0
        count2 = 0
        for x, y, z in zip(self.output, self.goal, self._input):
            if x == y:
                count1 = count1 + 1
            if y == z:
                count2 = count2 + 1
        total = len(self.goal)
        self.word_error = 1 - 1.0 * count1 / total
        self.ori_word_error = 1 - 1.0 * count2 / total
        with open(output_file, 'w') as fd:
            fd.write('Origin word error:%.8lf \n Current word error: %.8lf\n'%(self.ori_word_error, self.word_error))
            ' '.join( spellchecker.output )
            fd.write('%s\n%s\n%s'%(' '.join( self.goal ), ' '.join( self._input ),' '.join( self.output )))
        print '%s generated .'%output_file 

    def correctSentence(self):
        print "Start spell checking...\n"
        self.output = [ correct(x) for x in  self._input ]
        print "Finish spell checking...\n"

if __name__ == '__main__':
    ori_error_rate, correct_label , predict_label = loadFile(input_file)
    spellchecker = SpellChecker(goal = correct_label, _input = predict_label)
    spellchecker.correctSentence()
    spellchecker.errorRate()
