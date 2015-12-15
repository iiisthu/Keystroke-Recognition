import os

here = os.path.dirname(os.path.abspath(__file__))

input_file = "%s/../data/word_list.txt"%here
output_path = "%s/../data/word_by_len"%here


def split_by_len(input_file, output_path):
    word_dict = {}
    with open(input_file, 'r') as fd:
        for line in fd:
            line=line.strip()
            lenz = len(line)
            if lenz not in word_dict.keys():
                word_dict[lenz] = [line]
            else:
                word_dict[lenz].append(line)
    print word_dict
    for key in word_dict:
        with open("%s/%d.txt"%(output_path, key), 'w') as fd:
            for line in word_dict[key]:
               fd.write('%s\n'%line)
            


if __name__ == '__main__':
    split_by_len(input_file, output_path)
