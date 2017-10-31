import sys

def conll_split(file_path):
    # 分割                                                                                                                                                                         
    with open(file_path) as f:
        sentences,sentence = [],[]
        for line in f:
            if "\n" == line: # 文の終わり                                                                                                                                          
                sentences.append(sentence)
                sentence = []
            elif "#" in line:
                continue
            else:
                if line:
                    sentence.append(line)
    # 書き込み
    num = 0
    for sentence in sentences:
        with open("../auto/univ_dep_train/" + '{0:05d}'.format(num) + ".txt","w") as f:
            print("../auto/univ_dep_train/" + '{0:05d}'.format(num) + ".txt")
            for elem in sentence:
                f.write(elem)
        num += 1
    return

if __name__ == '__main__':
    file_path = sys.argv[1]
    conll_split(file_path)

