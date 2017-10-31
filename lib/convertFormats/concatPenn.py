import os
import glob

def load_conll(file_path):
    if os.path.isfile(file_path):
        loadfiles = [file_path]
    else:
        loadfiles = glob.glob(file_path)

    features = []
    for loadfile in loadfiles:
        with open(loadfile, "r") as f:
            for line in f:
                line = line.split("\t")
                if len(line) <= 1:
                    features.append("\n")
                    continue
                features.append(line[1])
    return features

if __name__ == '__main__':
    lex = load_conll("../auto/Penn_conbined_wsj/*/*.mrg")
    with open("../auto/concatFull.txt", "w") as f:
        f.write(" ".join(lex))
