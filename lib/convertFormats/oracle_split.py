import glob


def mywrite(elems, path, cnt):
    path = path.replace("Penn_Oracle", "Penn_Oracle_split")
    path = path.replace(".oracle", "_" + str(cnt) + ".oracle")
    with open(path, "w") as p:
        for elem in elems:
            p.write(elem)


def mysplit(path):
    cnt = 0
    result = []
    with open(path,"r") as f:
        for line in f:
            result.append(line)
            if line == "[ROOT-ROOT][]\n":
                mywrite(result, path, cnt)
                cnt += 1
                result = []
    return

if __name__ == '__main__':
    t = 0
    pathes = glob.glob("../auto/Penn_Oracle/*/*.oracle")
    for path in pathes:
        mysplit(path)
    print("done")
