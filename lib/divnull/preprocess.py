def _oracle_dump(oracle_path):
    """
    オラクルファイルを受け取って、学習可能な形式にして返す
    - [buffer状況,action状況,stack状況,ラベル]
    なるリスト
    1file-in,1datum-out
    """

    def str_eval(s):
        return s[1:len(s) - 1].split(", ")

    def oracle_split(lists_line):
        div = lists_line.find("][")
        stack = lists_line[:div + 1]
        stack = str_eval(stack)
        buffer = lists_line[div + 1:]
        buffer = str_eval(buffer)
        return [x for x in stack if x != " " or not x], [x for x in buffer if x != " " or not x]

    feature, label = [], []
    with open(oracle_path, "r") as f:
        """
        以下はoracle形式のつじつま合わせ
        """
        f.readline()
        cnt = 0
        for line in f:
            if "][" in line:
                line = line.rstrip()
                line = oracle_split(line)
                feature.append(line)
                cnt += 1
            elif line == "\n":
                """
                EOS
                """
                break
            else:
                line = line.rstrip()
                label.append(line)
                cnt += 1
    return feature, label

if __name__ == '__main__':
    path = "../auto/Penn_Oracle/00/wsj_0009.oracle"
    words, actions = _oracle_dump(path)
    result = []
    for i in range(len(words)):
        if words[i][0] == ["ROOT-ROOT"]:
            s, a, b = words[i][0], "END", words[i][1]
            result.append([s, a, b])
            continue

        s, a, b = words[i][0], actions[i], words[i][1]
        result.append([s, a, b])