import copy
import re
import glob
import gensim
import pickle
import numpy as np


class Configuration(object):
    def __init__(self):
        self.stack = []
        self.buffer = []
        self.arcs = []
        self.history = []

    def show(self):
        print("stack: ", self.stack)
        print("buffer: ", self.buffer)
        print("arcs: ", self.arcs)


class Transition(object):
    def __init__(self):
        return

    @staticmethod
    def right_arc(relation, conf):
        conf.arcs.append([conf.stack[1], relation, conf.stack[0]])
        # print([conf.stack[1], relation, conf.stack[0]])
        conf.stack.pop(0)

    @staticmethod
    def left_arc(relation, conf):
        conf.arcs.append([conf.stack[0], relation, conf.stack[1]])
        # print([conf.stack[0], relation, conf.stack[1]])
        conf.stack.pop(1)

    @staticmethod
    def shift(conf):
        idx_wi = conf.buffer.pop(0)
        conf.stack.insert(0, idx_wi)
        if not conf.stack[-1]: del conf.stack[-1]


class myVectorizer(object):
    """
    - cal buffer
    - cal stack
    - 変数のembed
    - ダミー化
    """

    def __init__(self):
        self.regex = re.compile('[a-zA-Z0-9]+')
        self.wv_model = gensim.models.KeyedVectors.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin',
                                                                        binary=True)
        with open("../model/tag_map.pkl", "rb") as f:
            self.tag_map = pickle.load(f)
        with open("../model/word2id.pkl", "rb") as f:
            self.corpus = pickle.load(f)
        with open("../model/act_map.pkl", "rb") as f:
            self.act_map = pickle.load(f)
        with open("../model/tag2id.pkl", "rb") as f:
            self.tag2id = pickle.load(f)

    def reg(self, word):
        g = self.regex.match(word)
        if g:
            return g.group()
        else:
            return ""

    def find_tag(self, word):
        word = self.regex.match(word)
        return self.tag_map[word]

    @staticmethod
    def dummy(ind, l):
        """
        chainerのembeddingへ委譲
        """
        m = [0 if i != ind else 1 for i in range(l)]
        return np.asarray(m, dtype=np.float32)

    def buf_embed(self, buffer):
        if not buffer:
            w = -1
            wlm = np.asarray([0 for i in range(300)], dtype=np.float32)
            tag = -1
            return [w,wlm,tag]

        word = buffer[-1]  # last

        def find(key):
            return self.corpus[key], self.wv_model[key], self.tag_map[key]

        def e_find(key):
            return self.corpus[key], np.asarray([0 for i in range(300)]), self.tag_map[key]

        def not_null(key):
            dicts = [self.corpus, self.wv_model, self.tag_map]
            if all([key in d for d in dicts]):
                return True
            else:
                return False

        # 正規表現でwordを洗浄
        word = self.reg(word)

        if not_null(word):
            w, wlm, tag = find(word)
        elif not_null(word.capitalize()):
            w, wlm, tag = find(word.capitalize())
        else:
            w, wlm, tag = e_find(word)

        #w = self.dummy(w, len(self.corpus))
        tag = self.tag2id[tag]
        return [w, wlm, tag]

    def edge_embed(self, arc):
        if not arc:
            h = np.asarray([0 for i in range(300)], dtype=np.float32)
            d = np.asarray([0 for i in range(300)], dtype=np.float32)
            r = -1
            return [h,d,r]

        edge = arc[-1]  # last arc
        edge[0], edge[2] = self.reg(edge[0]), self.reg(edge[2])  # 正規表現でwordを洗浄
        if edge[0] in self.wv_model:
            h = self.wv_model[edge[0]]
        elif edge[0].capitalize() in self.wv_model:
            h = self.wv_model[edge[0].capitalize()]
        else:
            h = np.asarray([0 for i in range(300)], dtype=np.float32)

        if edge[2] in self.wv_model:
            d = self.wv_model[edge[2]]
        elif edge[2].capitalize() in self.wv_model:
            d = self.wv_model[edge[2].capitalize()]
        else:
            d = np.asarray([0 for i in range(300)])

        r = self.act_map[edge[1]]
        # r = self.dummy(r, len(self.act_map))
        return [h, d, r]

    def cal_history(self, history):
        """
        スタティックメソッドのdummyを用いる
        :param バッファの最新ヒストリ
        :return: ダミー化されたヒストリ
        """
        if history:
            return self.act_map[history[-1]]
        else:
            return 0


def oracle_load(oracle_path):
    """
    オラクルファイルを受け取って、confに代入できるよう整形する
    featureとラベルが返る
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
    pathes = glob.glob("../auto/Penn_Oracle_split/*/*.oracle")
    vectorizer = myVectorizer()
    error = 0

    for path in pathes:
        words, actions = oracle_load(path)
        conf = Configuration()
        conf.stack = copy.deepcopy(words[0][0])
        conf.buffer = copy.deepcopy(words[0][1])
        cnt = 0
        try:
            for action in actions:
                # print(conf.show())
                # print("dumping..")
                his = vectorizer.cal_history(conf.history)
                buf = vectorizer.buf_embed(conf.buffer)
                stk = vectorizer.edge_embed(conf.arcs)
                #print("his:", his,"buf:", buf, "stk", stk)
                # print("label:", action)

                if action == "SHIFT":
                    Transition.shift(conf)
                elif "RIGHT" in action:
                    Transition.right_arc(action, conf)
                elif "LEFT" in action:
                    Transition.left_arc(action, conf)
                else:
                    """
                    予期しない命令
                    """
                    break
                target_dir_num = path.split("/")[-2]
                origin_name = path.split("/")[-1].replace(".oracle", "")
                target_path = "../auto/preprocessed/" + target_dir_num \
                              + "/" + origin_name + "_" + '{0:07d}'.format(cnt) + ".pkl"

                with open(target_path, "wb") as target:
                    print("gen:", target_path)
                    label = vectorizer.act_map[action]
                    pickle.dump(([his, buf, stk], label), target)

                conf.history.append(action)
                cnt += 1

        except IndexError:
            print("--ERROR-- ", path)
            print("IndexError")
            print("continue..")
            error += 1

    print("preprocess done. found Error ..")
    print(error)
