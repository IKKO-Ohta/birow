import glob
import os
import gensim
import numpy as np
import pandas as pd


class Oracle2Vec(object):
    def __init__(self):
        self.buffer = []
        self.stack = []
        self.action = []
        # self.wvmodel = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',
                                                                       binary=True)
        # with open("../auto/Penn_concat/gensim_corpora_dict_cF.pkl") as f:
            self.corpora = pickle.load(f)

    def _dummy(self, word_id):
        vec = [1 if i == word_id else 0 for i in range(len(self.corpora))]
        return np.ndarray(vec)

    def _word2vec(self, word):
        """
        かなりメモリを消費するのでノートブックではやらない
        """
        return self.wvmodel.wv[word]


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
            # datum = []
            # actions = []
            # for feat, l in zip(feature, label):
            #     if not feat:
            #         actions = []
            #     bufs, stus = oracle_split(feat)
            #     bufs = [self._embed_buffer(buf) for buf in bufs]
            #     stus = [self._embed_stack(stu) for stu in stus]
            #     datum.append([bufs, actions, stus, l])
            #     actions.append(l)
            # return datum

    def load_conll(file_path):
        if os.path.isfile(file_path):
            loadfiles = [file_path]
        else:
            loadfiles = glob.glob(file_path)
        features = []
        for loadfile in loadfiles:
            with open(loadfile, "r") as f:
                feature = [itemgetter(1, 3)(line.split("\t")) for line in f]
                feature.append(np.ndarray([0 for i in range(100)]))
                features.append(feature)
        return features

    def _embed_stack(self,stack):
        """
        部分木の計算方法は後で調べる
        線形変換Uとreluを再帰的に適用するのは知っている
        """
        return sum(stack)

    def _embed_buffer(self,buffer_w):
        """
        buffer =  "私"
        単語単位
        バッファは単に[w;wlm;t]を結合して返すだけ
        """
        w = self._dummy(self.corpora[buffer_w])
        w_lm = self._word2vec(buffer_w)
        t = np.ndarray([0 for i in range(10)])
        return np.concatenate(np.concatenate(w, w_lm), t)
