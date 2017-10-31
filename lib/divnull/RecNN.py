import chainer
import chainer.functions as F
import chainer.links as L


class RedNN(chainer.Chain):
    def __init__(self):
        super(RedNN, self).__init__()
        with self.init_scope():
            self.l = L.Linear(50, 1000)

    def __call__(self, L):
        while(1):
            if len(L) == 1:
                return L[0]
            ret0, ret1 = L[-1], L[-2]
            L.pop()
            L.pop()
            ret = self.L(ret0, mytype(ret1), ret1)
            ret = F.tanh(ret)
            L.append(ret)
