import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
 
class ConversationModel(chainer.Chain):
    avocab = {}
    bvocab = {}
 
    def __init__(self, av, bv, avo, bvo, k):
        super(ConversationModel, self).__init__(
            embedx = L.EmbedID(av, k),
            embedy = L.EmbedID(bv, k),
            H = L.LSTM(k, k),
            W = L.Linear(k, bv),
        )
        self.avocab = avo
        self.bvocab = bvo
 
    def __call__(self, aline, bline):
        for i in range(len(aline)):
            wid = self.avocab[aline[i]]
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
            h = self.H(x_k)
        x_k = self.embedx(Variable(np.array([self.avocab['<eos>']], dtype=np.int32)))
        tx = Variable(np.array([self.bvocab[bline[0]]], dtype=np.int32))
        h = self.H(x_k)
        accum_loss = F.softmax_cross_entropy(self.W(h), tx)
        for i in range(len(bline)):
            wid = self.bvocab[bline[i]]
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32)))
            next_wid = self.bvocab['<eos>']  if (i == len(bline) - 1) else self.bvocab[bline[i+1]]
            tx = Variable(np.array([next_wid], dtype=np.int32))
            h = self.H(x_k)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss += loss
        return accum_loss
