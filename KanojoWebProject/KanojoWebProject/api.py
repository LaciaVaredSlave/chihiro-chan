import numpy as np
from flask import Flask, render_template, url_for, request, redirect, jsonify, make_response
from KanojoWebProject import app
import Model
import chainer
from chainer import cuda, Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
import chainer.functions as F
import chainer.links as L
from janome.tokenizer import Tokenizer
import json
 
model = None
 
def getResponseSentence(model, sentencies):
    model.H.reset_state()
    for i in range(len(sentencies)):
        if sentencies[i] in avocab:
            wid = avocab[sentencies[i]]
            x_k = model.embedx(Variable(np.array([wid], dtype=np.int32)))
            h = model.H(x_k)
    x_k = model.embedx(Variable(np.array([avocab['<eos>']], dtype=np.int32)))
    h = model.H(x_k)
    wid = np.argmax(F.softmax(model.W(h)).data[0])
    res = id2wd[wid]
    loop = 0
    while (wid != bvocab['<eos>']) and (loop <= 30):
        x_k = model.embedy(Variable(np.array([wid], dtype=np.int32)))
        h = model.H(x_k)
        wid = np.argmax(F.softmax(model.W(h)).data[0])
        if wid in id2wd:
            res += id2wd[wid] 
        loop += 1
    return res
 
t_wakati = Tokenizer(wakati=True)
 
alines, blines, avocab, av, bvocab, bv, id2wd = np.load("data.npy")
 
demb = 100
model = Model.ConversationModel(av, bv, avocab, bvocab, demb)
serializers.load_npz('manzai.model', model)
 
@app.route("/conversation", methods=['GET'])
def conversation():
    global model
 
    res = ''
    sent = request.args.get('q', '')
    if sent != '':
        sentencies = t_wakati.tokenize(sent)
        alnr = sentencies[::-1]
        res = getResponseSentence(model, sentencies)
 
    data = {'result':'success',
            'request_sentence':sent,
            'responce_sentence':res}
 
    return make_response(jsonify(data))
