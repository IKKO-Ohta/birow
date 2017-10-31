import glob
import pickle
import sys
import math
import pandas as pd
import numpy as np
import nltk
from nltk import corpus
from nltk.parse import DependencyGraph, DependencyEvaluator
from nltk.parse.transitionparser import TransitionParser, Configuration, Transition
from nltk.corpus import conll2000, conll2002

# nltk arc eager
# 学習データは事前にsplit処理済み
dataset = glob.glob("../auto/Penn_conbined_wsj/*/*.mrg")
div = int(len(dataset) * 0.9)
trains, tests = dataset[:div],dataset[div:]

# mini size
graphs = []
parser = nltk.TransitionParser(algorithm="arc-eager")
for train in trains[:700]:
    with open(train,"r") as f:
        graphs.append(DependencyGraph(f.read()))
print("graph loaded")
parser.train(graphs,"../model/mini_arc_eager.md")
with open("../model/mini_arc_eager.parser","wb") as f:
    pickle.dump(parser,f)

# middle
graphs = []
parser = nltk.TransitionParser(algorithm="arc-eager")
for train in trains[:1400]:
    with open(train,"r") as f:
        graphs.append(DependencyGraph(f.read()))
print("graph loaded")
parser.train(graphs,"../model/middle_arc_eager.md")
with open("../model/middle_arc_eager.parser","wb") as f:
    pickle.dump(parser,f)

"""
# full
graphs = []
parser = nltk.TransitionParser(algorithm="arc-eager")
for train in trains:
    with open(train,"r") as f:
        graphs.append(DependencyGraph(f.read()))
print("graph loaded")
parser.train(graphs,"../model/full_arc_eager.md")
with open("../model/full_arc_eager.parser","wb") as f:
    pickle.dump(parser,f)


test_graphs = []
for test in tests:
    with open(test) as f:
        test_graphs.append(DependencyGraph(f.read()))
result = parser.parse(test_graphs,"../model/full_arc_eager.md")
print(result)
"""
