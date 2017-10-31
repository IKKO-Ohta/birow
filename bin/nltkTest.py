import glob
import sys
import pickle
import math
import pandas as pd
import numpy as np
import nltk
from nltk import corpus
from nltk.parse import DependencyGraph, DependencyEvaluator
from nltk.parse.transitionparser import TransitionParser, Configuration, Transition
from nltk.corpus import conll2000, conll2002

# nltk arc eager 
dataset = glob.glob("../auto/Penn_conbined_wsj/*/*.mrg")
div = int(len(dataset) * 0.9)
trains,tests = dataset[:div],dataset[div:]

test_graphs = []
for test in tests:
    with open(test) as f:
        test_graphs.append(DependencyGraph(f.read()))
print("testfile loaded..")

parsers_name = ["full_arc_eager","middle_arc_eager","mini_arc_eager"]

res = []
for psr in parsers_name:
    with open("../model/"+psr+".parser","rb") as f: 
        parser = pickle.load(f)
    print("perser loaded..")
    result = parser.parse(test_graphs,"../model/"+psr+".md")
    de = DependencyEvaluator(result,test_graphs)
    res.append(de)
    print("parser:",psr,"DONE")
print(res)
