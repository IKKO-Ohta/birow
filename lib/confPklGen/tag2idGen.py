import pickle

with open("../model/word2id.pkl", "br") as f:
    tag_map = pickle.load(f)

tagset = set([v for v in tag_map.values()])
tag2id = {v: i for i, v in zip(range(len(tagset)), tagset)}

with open("../model/tag2id.pkl", "wb") as f:
    pickle.dump(tag2id, f)
