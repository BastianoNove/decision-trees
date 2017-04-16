from decision_tree import accuracy
from decision_tree import build_tree
from decision_tree import chi_split
from decision_tree import gain
from decision_tree import group_by_fn

from scipy import stats

import pandas as pd

cc_data = pd.read_csv('../dataset/crx.data', header=None)
shuffled_data = cc_data.sample(frac=1).reset_index(drop=True)

def cc_class(x): return x[15]
def zeroth(x): return x[0]
def third(x): return x[3]
def fourth(x): return x[4]
def fifth(x): return x[5]
def sixth(x): return x[6]
def eight(x): return x[8]
def ninth(x): return x[9]
def eleventh(x): return x[11]
def twelveth(x): return x[12]

cc_att_fns = [zeroth, third, fourth,fifth, sixth, eight,
              ninth, eleventh, twelveth]
x = shuffled_data[:414].values.tolist()

test_data = shuffled_data[414:].values.tolist()

root = build_tree(x, cc_att_fns, cc_class, ('+', '-'), chi_split)

print(accuracy(root, test_data, cc_class))

