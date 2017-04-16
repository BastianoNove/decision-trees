from decision_tree import accuracy
from decision_tree import attrvalues
from decision_tree import build_tree
from decision_tree import chi_sqrd_from_groups
from decision_tree import gain

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

def group_by_fn(samples, fn):
    vals = attrvalues(samples, fn)
    groups = dict()

    for val in vals:
        groups[val] = []

    for x in samples:
        val = fn(x)
        groups[val].append(x)

    return groups

def cc_split_chi(samples, attrfns, classfn):
    # Compute information gain for every attfn used
    gains = []
    for attrfn in attrfns:
        g = gain(samples, classfn, ['+', '-'], attrfn, attrvalues(samples, attrfn))
        gains.append((g, attrfn))

    splits = sorted(gains, key=lambda x: x[0])
    g, fn, p, best_groups = None, None, None, None
    for gn, f in splits:
        groups = group_by_fn(samples, f)
        chi = chi_sqrd_from_groups( '+', '-', classfn, groups.values())
        p_value = stats.chi2.pdf(chi , len(groups) - 1)
        if p_value <= 0.01:
            g, fn, p = gn, f, p_value
            best_groups = groups
    return best_groups, fn

x = shuffled_data[:414].values.tolist()
test_data = shuffled_data[414:].values.tolist()

root = build_tree(x, cc_split_chi, cc_att_fns, cc_class)

print(accuracy(root, test_data, cc_class))

