from decision_tree import accuracy
from decision_tree import build_tree
from decision_tree import chi_sqrd_from_groups
from decision_tree import gain
from decision_tree import group_by_fn

from scipy import stats

import pandas as pd

# https://archive.ics.uci.edu/ml/datasets/Statlog+(Australian+Credit+Approval)
aust_data = pd.read_csv('../dataset/australian.dat', header=None, delimiter=' ')
shuffled_data = aust_data.sample(frac=1).reset_index(drop=True)

# categorial data
def zero(x): return x[0]
def three(x): return x[3]
def four(x): return x[4]
def five(x): return x[5]
def seven(x): return x[7]
def eight(x): return x[8]
def ten(x): return x[10]
def eleven(x): return x[11]

attrfns = [zero, three, four, five, seven, eight, ten, eleven]

def classfn(x): return int(x[14])

def cc_split_chi(samples, attrfns, classfn):
    # Compute information gain for every attfn used
    gains = []
    for attrfn in attrfns:
        g = gain(samples, classfn, [0, 1], attrfn)
        gains.append((g, attrfn))

    splits = sorted(gains, key=lambda x: x[0])
    g, fn, p, best_groups = None, None, None, None
    for gn, f in splits:
        groups = group_by_fn(samples, f)
        chi = chi_sqrd_from_groups( 0, 1, classfn, groups.values())
        p_value = stats.chi2.pdf(chi , len(groups) - 1)
        if p_value <= 0.01:
            g, fn, p = gn, f, p_value
            best_groups = groups
    return best_groups, fn

x = shuffled_data[:414].values.tolist()
test_data = shuffled_data[414:].values.tolist()
root = build_tree(x, cc_split_chi, attrfns, classfn)
print(accuracy(root, test_data, classfn))
