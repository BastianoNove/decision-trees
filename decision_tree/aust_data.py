from decision_tree import accuracy
from decision_tree import build_tree
from decision_tree import chi_split
from decision_tree import count_nodes
from decision_tree import depth
from decision_tree import gain
from decision_tree import group_by_fn


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

x = shuffled_data[:414].values.tolist()
test_data = shuffled_data[414:].values.tolist()
root = build_tree(x, attrfns, classfn, (0,1), chi_split)
print("Created decision tree with {0} nodes, depth {1}".format(count_nodes(root), depth(root)))
print(accuracy(root, test_data, classfn))
