from decision_tree import accuracy
from decision_tree import build_tree
from decision_tree import chi_split
from decision_tree import depth
from decision_tree import count_nodes
from decision_tree import gain
from decision_tree import group_by_fn

from scipy import stats

import pandas as pd

income_train = pd.read_csv('../dataset/adult.data', header=None).values.tolist()
income_test = pd.read_csv('../dataset/adult.test', header=None).values.tolist()


# categorial data
def workclass(x) : return x[1]
def edu(x) : return x[3]
def marital_status(x) : return x[4]
def occupation(x) : return x[5]
def relationship(x) : return x[6]
def race(x) : return x[7]
def sex(x) : return x[8]
def native_country(x) : return x[12]

def income(x): return x[14]

attrfns = [workclass, edu, marital_status, occupation, relationship,
        race, sex, native_country]


root = build_tree(income_train, attrfns, income, (' <=50K', ' >50K'), chi_split)
print("Created decision tree with {0} nodes, depth {1}".format(count_nodes(root), depth(root)))
print(accuracy(root, income_test, income))

