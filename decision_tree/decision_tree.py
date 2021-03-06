from collections import Counter
from collections import defaultdict

from scipy import stats

import math

class Node(object):
    def __init__(self, children):
        self.children = children

class Internal(Node):
    def __init__(self, attrfn, children):
        self.predicate = attrfn
        super().__init__(children)

class Leaf(Node):
    def __init__(self, label):
        self.class_label = label
        super().__init__(None)



def build_tree(samples, attrfns, classfn, classes, split):
    """Build a decision tree
       Parameters:
       samples   -- list of samples, where each sample is a list of attributes
       attrfns   -- list of attribute functions. Each of these functions can be applied to a
                    a sample to get a specific attribute.
                    e.g., [day_attr, color_attr],  day_attr(sample) => "Monday"
                    or color_attr(sample) => "Red"
       classfn   -- function that takes a single sample and returns the class for that sample
                    e.g., classfn(sample) => 'X'

       classes   -- tuple of classes
       split     -- function that takes a list of samples, a list of attribute functions, and
                    a classfn function. The function splits the data based on the best attribute,
                    and returns a tuple:
                    1) dict of data split, where key is the attribute value, and the value
                       is a list of examples that have that attribute value
                    2) the attribute function used for this split

                    Example:

                    samples = [['Monday', 'Yellow', 'X'], ['Monday', 'Red', 'O']]
                    def day(s): return s[0]
                    def color(s): return s[1]
                    def classfn(s): return s[2]


                    split(samples, [day, color], classfn)  (may return) =>

                    {'Yellow' => ['Monday', 'Yellow', 'X']
                     'Red'    => ['Monday', 'Red', 'O']} ,
                     color

                     if this function determines there is no best way to split these samples,
                     it will return None, None

           """
    if all(classfn(samples[0]) == classfn(sample) for sample in samples):
        return Leaf(classfn(samples[0]))

    if not attrfns:
        return Leaf(Counter(classfn(sample) for sample in samples).most_common()[0][0])

    splits, attrfn = split(samples, attrfns, classfn, classes)
    if not splits:
        return Leaf(Counter(classfn(sample) for sample in samples).most_common()[0][0])

    remaining_attrfns = [fn for fn in attrfns if fn != attrfn]

    child_nodes = dict()
    for key, group in splits.items():
        child_nodes[key] = build_tree(group, remaining_attrfns, classfn, classes, split)
    return Internal(attrfn, child_nodes)



def entropy(examples):
    '''
    Computes entropy of samples

    The min entropy is 0.0, the max entropy is log2(n), where n is number of classes.

    Parameters
    examples -- list of number of examples per class
    '''
    total = sum(examples)
    entropy = 0.0

    # filter out classes with 0 examples to compute - p * log(p)
    # (i.e., we define 0 * log(0) == 0)
    for n in filter(None, examples):
        entropy -= (n/total) * math.log(n/total , 2)
    return entropy

def attrvalues(examples, attrfn):
        return set(attrfn(example) for example in examples)

def counts_per_class(examples, classfn, classes):
    '''
     Returns the distribution of classes for a subset of examples.
       e.g., classfn(examples) -> [3, 4, 5]  (3 are of class 0, 4 of class 1, 5 of class 2)
    '''
    dist = dict()
    for cls in classes:
        dist[cls] = 0

    for example in examples:
        cls = classfn(example)
        dist[cls] += 1

    flat_dist = []
    for cls in classes:
        flat_dist.append(dist[cls])
    return flat_dist

def group_by_fn(samples, fn):
    vals = attrvalues(samples, fn)
    groups = dict()

    for val in vals:
        groups[val] = []

    for x in samples:
        val = fn(x)
        groups[val].append(x)

    return groups


def gain(examples, classfn, classes, attrfn):
    '''
    Calculates information gain after splitting on an attribute

    Parameters
    examples - list of examples. Each example has attributes.
    class_fn - function that returns the class of an example.
    classes  - list of all classses
    attrfn   - function that returns the value of a specific attribute given an example.
       e.g., attrnfn(example) -> attribute value
    attrvals - list of all possible values for the attribute used for this split
       e.g., [1, 2, 3], or ['red', 'yellow', 'green']

    '''

    en = entropy(counts_per_class(examples, classfn, classes))
    total = len(examples)
    attribute_values =  attrvalues(examples, attrfn)
    for val in attribute_values:
        # Get all examples whose value for the attribute is val
        sv = list(filter(lambda example: attrfn(example) == val, examples))
        en -= len(sv)/total * entropy(counts_per_class(sv, classfn, classes))
    return en


def chi_sqrd(groups):
    '''Computes chi-squared statistic
       parameters:
       groups - list of tuples, where each tuple x,y  represents counts per class
    '''
    row_totals = []
    col_totals = [0.0, 0.0]

    for group in groups:
        col_totals[0] = col_totals[0] + group[0]
        col_totals[1] = col_totals[1] + group[1]
        row_totals.append(sum(group))

    grid_total = sum(col_totals)

    expected_values = []
    for i, group in enumerate(groups):
        a = (row_totals[i] * col_totals[0]) / grid_total
        b = (row_totals[i] * col_totals[1]) / grid_total
        expected_values.append((a, b))

    chi_sqrd_stat = 0.0
    for observed, expected in zip(groups, expected_values):
        chi_sqrd_stat += (observed[0] - expected[0]) * (observed[0] - expected[0]) / expected[0]
        chi_sqrd_stat += (observed[1] - expected[1]) * (observed[1] - expected[1]) / expected[1]

    return chi_sqrd_stat


def chi_sqrd_from_groups(classes, classfn, groups):
    ''' computes chi squared statistic after splitting using attribute function attrnf

        parameters:
        classes - tuple of two classes
        classfn - function that takes a single sample and returns the class of that sample
        groups  - sequence of groups, where each group is a list of samples ''' # For each group, compute the number of samples per class
    group_counts = []
    for group in groups:
        counts = [0, 0]
        for sample in group:
            if classfn(sample) == classes[0]:
                counts[0] += 1
            else:
                counts[1] += 1
        group_counts.append(tuple(counts))
    return chi_sqrd(group_counts)


def chi_split(samples, attrfns, classfn, classes):
    # Compute information gain for every attfn used
    gains = []
    for attrfn in attrfns:
        g = gain(samples, classfn, classes, attrfn)
        gains.append((g, attrfn))

    splits = sorted(gains, key=lambda x: x[0])
    g, fn, p, best_groups = None, None, None, None
    for gn, f in splits:
        groups = group_by_fn(samples, f)
        chi = chi_sqrd_from_groups(classes, classfn, groups.values())
        p_value = stats.chi2.pdf(chi , len(groups) - 1)
        if p_value <= 0.01:
            g, fn, p = gn, f, p_value
            best_groups = groups
    return best_groups, fn

def classify(root, sample):
    node = root
    while not isinstance(node, Leaf):
        key = node.predicate(sample)

        if key in node.children:
            node = node.children[key]
        else:
            # handle unknown attribute keys.
            # Currently doing 2a.
            #
            # Some strategies to consider:
            #  1) assign most common label at this split?
            #  2) send sample down each child node, collect results
            #     a) assing most popular class label
            #     b) respect stats of data and teach this function how to handle probabilities.
            labels = defaultdict(int)
            for k in node.children.keys():
                label = classify(node.children[k], sample)
                labels[label] += 1
            return max(labels.items(), key=lambda x: x[1])[0]

    return node.class_label

def accuracy(root, test_data, classfn):
    correct, wrong = 0, 0
    for sample in test_data:
        if classfn(sample) == classify(root, sample):
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)

def count_nodes(node):
    if not node:
        return 0

    count = 0
    if node.children:
        for child in node.children.values():
            count += count_nodes(child)
    return count + 1

def depth(node):
    if not node:
        return 0

    depths = [0]
    if node.children:
        for child in node.children.values():
            depths.append(depth(child))
    return max(depths) + 1

