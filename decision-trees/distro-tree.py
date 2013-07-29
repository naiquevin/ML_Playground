"""Decision tree algorithm from the book "Machile learning in Action"

   This algorithm uses the ID3 algorithm

"""
from __future__ import division
import math
from collections import Counter
from pprint import pprint

import numpy as np
import pandas as pd
import random
import pydot

from tree import make_node, make_child_node, node_children, node_name, \
    child_node_node, child_node_edgeattrs


## A sample dataframe for writing tests
test_df = pd.DataFrame([[1, 1, 'yes'],
                       [1, 1, 'yes'],
                       [1, 0, 'no'],
                       [0, 1, 'no'],
                       [0, 1, 'no']])


def rm_feature(feature, df):
    """Removes feature/column from a dataframe

    :param feature : String
    :param df      : pd.DataFrame
    :rtype         : pd.DataFrame

    """
    newdf = df[:]
    del newdf[feature]
    return newdf


def shannon_entrophy(feature_column):
    """Calculate the shannon entropy of the column on values (one feature)

    :param feature_column : pd.Series
    :rtype                : Float

    """
    counts = Counter(feature_column)
    total = len(feature_column)
    probs = (v/total for k,v in counts.iteritems())
    e = sum(p * math.log(p, 2) for p in probs)
    return 0 - e


def test_shannon_entropy():
    np.testing.assert_approx_equal(shannon_entrophy(test_df[2]), 0.9709505944)


def split_dataset(df, feature, value):
    """Gets a subset of a dataframe with the feature equal to value

    :param df      : pd.DataFrame
    :param feature : String
    :param value   : Mixed
    :rtype         : pd.DataFrame

    """
    new_df = df[df[feature] == value]
    del new_df[feature]
    return new_df


def split_entropy(df, feature):
    """Total shannon entropy for every value of the feature 

    ie. Sum of the product of shannon entropies and the probability of
    the name values

    :param df      : pd.DataFrame
    :param feature : String
    :rtype         : Float

    """
    data_column = df[feature]
    def value_entropy(value):
        subset_df = split_dataset(df, feature, value)
        prob = len(subset_df) / len(df)
        return prob * shannon_entrophy(subset_df['name'])
    return sum(map(value_entropy, set(data_column)))


def choose_best_feature_to_split(df):
    """Choose the best feature to split the DataFrame on 

    It does this by comparing the information gain

    :param df : pd.DataFrame
    :rtype    : String

    """
    base_entropy = shannon_entrophy(df['name'])
    def feature_info_gain(feature):
        new_entropy = split_entropy(df, feature)
        return base_entropy - new_entropy
    features = (c for c in df.columns if c != 'name')
    return max(features, key=feature_info_gain)


def majority_count(classlist):
    """Produces the value from an iterable that appears maximum no. of
    times

    :param classlist : Iterable
    :rtype           : Mixed

    """
    return max(classlist, key=lambda c: classlist.count(c))


def tree(df):
    """Produces a tree from a dataframe

    Chooses the best feature to split the dataset Splits the dataset
    into multiple datasets Recursively generates children for the root
    node for each datasets

    :param df : pd.DataFrame
    :rtype    : tree.Node

    """
    classes = list(df['name'])
    # base cond 1: only one type in the target attribute return that type
    if classes.count(classes[0]) == len(classes):
        return make_node(list(df['name'])[0])
    # base cond 2: only the name is left, return by majority count
    if len(df.columns) == 1 and 'name' in df.columns:
        return make_node('/'.join(list(df['name'])))
    # find the best feature and create node add children to the current
    # node
    best_feature = choose_best_feature_to_split(df)
    unique_vals = set(df[best_feature])
    children = [make_child_node(tree(split_dataset(df, best_feature, v)), {'ans': v})
                for v in unique_vals]
    return make_node('[%d] What is %s?' % (random.randint(0, 1000), best_feature), 
                     children=children)


def plot(tree):
    """Plots the tree using pydot

    :param tree : tree.Node
    :rtype      : None

    """
    def inner(node, graph, parent_edge=None):
        name = node_name(node)
        graph_node = pydot.Node(name)
        graph.add_node(graph_node)
        if parent_edge is not None:
            parent, edge_label = parent_edge
            graph.add_edge(pydot.Edge(node_name(parent), name, label=edge_label))
        children = node_children(node)
        if len(children) > 0:
            for child in children:
                edge_label = child_node_edgeattrs(child).get('ans')
                child_node = child_node_node(child)
                inner(child_node, graph, parent_edge=(node, edge_label))
    graph = pydot.Dot(graph_type='graph')
    inner(tree, graph)
    graph.write_png('tree1.png')


def main():
    df = pd.read_csv('distros.csv')
    df['release_year'] = df['release_year'].map(int)
    df['age'] = df['release_year'].map(lambda x: 'new' if x > 2002 else 'old')
    df = rm_feature('release_year', df)
    t = tree(df)
    pprint(t)
    plot(t)


if __name__ == '__main__':
    main()
