## Data Definitions
## 
## Node (nodename, nodeattrs, children)
## where:
##   nodename is a String
##   nodeatts is a dict
##   children is a list of ChildNodes
## 
## ChildNode (node, edgeattrs)
## where:
##   edgeattrs is a dict
##   node is a Node


def make_node(name, attrs=None, children=None):
    attrs = {} if attrs is None else attrs
    children = [] if children is None else children
    return (name, attrs, children)

def node_name(node):
    return node[0]

def node_attrs(node):
    return node[1]

def node_children(node):
    return node[2]

def node_set_name(name, node):
    node[0] = name
    return node

def node_set_attrs(attrs, node):
    node[1] = attrs
    return node

def node_set_children(children, node):
    node[2] = children
    return node

def node_set_attr(key, value, node):
    attrs = node_attrs(node)
    attrs[key] = value
    newnode = node_set_attrs(attrs)
    return newnode

def node_append_child(childnode, node):
    children = node_children(node)
    newchildren = children + [childnode]
    newnode = node_set_children(newchildren)
    return newnode

def node_has_children(node):
    return len(node_children(node)) > 0


def make_child_node(node, edgeattrs=None):
    edgeattrs = {} if edgeattrs is None else edgeattrs
    return (node, edgeattrs)


def child_node_edgeattrs(node):
    return node[1]

def child_node_node(node):
    return node[0]



## examples

n1 = make_node('apple', {'type': 'fruit'})
nc1 = make_child_node(n1)
n2 = make_node('blue', {'type': 'color'}, [nc1])


## tests
def tests():
    assert n1 == ('apple', {'type': 'fruit'}, [])
    assert child_node_node(nc1) == n1
    assert child_node_edgeattrs(nc1) == {}

