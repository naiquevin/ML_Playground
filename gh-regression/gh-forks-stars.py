"""Linear regression using gradient descent on github stars~forks

Usage: gh-forks-stars.py scatterplot [ -f FILE -o OUTPUT ]
       gh-forks-stars.py linregress [ -f FILE -o OUTPUT -a ALPHA -n NUMITERS ]
       gh-forks-stars.py predict STARS [ -f FILE -a ALPHA -n NUMITERS ]
       gh-forks-stars.py ( -h | --help )

Options:
    -f FILE       Name of JSON file to read the input data from [default: ./gh-repos.json]
    -o OUTPUT     Name of file to plot the output to
    -a ALPHA      The learning Rate for linear regression [default: 0.0001]
    -n NUMITERS   No. of times to repeat gradient descent [default: 1500]
    -h --help     Show help

"""

import json
import os
import math

from lookupy import Collection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from docopt import docopt


def get_data(fp):
    with open(fp) as f:
        repos = json.load(f)
    return Collection(repos).filter(fork=False) \
                            .filter(followers__lte=50) \
                            .filter(forks__lte=10) \
                            .select('name', 'followers', 'forks')


## exploring the data

def scatterplot(df, output, lin=None):
    ax = plt.figure().gca()
    plt.scatter(df['followers'].values,
                df['forks'].values)
    if lin is not None:
        df, coeffs = lin
        intercept, slope = coeffs
        x = df['followers']
        miny = intercept + x.min() * slope
        maxy = intercept + x.max() * slope
        ax.add_line(lines.Line2D([x.min(), x.max()],
                                 [miny, maxy],
                                 color='r', linewidth=2))
    plt.savefig(output)


## gradient descent

TEST_DATA = np.array([ [2536, 692],
                       [915, 388],
                       [1028, 201],
                       [1585, 481],
                       [1098, 125] ])


def sep_xy(data):
    return (data[:, 0], data[:, 1])


def input_matrix(xs):
    xm = np.ones(shape=(xs.size, 2))
    xm[:, 1] = xs
    return xm


def hypothesis(thetas, xm):
    return xm.dot(thetas.T).flatten()


def test_hypothesis():
    xs, ys = sep_xy(TEST_DATA)
    thetas = np.array([ -1, 2.3 ])
    xm = input_matrix(xs)
    pred = hypothesis(thetas, xm)
    np.testing.assert_allclose(pred, [5831.8,  2103.5,  2363.4,  3644.5,  2524.4])


def cost(predicted, actual):
    return 0.5 * ((predicted - actual) ** 2).sum()


def test_cost():
    xs, ys = sep_xy(TEST_DATA)
    predicted = np.array([ 5831.8,  2103.5,  2363.4,  3644.5,  2524.4 ])
    np.testing.assert_approx_equal(cost(predicted, ys), 24900655.275)


def next_theta(theta, x, predicted, ys, alpha, m):
    return theta - ((alpha / m) * ((predicted - ys) * x).sum())


def test_next_thetas():
    xs, ys = sep_xy(TEST_DATA)
    thetas = np.array([ -1, 2.3 ])
    predicted = np.array([ 5831.8, 2103.5, 2363.4, 3644.5, 2524.4 ])
    xm = input_matrix(xs)
    alpha = 0.01
    theta1, theta2 = thetas
    nexttheta1 = next_theta(thetas[0], xm[:, 0], predicted, ys, alpha, ys.size)
    nexttheta2 = next_theta(thetas[1], xm[:, 1], predicted, ys, alpha, ys.size)
    np.testing.assert_approx_equal(nexttheta1, -30.1612)
    np.testing.assert_approx_equal(nexttheta2, -48949.4024)


def gradient_descent(data, thetas, times, alpha):
    xs, ys = sep_xy(data)
    xm = input_matrix(xs)
    m = ys.size
    costs = []
    while times > 0:
        predicted = hypothesis(thetas, xm)
        t1 = next_theta(thetas[0], xm[:, 0], predicted, ys, alpha, m)
        t2 = next_theta(thetas[1], xm[:, 1], predicted, ys, alpha, m)
        thetas = np.array([t1, t2])
        times -= 1
        costs.append(cost(predicted, ys))
    return (thetas, costs)


def predict(coeffs, x):
    intercept, slope = coeffs
    return x * slope + intercept
    

if __name__ == '__main__':
    args = docopt(__doc__)

    input_file = args['-f']
    slug = os.path.splitext(os.path.basename(input_file))[0]

    data = list(get_data(input_file))
    df = pd.DataFrame(data)

    if args['scatterplot']:
        output = args.get('-o') if args.get('-o') is not None else 'scatterplot-%s.png' % (slug,)
        scatterplot(df, output)
        exit(0)

    def gd(data):
        alpha = float(args.get('-a'))
        numiters = int(args.get('-n'))
        data = np.array([[d['followers'], d['forks']] for d in data])
        return gradient_descent(data, np.array([ 0, 0 ]), numiters, alpha)

    if args['linregress']:
        output = args.get('-o') if args.get('-o') is not None else 'linregress-%s.png' % (slug,)
        coeffs, costs = gd(data)
        scatterplot(df, output, (df, coeffs))
        print(coeffs)
        exit(0)

    if args['predict']:
        coeffs, costs = gd(data)
        print(math.floor(predict(coeffs, int(args['STARS']))))
        plt.plot(range(1, len(costs)+1), costs)
        plt.savefig('costs.png')
        exit(0)
        
