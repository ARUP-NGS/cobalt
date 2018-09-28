
import pytest
import numpy as np
from cobaltcnv.hmm import HMM, HMMError, forward_likelihood_vector

class SimpleDiscreteDist(object):
    """ Helper for testing very simple cases"""

    def __init__(self, probs):
        self.probs = np.array(probs)
        if abs(sum(probs)-1.0) > 1e6:
            raise ValueError("Sum of probablities is not 1.0 (found {})".format(sum(probs)))

    def pmf(self, x, exposure=1, site=-1):
        if site==-1:
            return np.array([self.probs[v] if v>=0 and v<len(self.probs) else 0.0 for v in x])
        else:
            if type(x)==list or type(x)==np.ndarray:
                x = x[site]
            return self.probs[x] if x>=0 and x<len(self.probs) else 0.0

    def logpmf(self, x, exposure=1, site=-1):
        return np.log(self.pmf(x, site=site))


def almost_equal(a,b, tol=1e-6):
    return abs(a-b) < tol

def test_bad_t_matrix():
    m = np.matrix([[0.333, 0.666], [0.666, 0.333]])
    with pytest.raises(HMMError):
        hmm = HMM(m, [0, 0], [1.0, 0.0])


def test_bad_state_len():
    m = np.matrix([[0.4, 0.6], [0.6, 0.4]])
    with pytest.raises(HMMError):
        hmm = HMM(m, [0, 0], [1.0, 0.0, 0.0])



def test_happy_init():
    m = np.matrix([[0.4, 0.6], [0.6, 0.4]])
    try:
        hmm = HMM(m, [0, 0], [1.0, 0.0])
    except HMMError:
        assert False, "Unexpected error on initialization"

def test_simple_forward():
    m = np.matrix([[0.4, 0.6], [0.6, 0.4]])
    inits = [0.3, 0.7]
    edists = [
        SimpleDiscreteDist([0.1, 0.9]),
        SimpleDiscreteDist([0.8, 0.2]),
    ]

    hmm = HMM(m, edists, inits)
    l = hmm._forward([0, 1, 1])
    EXPECTED_RESULT = -2.063668985 # Painstakingly calculated entirely by hand...
    assert almost_equal(l, EXPECTED_RESULT)

def test_bigger_forward():
    alpha = 0.1
    m = np.matrix([[1.0-alpha, alpha, 0.0], [alpha, 1.0-2*alpha, alpha], [0.0, alpha, 1.0-alpha]])
    inits = [0.3, 0.7, 0.0]
    edists = [
        SimpleDiscreteDist([0.1, 0.9]),
        SimpleDiscreteDist([0.4, 0.4, 0.1, 0.1]),
        SimpleDiscreteDist([0.1, 0.2, 0.5, 0.1, 0.1]),
    ]

    hmm = HMM(m, edists, inits)
    obs = [0, 1, 1, 2, 2, 1, 0, 0, 1, 1, 4, 3, 4]
    l = hmm._forward(obs)
    dbg = hmm._forward_dbg(obs)
    assert almost_equal(l, dbg)
    EXPECTED_RESULT = -21.7826845758  # Calculated by simple debugging forward algo
    assert almost_equal(l, EXPECTED_RESULT)

def test_vectorized_forward():
    """
    Simple test of vectorization code, this is the same data as test_bigger_forward, but uses vectorized
    hmm code
    """
    alpha = 0.1
    m = np.matrix([[1.0 - alpha, alpha, 0.0], [alpha, 1.0 - 2 * alpha, alpha], [0.0, alpha, 1.0 - alpha]])
    inits = [0.3, 0.7, 0.0]
    edists = [
        SimpleDiscreteDist([0.1, 0.9]),
        SimpleDiscreteDist([0.4, 0.4, 0.1, 0.1]),
        SimpleDiscreteDist([0.1, 0.2, 0.5, 0.1, 0.1]),
    ]

    hmm = HMM(m, edists, inits)
    obs = [0, 1, 1, 2, 2, 1, 0, 0, 1, 1, 4, 3, 4]
    exposure = 1.0
    l = forward_likelihood_vector(obs, exposure, initial=hmm.initial, transitions=hmm.tr, emissions=hmm.em)
    dbg = hmm._forward_dbg(obs)
    assert almost_equal(l, dbg)
    EXPECTED_RESULT = -21.7826845758  # Calculated by simple debugging forward algo
    assert almost_equal(l, EXPECTED_RESULT)


def test_viterbi_small():
    """ This example is taken directly from wikipedia as of 10/15/2016 """
    m = np.matrix([[0.7, 0.3],
                   [0.4, 0.6]])
    inits = [0.6, 0.4]
    edists = [
        SimpleDiscreteDist([0.5, 0.4, 0.1]),
        SimpleDiscreteDist([0.1, 0.3, 0.6]),
    ]

    hmm = HMM(m, edists, inits)
    path, prob = hmm.viterbi([0, 1, 2], exposure=1.0)
    assert [0, 0, 1]== path
    assert almost_equal(np.log(0.01512), prob)

def test_viterbi_small2():
    """ This example is taken directly from http://homepages.ulb.ac.be/~dgonze/TEACHING/viterbi.pdf as of 10/15/2016 """
    m = np.matrix([[0.5, 0.5],
                   [0.4, 0.6]])
    inits = [0.5, 0.5]
    edists = [
        SimpleDiscreteDist([0.2, 0.3, 0.3, 0.2]),
        SimpleDiscreteDist([0.3, 0.2, 0.2, 0.3]),
    ]

    hmm = HMM(m, edists, inits)
    path, prob = hmm.viterbi([2, 2, 1, 0, 1, 3, 2, 0, 0], exposure=1.0)
    assert [0, 0, 0, 1, 1, 1, 1, 1, 1] == path
    assert almost_equal(-16.9734022962, prob)

def test_forward_backward():
    """
    Test forward-backward algorithm. This example taken from wikipedia: https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
    """
    m = np.matrix([[0.7, 0.3], [0.3, 0.7]])
    inits = [0.5, 0.5]
    edists = [
        SimpleDiscreteDist([0.9, 0.1]),
        SimpleDiscreteDist([0.2, 0.8]),
    ]

    hmm = HMM(m, edists, inits)
    obs = np.array([0, 0, 1, 0, 0])
    l = hmm.forward_backward(obs)
    assert almost_equal(l[0][0], 0.6469, tol=1e-4)
    assert almost_equal(l[0][1], 0.3531, tol=1e-4)
    assert almost_equal(l[1][0], 0.8673, tol=1e-4)
    assert almost_equal(l[1][1], 0.1327, tol=1e-4)
    assert almost_equal(l[2][0], 0.8204, tol=1e-4)
    assert almost_equal(l[2][1], 0.1796, tol=1e-4)
    assert almost_equal(l[3][0], 0.3075, tol=1e-4)
    assert almost_equal(l[3][1], 0.6925, tol=1e-4)
    assert almost_equal(l[4][0], 0.8204, tol=1e-4)
    assert almost_equal(l[4][1], 0.1796, tol=1e-4)


