
import numpy as np
import scipy.stats as stats
import math
import logging

class HMMError(Exception):
    """ Configuration / validation exceptions for HMM """
    pass

class ParameterViolation(Exception):
    """ Invalid parameter values cause these - usually we just return -inf for probability and continue """
    pass

class InvalidResultError(Exception):
    """ Raised when numerical issues arise during probability calculations """
    pass

class HMM(object):
    """
    A simple HMM, defined by a a transition matrix, list of emission distributions, and initial distribution
    This provides functions for computing the forward likelihood and the likelihood of a particular series
     of observations when the true state is known
    """

    def __init__(self, transitions, emissions, init_dist):
        self.tr = np.matrix(transitions)
        self.em = list(emissions)
        self.initial = init_dist
        self._validate()

    def _validate(self):
        matrix_dim = self.tr.shape
        if len(matrix_dim) != 2:
            raise HMMError("Transition matrix must have exactly two dimensions (found {})".format(len(matrix_dim)))
        if matrix_dim[0] != matrix_dim[1]:
            raise HMMError("Transition matrix must be square (found {} rows and {} cols)".format(matrix_dim[0], matrix_dim[1]))
        if len(self.em) != matrix_dim[0]:
            raise HMMError("Unequal number of states in transition vs emission (found {} vs {})".format(len(self.em), matrix_dim[0]))
        if len(self.initial) != matrix_dim[0]:
            raise HMMError("Unequal initial state dist length vs transitions(found {} vs {})".format(len(self.initial), matrix_dim[0]))
        # Make sure column and row sums are (pretty close) to 1
        for i, s in enumerate(self.tr.sum(axis=1).getA1()):
            if abs(s - 1.0) > 1e-9:
                raise HMMError("Sum of row {} is not equal to 1: {}".format(i, s))

    def emission_dist_moments(self, site=-1):
        """
        Return list of (mean, variance) tuples for each emission distribution
        """
        return [d.meanvar(site) for d in self.em]

    @property
    def emission_dists(self):
        return self.em

    @emission_dists.setter
    def emission_dists(self, new_dists):
        self.em = list(new_dists)
        self._validate()

    def update_params(self, params):
        """
        Set new parameters for the emission distributions
        """
        for dist in self.em:
            dist.update_params(params)

    def logL(self, obs, known_states=None):
        """
        Compute the likelihood of the set of observations
        :param obs: Either a list or matrix of observations, if list, simple forward likelihood is computed.  If matrix,
         then assume each row contains a list of observed values across multiple samples. Using this is a lot more
         efficient than computing each sample probability independently. This option is NOT compatible with the using
         known_states
        :param known_states: List of known / true states for each observation.
        :return: log-likelihood of generating the observations under this HMM
        """
        if known_states is None:
            # Compute likelihood of observations using forward algorithm
            return self._forward(obs)

        else:
            # Compute likelihood of observations using known states using trivial algorithm
            return self._path_likelihood(obs, known_states)


    def _path_likelihood(self, obs, states):
        """ Compute likelihood of a single series of observations with a given known sequence of states """
        return path_likelihood((obs, states), initial=self.initial, transitions=self.tr, emissions=self.em)


    @property
    def states(self):
        return len(self.em)

    def _forward(self, obs):
        """
        Compute the log-likelihood of the sequence of observations using the Forward algorithm
        :param obs: Series of observed values
        """
        return forward_likelihood(obs, self.initial, self.tr, self.em)

    @staticmethod
    def _row_count(obj):
        """
        Return the number of 'rows' in this object - either the length of the first dimension if an ndarray, or
        simply len(obj) otherwise
        """
        if type(obj)==np.ndarray:
            return np.shape[0]
        else:
            return len(obj)

    def _forward_dbg(self, obs):
        """
        Simplest implementation of forward algorithm with no optimizations or numerical tricks for logs,
        for debugging only
        :param obs: Series of observed values
        """
        trellis = [x for x in self.initial]
        for t, o in enumerate(obs):
            newcol = [0. for _ in range(self.states)]
            for j in range(self.states):
                newcol[j] = sum(trellis[i]*self.tr[i,j]*self.em[j].pmf(o, site=t) for i in range(self.states))
            trellis = newcol
        return np.log(sum(x for x in trellis))

    def forward_backward(self, obs, exposures=1.0):
        return forward_backward(obs, self.initial, self.tr, self.em, exposures=exposures)

    def simulate(self, steps):
        obs = []
        states = []
        sprobs = self.initial
        for i in range(steps):
            newstate = stats.rv_discrete(values=(range(self.states), sprobs)).rvs(size=1)[0]
            states.append(newstate)
            o = self.em[newstate].rvs(size=1, site=i)[0]
            obs.append(o)
            sprobs = self.tr[newstate,:].getA1()
        return obs, states

    def viterbi(self, obs, exposure):
        """
        Run viterbi algorithm to identify the most likely path through the data
        :param obs: List of observations, format depends on what emission distributions are expecting
        :param exposure: Exposure parameter, only used by some emission distributions
        :return: Tuple of path, probability, where path is a list of state indices traversed and prob is the probability of that path given the model
        """

        eds = [d.pmf(obs, exposure) for d in self.em]

        # trellis = [np.log(self.initial[i]*self.em[i].pmf(obs[0], exposure, site=0)) for i in range(self.states)]
        trellis = [np.log(self.initial[i] * eds[i][0]) for i in range(self.states)]

        backptrs = []

        if type(obs[0])==list:
            site_max = len(obs[0])
        elif type(obs)==np.matrix:
            site_max = np.max(obs.shape)
        else:
            site_max = len(obs)
        for t in range(1, site_max):
            backtrace_col = []
            newcol = []

            if t%1000==0:
                logging.info("Computing state {} of {} ({:.2f}%)".format(t, site_max, 100.0*float(t)/site_max))
            for j in range(self.states):
                probs = [trellis[i] + np.log(self.tr[i,j]) for i in range(self.states)]
                max_index = np.argmax(probs)
                #newcol.append(probs[max_index] + eds[j][t]) # np.log(self.em[j].pmf(obs, exposure, site=t)))
                #newcol.append(probs[max_index] + np.log(self.em[j].pmf(obs[t], exposure, site=t)))
                newcol.append(probs[max_index] + np.log(eds[j][t]))
                backtrace_col.append(max_index)

            trellis = newcol
            backptrs.append(backtrace_col)

        state = np.argmax(newcol)
        path_prob = newcol[state]
        states = [state]
        for t in range(len(backptrs)-1, -1, -1):
            states.append(backptrs[t][states[-1]])
        states.reverse()
        return states, path_prob


    def viterbi_dbg(self, obs):
        trellis = [self.initial[i] * self.em[i].pmf(obs[0], site=0) for i in range(self.states)]
        backptrs = []
        print("t{}\t {}\t{}".format(0, trellis, backptrs))
        for t, o in zip(range(1, len(obs)), obs[1:]):
            backtrace_col = []
            newcol = []
            for j in range(self.states):
                probs = [trellis[i] * self.tr[i, j] for i in range(self.states)]
                max_index = np.argmax(probs)
                newcol.append(probs[max_index] * self.em[j].pmf(o, site=t))
                backtrace_col.append(max_index)

            trellis = newcol
            backptrs.append(backtrace_col)
            print("t{}\t {}\t{}".format(t, newcol, backtrace_col))

        state = np.argmax(newcol)
        states = [state]
        for t in range(len(backptrs) - 1, -1, -1):
            states.append(backptrs[t][state])
        states.reverse()
        return states

def forward_likelihood_vector(obs, exposure=1.0, initial=None, transitions=None, emissions=None):
    """
    Compute the log-likelihood of the sequence of observations using the Forward algorithm
    We do this in a unbound function so we can parallelize more easily
    :param obs: Series of observed values
    """
    trellis = np.array([x for x in initial])
    offset = 0.0  # Running tab of log-likelihood added to trellis so we can avoid overflow issues when using np.exp

    # Compute logpmfs for all sites for all emissions distributions at outset,
    eds = np.array([d.logpmf(obs, exposure) for d in emissions])

    try:
        for t, o in enumerate(obs):
            #TODO : Replace sum builtin with numpy version (but naive replacement doesn't work)
            newcol = (eds[:,t] + np.log(sum(np.dot(trellis, transitions)))) + offset
            offset = np.max(newcol)
            trellis = (np.exp(newcol - offset)).getA1()

    except ParameterViolation:
        return float("-inf")

    logL = np.log(sum(trellis)) + offset
    return logL

def forward_likelihood(obs, initial=None, transitions=None, emissions=None):
    """
    Compute the log-likelihood of the sequence of observations using the Forward algorithm
    We do this in a unbound function so we can parallelize more easily
    :param obs: Series of observed values
    """
    trellis = [x for x in initial]
    offset = 0.0  # Running tab of log-likelihood added to trellis so we can avoid overflow issues when using np.exp
    num_states = len(initial)

    try:
        for t, o in enumerate(obs):
            newcol = [emissions[j].logpmf(o, site=t) + np.log(sum(trellis[i] * transitions[i, j] for i in range(num_states))) + offset
                      for j in range(num_states)]

            offset = max(newcol)
            for j in range(len(newcol)):
                trellis[j] = np.exp(newcol[j] - offset)
                if math.isnan(trellis[j]):
                    raise ValueError("NaN encountered at step {} in state {}".format(t, j))


    except ParameterViolation:
        return float("-inf")

    logL = np.log(sum(trellis)) + offset
    return logL

def _compute_forward_backward_raw(obs, initial, transitions, emissions, exposures=1.0):
    """
    Run forward-backward algorithm on the observations to determine the probability of being in each state
    at every step.
    :param obs: Set of observations, format is dependent on emission distribution requirements
    :param initial: Initial state vector
    :param transitions: Transition probability matrix
    :param emissions: List of emission distributions
    :return: List of vectors representing probability of being in state j at step t
    """
    forward_trellis = [np.array([x for x in initial])]
    forward_offsets = [0.0] # List of log-likelihood added to trellis so we can avoid overflow issues when using np.exp

    num_states = len(initial)
    #Initialize backward trellis - t-1 column is a special case
    backward_trellis = [np.ones(num_states)]
    backward_offsets = [0.0]

    logging.info("Precomputing observation probs...")
    eds = [d.pmf(obs, exposures) for d in emissions]

    bad_sites = []
    for ed in eds:
        if any(np.isnan(ed)):
            which = np.where(np.isnan(ed))[0]
            bad_sites.extend(which)

    # Also, check for sites at which all probabilities are zero - these will cause things to fail as well
    # TODO Probably the reason for all zeros is a very extreme observation value, which is likely indicative of a CNV
    # Nullifying these will prevent these CNVs from being detected, which seems like a bad idea. Is there a better way
    # to handle this?
    for i in range(len(eds[0])):
        s = sum(eds[j][i] for j in range(num_states))
        if s == 0.0:
            bad_sites.append(i)

    if len(bad_sites) > 10:
        raise ValueError('Too many sites found with NaNs ({}), aborting'.format(len(bad_sites)))

    #Replace data at sites with NaNs with default values
    for site in bad_sites:
        logging.warning("Replacing NaN at site {} with uniform data, site {} will be excluded from analysis!!".format(site, site))
        for ed in eds:
            ed[site] = 1.0

    try:
        for t, o in enumerate(obs):
            if (t%1000==0):
                logging.info("Processing site {} of {} ({:.2f}%)".format(t, len(obs), 100.0*float(t)/len(obs)))

            newcol = np.array([
                            np.log(eds[j][t]*np.sum(forward_trellis[t] * transitions[:, j]))
                            for j in range(num_states)
                            ]) + forward_offsets[-1]


            forward_offsets.append(max(newcol))
            forward_trellis.append(np.exp(newcol - forward_offsets[-1]))

            if any(np.isnan(forward_trellis[-1])):
                    raise ValueError("NaN encountered in forward trellis at step {}".format(t))

            # Backward trellis - start from end and iterate toward beginning
            bt = len(obs)-t-1
            eds2 = np.matrix([[eds[j][bt] * backward_trellis[0][j] for j in
                               range(num_states)]])
            eds2 = np.transpose(eds2)
            back_col = (np.log( transitions.dot(eds2)) + backward_offsets[0]).getA1()

            backward_offsets.insert(0, max(back_col))
            col_to_append = np.exp(back_col - backward_offsets[0])

            if any(np.isnan(col_to_append)):
                raise ValueError("NaN encountered in backward trellis at site {}, obs: {} vals: {}".format(bt, o, col_to_append))

            backward_trellis.insert(0, col_to_append)

    except ParameterViolation:
        return float("-inf")

    return forward_trellis, backward_trellis, forward_offsets, backward_offsets


def forward_backward(obs, initial, transitions, emissions, exposures=1.0):
    """
    Run full forward-backward algorithm and combine forward and backward trellises to
    get all state probabilities
    :param obs:
    :param initial:
    :param transitions:
    :param emissions:
    :param exposures:
    :return:
    """

    (forward_trellis, backward_trellis, foffsets, boffsets) = _compute_forward_backward_raw(obs, initial, transitions, emissions, exposures=exposures)

    final_probs = []
    for t,(f,b) in enumerate(zip(forward_trellis, backward_trellis)):
        probs = f*b / sum(f*b)
        final_probs.append(probs)

    return final_probs

def path_likelihood_fixed_state(obs, state, initial=None, transitions=None, emissions=None):
    """
    Compute likelihood of a single series of observations with a given known sequence of states
    This is implemented as an unbound function to facilitate use with the multiprocessing modules

    :param obs_n_states: List of tuples of (observation, known_state)
    """
    logL = 0.0
    # logT = np.log(transitions)
    try:
        logL += np.log(initial[state] * transitions[state, state] * emissions[state].pmf(obs[0], site=0))
        for t, o in zip(range(1, len(obs) - 1), obs[1:]):
            logL += np.log(transitions[state, state] * emissions[state].pmf(o, site=t))
            if math.isinf(logL):
                logging.warning("Encountered infinite logL for step {} (obs {}) in path_likelihood".format(t, o))
                logging.debug("Transition from {} -> {} : {}".format(state, state, transitions[state, state]))
                logging.debug("PMF: {}".format(emissions[state].pmf(o, site=t)))
                return float("-inf")

        # logL += sum(logT[[known_states[t - 1], known_states[t]]] + emissions[known_states[t]].pmf(o, site=t) for t,o in zip(range(1, len(obs) - 1), obs[1:]))
    except ParameterViolation:
        return float("-inf")

    return logL

def path_likelihood(obs_n_states, initial=None, transitions=None, emissions=None):
    """
    Compute likelihood of a single series of observations with a given known sequence of states
    This is implemented as an unbound function to facilitate use with the multiprocessing modules

    :param obs_n_states: List of tuples of (observation, known_state)
    """
    logL = 0.0
    obs, known_states = obs_n_states
    # logT = np.log(transitions)
    try:
        logL += np.log(initial[known_states[0]] * transitions[known_states[0], known_states[0]] * emissions[known_states[0]].pmf(obs[0], site=0))
        for t, o in zip(range(1, len(obs) - 1), obs[1:]):
            logL += np.log(transitions[known_states[t - 1], known_states[t]] * emissions[known_states[t]].pmf(o, site=t))
            if math.isinf(logL):
                logging.warning("Encountered infinite logL for step {} (obs {}) in path_likelihood".format(t, o))
                logging.debug("Transition from {} -> {} : {}".format(known_states[t-1], known_states[t], transitions[known_states[t - 1], known_states[t]]))
                logging.debug("PMF: {}".format(emissions[known_states[t]].pmf(o, site=t)))
                return float("-inf")

        # logL += sum(logT[[known_states[t - 1], known_states[t]]] + emissions[known_states[t]].pmf(o, site=t) for t,o in zip(range(1, len(obs) - 1), obs[1:]))
    except ParameterViolation:
        return float("-inf")

    return logL

# def q_some_likelihood(segments, obs, state_probs exposures=1.0):
#     """
#     Compute a clumsy probability of observing a particular set of states
#     Currently, this returns the log of the geometric mean of the state probabilities
#     """
#     (forward_trellis, backward_trellis, foffsets, boffsets) = _compute_forward_backward_raw(obs, initial, transitions, emissions, exposures=exposures)
#
#     forward_prob = forward_likelihood(obs, initial, transitions, emissions)
#     num_states = len(initial)
#     segment_logprobs = []
#
#     for seg in segments:
#         start_pos, path = seg[0], seg[1]
#
#         logprob = 0.0
#         for i, state in enumerate(path[0:]):
#             t = start_pos + i
#             f_probs = forward_trellis[t] / sum(forward_trellis[t])
#             b_probs = backward_trellis[t] / sum(backward_trellis[t])
#             logprob += np.log(f_probs[state] * b_probs[state]/sum(f_probs*b_probs))
#
#         segment_logprobs.append(logprob / len(path))
#
#     return segment_logprobs