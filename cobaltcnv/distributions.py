
import scipy.stats as stats
import numpy as np
import logging


class BaseDistribution(object):
    """
    Base class for distributions that are used for emission probabilities to avoid
    re-writing the same functions for every class we might ever want to use
    """

    def pmf(self, x, site=-1):
        return self.dist.pmf(x)

    def logpmf(self, x, site=-1):
        return self.dist.logpmf(x)

    def rvs(self, size=1, site=-1):
        return self.dist.rvs(size=size)

    def meanvar(self, site=-1):
        return self.dist.stats(moments='mv')

    def update_params(self, newparams):
        pass

    def desc(self):
        """
        Short, user-readable description of model state e.g. "Diploid", "Copy-neutral LOH", "Het deletion", etc.
        """
        return None

    def copy_number(self):
        """
        Copy number associated with this state (integer for germline calls)
        """
        return None


class TestNormal(BaseDistribution):

    def __init__(self, mean, sd):
        self.dist = stats.norm(mean, sd)

class PosDependentSkewNormal(BaseDistribution):
    """
    Skew-normal distributions with independent params for each site
    Unlike other distributions, this one doesn't have updatable parameters and
    can't be used in a training context
    """

    MAX_DENSITY = 1e9
    MIN_DENSITY = 1e-300

    def __init__(self, params, user_desc=None, copy_number=None):
        """
        Params should be a list of lists where params[i] are the parameters for the distribution at the ith site
        :param params:
        """
        self.params = params
        self.user_desc = user_desc
        self.copynum = copy_number
        self.dists = self._build_dists()

    def desc(self):
        return self.user_desc

    def copy_number(self):
        return self.copynum

    def _build_dists(self):
        a = np.array([p[0] for p in self.params])
        shape = np.array([p[1] for p in self.params])
        loc = np.array([p[2] for p in self.params])
        return stats.skewnorm(a, shape, loc)

    def pmf(self, obs, exposure=-1, site=-1):
        if site==-1:
            d = self.dists.pdf(obs)
            d[d < PosDependentSkewNormal.MIN_DENSITY ] = PosDependentSkewNormal.MIN_DENSITY
            which = np.where(np.isnan(d))[0]
            for i in which:
                m, v = self.meanvar(i)
                logging.debug("Replacing NaN at site {} with MIN_DENSITY, obs was {}, mean / var are {} / {}".format(i, obs, m, v))
                d[i] = PosDependentSkewNormal.MIN_DENSITY

            return d
        else:
            val = self.dist_for_site(site).pdf(obs)
            if val > PosDependentSkewNormal.MAX_DENSITY:
                logging.debug("Replacing +infinite value with {} (site: {})".format(PosDependentSkewNormal.MAX_DENSITY, site))
                val = PosDependentSkewNormal.MAX_DENSITY
            if val < PosDependentSkewNormal.MIN_DENSITY:
                logging.debug("Replacing tiny value {} with {} (site: {})".format(val, PosDependentSkewNormal.MIN_DENSITY, site))
                val = PosDependentSkewNormal.MIN_DENSITY
            if np.isnan(val):
                logging.warning("Replacing NaN value with {} (site: {})".format(PosDependentSkewNormal.MIN_DENSITY, site))
                val = PosDependentSkewNormal.MIN_DENSITY
            return val

    def logpmf(self, obs, exposure=-1, site=-1):
        if site==-1:
            return self.dists.logpdf(obs)[0]
        else:
            val = self.dist_for_site(site).logpdf(obs)
            return val

    def dist_for_site(self, site):
        return stats.skewnorm(self.params[site][0], self.params[site][1], self.params[site][2])

    def params_for_site(self, site):
        return self.params[site][0], self.params[site][1], self.params[site][2]

    def update_params(self, newparams):
        raise ValueError('No params to update here')

    def meanvar(self, site=-1):
        if site==-1:
            return self.dists.stats(moments='mv')

        return self.dist_for_site(site).stats(moments='mv')


