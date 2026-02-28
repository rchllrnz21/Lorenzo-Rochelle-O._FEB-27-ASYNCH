# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 20:46:39 2026

@author: rochelle
"""

import scipy as sp

unnormalized_posterior = likelihood_out * uniform_dist
plt.plot(mu, unnormalized_posterior)
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.show()