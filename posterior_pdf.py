import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

# Assumes mean height between 1.65 and 1.8.
# Creates 50 values saved as an array to variable mu.
mu = np.linspace(1.65, 1.8, num = 50)
test = np.linspace(0, 2)

# Prior Distributions

uniform_dist = (sts.uniform.pdf(mu) + 1)
uniform_dist = uniform_dist/uniform_dist.sum() # Non-informative prior.

beta_dist = sts.beta.pdf(mu , 2, 5, loc = 1.65, scale = 0.2)
beta_dist = beta_dist/beta_dist.sum() # Subjective/Informative prior.

plt.plot(mu, beta_dist, label = "Beta Dist (subjective prior)")
plt.plot(mu, uniform_dist, label = "Uniform Dist (non-informative prior)")
plt.title("Probability plot of hypothesized $\mu$ given observed 1.7m")
plt.xlabel("Value of $\mu$ in meters")
plt.ylabel("Probability density")
plt.legend()

# Only shows beta and uniform dists.
plt.show()
# Close the popped-up plt.show() plot/window to show the next plt.show().

def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1)
    return likelihood_out/likelihood_out.sum()

likelihood_out = likelihood_func(1.7, mu)

plt.plot(mu, beta_dist, label = "Beta Dist (subjective prior)")
plt.plot(mu, uniform_dist, label = "Uniform Dist (non-informative prior)")
plt.plot(mu, likelihood_out, label = "Likelihood Dist")
plt.title("Probability plot of hypothesized $\mu$ given observed 1.7m")
plt.xlabel("Value of $\mu$ in meters")
plt.ylabel("Probability density/Likelihood")
plt.legend()

# Shows beta, uniform, and likelihood dists.
# If an existing instance of plt.show() is removed, the next code run
# would show 5 plots (2 betas, 2 uniforms, 1 likelihood since it references
# all existing plots between previous and next instance).
plt.show()

unnormalized_posterior = likelihood_out * uniform_dist
plt.plot(mu, beta_dist, label = "Beta Dist (subjective prior)")
plt.plot(mu, uniform_dist, label = "Uniform Dist (non-informative prior)")
plt.plot(mu, likelihood_out, label = "Likelihood Dist")
plt.plot(mu, unnormalized_posterior, label = "Unnormalized posterior")
plt.title("Probability plot of hypothesized $\mu$ given observed 1.7m")
plt.xlabel("Value of $\mu$ in meters")
plt.ylabel("Probability density/Likelihood")
plt.legend()

# Shows beta, uniform, likelihood, and unnormalized posterior dists.
plt.show()

normalized_posterior = unnormalized_posterior/unnormalized_posterior.sum()
plt.plot(mu, beta_dist, label = "Beta Dist (subjective prior)")
plt.plot(mu, uniform_dist, label = "Uniform Dist (non-informative prior)")
plt.plot(mu, likelihood_out, label = "Likelihood Dist")
plt.plot(mu, unnormalized_posterior, label = "Unnormalized posterior")
plt.plot(mu, normalized_posterior, label = "Normalized posterior")
plt.title("Probability plot of hypothesized $\mu$ given observed 1.7m")
plt.xlabel("Value of $\mu$ in meters")
plt.ylabel("Probability density/Likelihood")
plt.legend()

# Shows beta, uniform, likelihood, unnormalized posterior, and
# normalized posterior dists.
plt.show()


# This model now shows the actual pdf for the posterior since it
# has already accounted for the marginal probability or the 
# normalizing value (the denominator of bayes' theorem formula).