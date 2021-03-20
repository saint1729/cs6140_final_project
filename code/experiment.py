import numpy as np


if __name__ == '__main__':
    prior_probabilities = np.load("./resources/prior_probs.npy")
    print (sum(prior_probabilities))

    pts_in_hull = np.load("./resources/pts_in_hull.npy")
    print (len(pts_in_hull))
