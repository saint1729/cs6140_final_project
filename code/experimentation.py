import numpy as np

if __name__ == '__main__':
    l = np.load("./old/class8_313.npy")
    print(l.shape)
    l = np.load("./old/resources/pts_in_hull.npy")
    print(l[:, 0].shape)
    l = np.load("./old/resources/prior_lab_distribution_train.npz", allow_pickle=True)
    lst = l.files

    print(l['w_bins'])
    if np.array_equal(l['ab_bins'], np.array([l['a_bins'], l['b_bins']]).T):
        print("Equal")
    else:
        print("Not equal")

    for item in lst:
        print(item)
        print(l[item].shape)
