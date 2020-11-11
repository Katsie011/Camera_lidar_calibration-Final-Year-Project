import numpy as np
from scipy.stats import norm, gaussian_kde, entropy
import statsmodels.api as sm


def get_omega(X, Y=None):
    """Returns the smoothing matrix kernel obtained by silverman's estimate of the form:
    [std(X), 0],[0, std(Y)]
    """
    if not Y:
        return 1.06 * (len(X)**(-1 / 5)) * np.std(X)
    else:
        n = len(X)
        omega = 1.06 * (n**(1 / 5)) * np.array([np.std(X), 0], [0, np.std(Y)])
        return omega
    
def get_kde_1D(X):
    """Return the smoothed kde for X"""
    w = get_omega(X)
    print("Bandwidth for kde: {}".format(w))
    kde = sm.nonparametric.KDEUnivariate(X)
    # kde.fit(bw=omega)  # Estimate the densities
    kde.fit(bw=(w))
    
    return kde


def get_kde_2D(im1,im2):
    
    x, y = np.mgrid[0:255:256j, 0:255:256j,]
    positions = np.vstack([x.ravel(), y.ravel()])


    values = np.vstack([im1, im2])
    kernel = gaussian_kde(values, bw_method='silverman')
    Z = np.reshape(kernel(positions).T, x.shape)
    
    return Z


# Unsmoothed MI scores
def get_mutual_info_score(x, y, threshold=3000,too_small_reduction = 0.1, use_hist = True):
    """
    Return the mutual information score for a joint histogram
    """
    
    X = np.array(x).ravel()
    Y = np.array(y).ravel()
    
    too_small = False
    
    if len(X)*len(Y) < threshold:
        too_small=True
    

    if use_hist:
        Z, _, _ = np.histogram2d(X, Y, bins=255)
    else:                   
        Z = get_kde_2D(X, Y)

    p_joint = Z / np.sum(Z)
    p_x = np.sum(p_joint, axis=1)
    p_y = np.sum(p_joint, axis=0)

    px_py = p_x[:, None] * p_y[None, :]
    non_zeros = p_joint > 0  # Only non-zero pxy values contribute to the sum

    if too_small:
        print("Overlap too small, reducing score")
        return too_small_reduction * np.sum(p_joint[non_zeros] *
                      np.log(p_joint[non_zeros] / px_py[non_zeros]))
    else:
        return np.sum(p_joint[non_zeros] *
                      np.log(p_joint[non_zeros] / px_py[non_zeros]))
    

    
def get_Normalised_mutual_info_score(x, y, threshold=3000,too_small_reduction = 0.1, use_hist = True):
    """
    Return the normalised mutual information score for a joint histogram
    """
    
    NMI = 2*get_mutual_info_score(x, y, threshold,too_small_reduction, use_hist)/(entropy(np.array(x).ravel()) + entropy(np.array(y).ravel()))
    
    return NMI