"""
Basic statistics computation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def basic_stats(data, ddof=1):
    """
    Compute basic stats on data
    """
    # Compute basic stats
    d_count = data.size
    n = d_count
    d_min = np.min(data)
    d_max = np.max(data)
    d_median = np.median(data)
    d_mean = np.mean(data)
    d_var = np.var(data, ddof=ddof)
    d_std = np.std(data, ddof=ddof)

    results = {
        "min":d_min,
        "max":d_max,
        "median":d_median,
        "mean":d_mean,
        "var":d_var,
        "std":d_std,
        "count":d_count,
    }
    return results


def compute_CI(data, mu_X=None, sigma_X=None, law_X="normal", alpha=0.05, print_CI=False):
    """
    Compute Confidence Interval.
    """
    d_count = data.size
    n = d_count
    
    # Mean
    ## If X follows normal lax
    if law_X=="normal":
        # In all case, Xbar follows a symetric lax
        alpha_Xbar_2 = alpha/2

        # If X variance is known
        if sigma_X is not None:
            # Compute std of Xbar law
            sigma_Xbar = sigma_X/np.sqrt(n)

            # Then Y=((Xbar-mu)/sigma_xbar) follows a N(0,1) law

            # Compute boundary for N(0,1) such that the area between boundaries is proba_Xbar%

            # Using the percent point function, such that ppf(proba) give the upper bound
            zscore_CI_p = st.norm.ppf(1 - alpha_Xbar_2)
            # Since the law is N(0,1), law is symetric so
            zscore_CI_m = - zscore_CI_p

            # Translate the bounds in terms of X such that mu is in Xbar +-sigma_Xbar, 
            # with probability proba_Xbar%
            ci_Xbar_p = zscore_CI_p * sigma_xbar
            ci_Xbar_m = zscore_CI_m * sigma_xbar

            # In the end, we know mu should be in XBar+-ci at proba_Xbar%
            if print_CI:
                print("\sigma_X known, using z-score to compute CI for Xmean")
                print("\mu is within {}+/-{} at {}%".format(d_mean, ci_Xbar_p, alpha))

        # If X variance is not known
        if sigma_X is None:
            # First, estimate sigma_X, and then Xbar follow a Student t law
            s_X = np.std(data, ddof=ddof)
            s_Xbar = s_X/np.sqrt(n)

            df_Xbar = n-1
            tscore_CI_p = st.t.ppf(1 - alpha_Xbar_2, df=df_Xbar)
            # Since we have a Studen law, it is symetric
            tscore_CI_m = - tscore_CI_p

            # Translate the bounds in terms of X such that mu is in Xbar +-sigma_Xbar at proba_Xbar% 
            ci_Xbar_p = tscore_CI_p * s_Xbar
            ci_Xbar_m = tscore_CI_m * s_Xbar
            if print_CI:
                print("\sigma_{X} unknow, estimating it and using Student's t distribution for Xmean")
                print("\mu is within {}+-{} at {}%".format(d_mean, ci_Xbar_p, alpha))


    # Variance
    ## Variance follows a Ki2 lax with df=n-1 dof
    if law_X=="normal":
        # If X mean is known
        if mu_X is not None:
            # s2 follows a ki2 with N degrees 
            s2 = np.mean(np.abs(data - mu_X)**2) * n/(n-ddof)

            # Compute reduced variable bounds 
            kiscore_CI_var_p = st.chi2.ppf(1 - alpha, df=n)
            kiscore_CI_var_m = st.chi2.ppf(alpha, df=n)

            # Translate in terms of variance
            ci_var_p = s2 *(n-ddof) / kiscore_CI_var_m
            ci_var_m = s2 *(n-ddof) / kiscore_CI_var_p
            if print_CI:
                print("\mu_X known, using chi2 score with df=n to compute CI for var")
                print("Var_x is wihtin {}+-{}/{} at {}%".format(s2, ci_var_m, ci_var_p, alpha))


        # If X mean is not knwon
        if mu_X is None:
            # s2 folllow a ki2 lax at N-1 degrees
            s2 = np.var(data, ddof=ddof)

            # Compute reduced variable bounds
            kiscore_CI_var_p = st.chi2.ppf(1 - alpha, df=n)
            kiscore_CI_var_m = st.chi2.ppf(alpha, df=n)

            # Translate in terms of variance
            ci_var_p = s2 *(n-ddof) / kiscore_CI_var_m
            ci_var_m = s2 *(n-ddof) / kiscore_CI_var_p
            if print_CI:
                print("\mu_X not known, using chi2 score with df=n-1 to compute CI for var")
                print("Var_x is wihtin {}+-{}/{} at {}%".format(s2, ci_var_m, ci_var_p, alpha))

    results = {
        "ci_Xbar_m":ci_Xbar_m,
        "ci_Xbar_p":ci_Xbar_p,
        "ci_var_m":ci_var_m,
        "ci_var_p":ci_var_p,
    }
    return results





def analyse_normal_seq(data, mu_X=None, sigma_X=None, law_X="normal", alpha=0.05, print_CI=False, ddof=1):
    """
    Compute basic stats about data.
    Return the results in a dict.
    """

    # Compute basic stats
    d_count = data.size
    n = d_count
    d_min = np.min(data)
    d_max = np.max(data)
    d_median = np.median(data)
    d_mean = np.mean(data)
    d_var = np.var(data, ddof=1)
    d_std = np.std(data, ddof=1)

    # Mean
    ## If X follows normal lax
    if law_X=="normal":
        # In all case, Xbar follows a symetric lax
        alpha_Xbar_2 = alpha/2

        # If X variance is known
        if sigma_X is not None:
            # Compute std of Xbar law
            sigma_Xbar = sigma_X/np.sqrt(n)

            # Then Y=((Xbar-mu)/sigma_xbar) follows a N(0,1) law

            # Compute boundary for N(0,1) such that the area between boundaries is proba_Xbar%

            # Using the percent point function, such that ppf(proba) give the upper bound
            zscore_CI_p = st.norm.ppf(1 - alpha_Xbar_2)
            # Since the law is N(0,1), law is symetric so
            zscore_CI_m = - zscore_CI_p

            # Translate the bounds in terms of X such that mu is in Xbar +-sigma_Xbar, 
            # with probability proba_Xbar%
            ci_Xbar_p = zscore_CI_p * sigma_xbar
            ci_Xbar_m = zscore_CI_m * sigma_xbar

            # In the end, we know mu should be in XBar+-ci at proba_Xbar%
            if print_CI:
                print("\sigma_X known, using z-score to compute CI for Xmean")
                print("\mu is within {}+/-{} at {}%".format(d_mean, ci_Xbar_p, alpha))

        # If X variance is not known
        if sigma_X is None:
            # First, estimate sigma_X, and then Xbar follow a Student t law
            s_X = np.std(data, ddof=ddof)
            s_Xbar = s_X/np.sqrt(n)

            df_Xbar = n-1
            tscore_CI_p = st.t.ppf(1 - alpha_Xbar_2, df=df_Xbar)
            # Since we have a Studen law, it is symetric
            tscore_CI_m = - tscore_CI_p

            # Translate the bounds in terms of X such that mu is in Xbar +-sigma_Xbar at proba_Xbar% 
            ci_Xbar_p = tscore_CI_p * s_Xbar
            ci_Xbar_m = tscore_CI_m * s_Xbar
            if print_CI:
                print("\sigma_{X} unknow, estimating it and using Student's t distribution for Xmean")
                print("\mu is within {}+-{} at {}%".format(d_mean, ci_Xbar_p, alpha))


    # Variance
    ## Variance follows a Ki2 lax with df=n-1 dof
    if law_X=="normal":
        # If X mean is known
        if mu_X is not None:
            # s2 follows a ki2 with N degrees 
            s2 = np.mean(np.abs(data - mu_X)**2) * n/(n-ddof)

            # Compute reduced variable bounds 
            kiscore_CI_var_p = st.chi2.ppf(1 - alpha, df=n)
            kiscore_CI_var_m = st.chi2.ppf(alpha, df=n)

            # Translate in terms of variance
            ci_var_p = s2 *(n-ddof) / kiscore_CI_var_m
            ci_var_m = s2 *(n-ddof) / kiscore_CI_var_p
            if print_CI:
                print("\mu_X known, using chi2 score with df=n to compute CI for var")
                print("Var_x is wihtin {}+-{}/{} at {}%".format(s2, ci_var_m, ci_var_p, alpha))


        # If X mean is not knwon
        if mu_X is None:
            # s2 folllow a ki2 lax at N-1 degrees
            s2 = np.var(data, ddof=ddof)

            # Compute reduced variable bounds
            kiscore_CI_var_p = st.chi2.ppf(1 - alpha, df=n)
            kiscore_CI_var_m = st.chi2.ppf(alpha, df=n)

            # Translate in terms of variance
            ci_var_p = s2 *(n-ddof) / kiscore_CI_var_m
            ci_var_m = s2 *(n-ddof) / kiscore_CI_var_p
            if print_CI:
                print("\mu_X not known, using chi2 score with df=n-1 to compute CI for var")
                print("Var_x is wihtin {}+-{}/{} at {}%".format(s2, ci_var_m, ci_var_p, alpha))
                
    results = {
        "min":d_min,
        "max":d_max,
        "median":d_median,
        "mean":d_mean,
        "var":d_var,
        "std":d_std,
        "count":d_count,
        "ci_Xbar_m":ci_Xbar_m,
        "ci_Xbar_p":ci_Xbar_p,
        "ci_var_m":ci_var_m,
        "ci_var_p":ci_var_p,
    }
    return results

    # visual 
    ## histogram
    ## qqplot
    ## ppplot