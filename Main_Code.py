"""
Created on Mon Nov 17 23:03:50 2025

@author: alan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#3.1: data
data_frame = pd.read_excel("/Users/alan/Library/CloudStorage/OneDrive-ImperialCollegeLondon/CP/CP_data.xlsx")
data_column = data_frame.iloc[:, 0]
unosc_column = data_frame.iloc[:, 1]

data_array = np.array(data_column, dtype=float)
unosc_array = np.array(unosc_column, dtype=float)
n_bins = len(data_array)

E_min = 0.0
E_max = 10.0  # GeV
bin_width = (E_max - E_min) / n_bins

E_center = np.zeros(n_bins)
for i in range(n_bins):
    E_center[i] = E_min + (i + 0.5) * bin_width

#plotting
plt.figure(figsize=(8, 5))
plt.step(E_center, data_array, label="Observed data", linewidth=1.5)
plt.step(E_center, unosc_array, label="Unoscillated prediction", linewidth=1.5)
plt.xlabel("Neutrino energy E (GeV)")
plt.ylabel("Events per bin")
plt.title("Section 3.1: Data and unoscillated prediction")
plt.legend()
plt.show()


#3.2: fit function
def survival_probability(E_center, theta23, dm_23_sq, L=295.0):
    n = len(E_center)
    P_array = np.zeros(n)
    sin_2theta_sq = (math.sin(2.0 * theta23))**2

    for i in range(n):
        E = float(E_center[i])
        phase = 1.267 * dm_23_sq * L / E
        sin_phase_sq = (math.sin(phase))**2
        P = 1.0 - sin_2theta_sq * sin_phase_sq
        P_array[i] = P
        
    return P_array

theta23_guess = 0.21*np.pi
dm_23_sq_guess = 2.36e-3 

P_array = survival_probability(E_center, theta23_guess, dm_23_sq_guess)
plt.figure(figsize=(8, 5))
plt.plot(E_center, P_array)
plt.xlabel("Neutrino energy E (GeV)")
plt.ylabel("Survival probability P")
plt.title("Section 3.2: Survival probability vs energy")
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.show()


def oscillated_prediction(E_array, unosc_array, theta23, dm_23_sq, L=295.0):
    n = len(E_array)
    lambda_array = np.zeros(n)
    P_array = survival_probability(E_center, theta23, dm_23_sq) 
    for i in range(n):
        lambda_array[i] = unosc_array[i] * P_array[i]
        
    return lambda_array 

lambda_array = oscillated_prediction(E_center, unosc_array, theta23_guess,dm_23_sq_guess)

plt.figure(figsize=(8, 5))
#plt.step(E_center, unosc_array, label="Unoscillated prediction", linewidth=1.5)
plt.step(E_center, lambda_array,label="Oscillated prediction",linewidth=1.5)
plt.step(E_center, data_array,label="Observed data",linewidth=1.5)

plt.xlabel("Neutrino energy E (GeV)")
plt.ylabel("Events per bin")
plt.title("Section 3.2: Data vs unoscillated vs oscillated prediction")
plt.legend()
plt.show()


#3.3: NLL
def NLL(theta23, dm_23_sq, E_center, unosc_array, data_array, L=295.0):
    """
    Compute NLL(theta23) for fixed dm_23_sq.
    """
    lambda_array = oscillated_prediction(E_center, unosc_array, theta23, dm_23_sq, L)
    n = len(E_center)
    nll_sum = 0.0

    for i in range(n):
        lambda_i = float(lambda_array[i])
        m_i = float(data_array[i])
        term = lambda_i - m_i * math.log(lambda_i)
        nll_sum += term

    return 2*nll_sum

dm_23_sq_fixed = 2.4e-3
theta_vals = np.linspace(0.0, 0.5 * np.pi, 200)
nll_vals = []

for theta in theta_vals:
    nll = NLL(theta, dm_23_sq_fixed, E_center, unosc_array, data_array)
    nll_vals.append(nll)

nll_vals = np.array(nll_vals, dtype=float)

plt.figure(figsize=(8,5))
plt.plot(theta_vals/np.pi, nll_vals)
plt.xlabel(r'$\theta_{23} / \pi$')
plt.ylabel('Negative log-likelihood  (NLL)')
plt.title(r'Section 3.3: NLL as a function of $\theta_{23}$ for fixed $\Delta m_{23}^2 = 2.4\times10^{-3}$')
plt.grid(True)
plt.show()


#3.4: 1—D Minimize
def parabola_x3(x0, x1, x2, y0, y1, y2):
    x3=0.5*((x2**2-x1**2)*y0+(x0**2-x2**2)*y1+(x1**2-x0**2)*y2)/((x2-x1)*y0+(x0-x2)*y1+(x1-x0)*y2)
    return x3

def parabolic_minimiser(func, x0, x1, x2, tol, max_iter=50):
    """
    1D parabolic minimizer
    """
    xs = [x0, x1, x2]
    xs.sort()
    x0, x1, x2 = xs
    y0 = func(x0)
    y1 = func(x1)
    y2 = func(x2)

    points = [(x0, y0), (x1, y1), (x2, y2)]

    for it in range(max_iter):
        x3 = parabola_x3(x0, x1, x2, y0, y1, y2)
        y3 = func(x3)
        points.append((x3, y3))

        if abs(x3 - x1) < tol:
            return x3, y3, it + 1

        if x3 < x1:
            if y3 < y1:
                x2, y2 = x1, y1
                x1, y1 = x3, y3
            else:
                x0, y0 = x3, y3
        else:  # x3 > x1
            if y3 < y1:
                x0, y0 = x1, y1
                x1, y1 = x3, y3
            else:
                x2, y2 = x3, y3

    # if don't converge after max_iterations
    return x1, y1, max_iter
    
def test_func(x):
    return -np.sin(x)

x0, x1, x2 = 0.1, 0.5, 2
x_min, f_min, n_iter = parabolic_minimiser(test_func, x0, x1, x2, tol=1e-6, max_iter=1000)

print("x_min =", x_min)
print("f_min =", f_min)
print("iters =", n_iter)

def NLL_theta23(theta23):
    return NLL(theta23, dm_23_sq_fixed, E_center, unosc_array, data_array)

x0, x1, x2 = 0.16*np.pi, 0.2*np.pi, 0.24*np.pi
x_min, y_min, n_iter = parabolic_minimiser(NLL_theta23, x0, x1, x2, tol=1e-6, max_iter=1000)
print("x_min =", x_min/np.pi)
print("y_min =", y_min)
print("iters =", n_iter)

theta_plot = np.linspace(x0, x2, 200)
nll_plot = [NLL_theta23(t) for t in theta_plot]

plt.figure(figsize=(8, 5))
plt.plot(theta_plot/np.pi, nll_plot, label='NLL$(\\theta_{23})$ scan')
plt.axvline(x_min/np.pi, linestyle='--', label='Best-fit $\\theta_{23}$', color="red")
plt.scatter(x_min/np.pi, y_min, color="red")
plt.xlabel(r'$\theta_{23} / \pi$')
plt.ylabel('Negative log-likelihood (NLL)')
plt.title(r'Section 3.4: NLL vs $\theta_{23}$ with parabolic minimum')
plt.legend()
plt.grid(True)
plt.show()


#3.5: the accuracy of fit result
def error_from_scan(NLL_func, theta_best, nll_best, step=0.001*np.pi, theta_min_bound=0.0, theta_max_bound=0.25*np.pi):
    """
    error from ΔNLL=1
    """
    target = nll_best + 1.0   # ΔNLL = 1 → 1σ

    theta_left = theta_best
    nll_left = nll_best
    while theta_left > theta_min_bound:
        theta_left -= step
        nll_left = NLL_func(theta_left)
        if nll_left >= target:
            break

    theta_right = theta_best
    nll_right = nll_best
    while theta_right < theta_max_bound:
        theta_right += step
        nll_right = NLL_func(theta_right)
        if nll_right >= target:
            break

    err_minus = theta_best - theta_left
    err_plus  = theta_right - theta_best

    return err_minus, err_plus

err_minus, err_plus = error_from_scan(NLL_theta23, theta_best=x_min, nll_best=y_min, step=0.001*np.pi, theta_min_bound=0.0, theta_max_bound=0.25*np.pi)

print("err_minus/pi =", err_minus / np.pi)
print("err_plus/pi =", err_plus  / np.pi)

plt.figure(figsize=(8,5))
plt.xlim(0.16,0.24)
plt.ylim(-200,0)
plt.plot(theta_vals/np.pi, nll_vals, label='NLL$(\\theta_{23})$ scan')
plt.axvline(x_min/np.pi, linestyle='--', color='red',  label='Best-fit (minimiser)')
plt.scatter(x_min/np.pi, y_min, color="red")
plt.axvline((x_min-err_minus)/np.pi, linestyle=':', color='green', label='ΔNLL = 1')
plt.scatter((x_min-err_minus)/np.pi, y_min+1, color="green")
plt.axvline((x_min+err_plus)/np.pi, linestyle=':', color='green')
plt.scatter((x_min+err_plus)/np.pi, y_min+1, color="green")
plt.xlabel(r'$\theta_{23} / \pi$')
plt.ylabel('Negative log-likelihood (NLL)')
plt.title(r'Section 3.5: $\theta_{23}$ error from $\Delta$NLL = 1')
plt.legend()
plt.show()

# 3.5: second method
def error_from_curvature(NLL_func, theta_best, step=0.001*np.pi):
    """
    error from curvature
    """
    theta1 = theta_best - step
    theta2 = theta_best
    theta3 = theta_best + step

    y1 = NLL_func(theta1)
    y2 = NLL_func(theta2)
    y3 = NLL_func(theta3)

    # fit a 2nd order ploynomial：a*theta^2 + b*theta + c
    a, b, c = np.polyfit([theta1, theta2, theta3], [y1, y2, y3], 2)
    # error from the curvature of the last parabolic estimate：sigma = 1/sqrt(a)
    sigma = 1.0 / math.sqrt(a)

    return sigma

sigma_curv = error_from_curvature(NLL_theta23, theta_best=x_min, step=0.001*np.pi)
print("sigma (curvature)/pi =", sigma_curv / np.pi)


# 4: Two-dimensional minimisation
def NLL_2D(theta23, dm_23_sq):
    return NLL(theta23, dm_23_sq, E_center, unosc_array, data_array)
def parabolic_minimiser_2D(func2d, theta_init, dm_init, theta_step=0.02*np.pi, dm_step=0.2e-3, theta_bounds=(0.0, 0.25*np.pi), dm_bounds=(0, 1.0e-2), tol_theta=1e-6, tol_dm=1e-7, max_iter=20):
    """
    univariate method 2D minimization
    """
    theta = float(theta_init)
    dm = float(dm_init)

    for it in range(max_iter):
        #Minimize θ for fixed dm
        def f_theta(t):
            return func2d(t, dm)
        x0 = max(theta - theta_step, theta_bounds[0])
        x2 = min(theta + theta_step, theta_bounds[1])
        x1 = theta
        xs = sorted(set([x0, x1, x2]))
        theta_new, nll_theta, _ = parabolic_minimiser(f_theta, xs[0], xs[1], xs[2], tol=tol_theta, max_iter=100)
                                                      
        # fix new θ, minimize dm
        def f_dm(m):
            return func2d(theta_new, m)
        y0 = max(dm - dm_step, dm_bounds[0])
        y2 = min(dm + dm_step, dm_bounds[1])
        y1 = dm
        ys = sorted(set([y0, y1, y2]))
        dm_new, nll_dm, _ = parabolic_minimiser(f_dm, ys[0], ys[1], ys[2], tol=tol_dm, max_iter=100)

        # check convergence
        if abs(theta_new - theta) < tol_theta and abs(dm_new - dm) < tol_dm:
            theta, dm = theta_new, dm_new
            nll_best = func2d(theta, dm)
            return theta, dm, nll_best, it + 1

        theta, dm = theta_new, dm_new

#fit a toy function
def test_func_2D(theta, dm):
    return (theta - 0.6)**2 + (dm - 2.2e-3)**2

theta_test_best, dm_test_best, f_test_min, n_iter_test = parabolic_minimiser_2D(test_func_2D, theta_init=0.3*np.pi, dm_init=1.5e-3)

print("expected minimum: θ = 0.6, Δm² = 2.2e-3")
print("found: θ = {:.6f}, Δm² = {:.6e}, f_min = {:.3e}, iterations = {}".format(theta_test_best, dm_test_best, f_test_min, n_iter_test))

#fit NLL_2D
theta_init = x_min
dm_init = dm_23_sq_fixed

theta_best_2d, dm_best_2d, nll_min_2d, n_iter_2d = parabolic_minimiser_2D(NLL_2D,theta_init=theta_init,dm_init=dm_init)

print("theta23_best/π = {:.6f}".format(theta_best_2d / np.pi))
print("Δm²_23_best = {:.6e}".format(dm_best_2d))
print("NLL_min = {:.3f}".format(nll_min_2d))
print("iterations = {}".format(n_iter_2d))


# Profile-likelihood error (ΔNLL = 1)
def profile_theta(theta):
    """
    for certain θ, use 1D parabolic_minimiser to minimize dm
    """
    def f_dm(dm2):
        return NLL_2D(theta, dm2)

    dm0 = dm_best_2d
    dm_step_local = 0.3e-3
    dm_left  = dm0 - dm_step_local
    dm_right = dm0 + dm_step_local

    dm_min_local, nll_min_local, _ = parabolic_minimiser(f_dm, dm_left, dm0, dm_right, tol=1e-7, max_iter=100)
    return nll_min_local


def profile_dm(dm2):
    """
    for certain dm, use 1D parabolic_minimiser to minimize θ
    """
    def f_theta(theta):
        return NLL_2D(theta, dm2)

    th0 = theta_best_2d
    th_step_local = 0.04*np.pi
    th_left = th0 - th_step_local
    th_right = th0 + th_step_local

    theta_min_local, nll_min_local, _ = parabolic_minimiser(f_theta, th_left, th0, th_right, tol=1e-6, max_iter=100)
    return nll_min_local   


# error for θ
theta_err_minus, theta_err_plus = error_from_scan(NLL_func=profile_theta, theta_best=theta_best_2d, nll_best=nll_min_2d, step=0.001*np.pi, theta_min_bound=0.0, theta_max_bound=0.5*np.pi)

# error for Δm²_23 
dm_err_minus, dm_err_plus = error_from_scan(NLL_func=profile_dm, theta_best=dm_best_2d, nll_best=nll_min_2d, step=0.05e-3, theta_min_bound=1.0e-3, theta_max_bound=4.0e-3)

print("θ_23 = ({:.5f} - {:.5f} + {:.5f})π".format(theta_best_2d/np.pi, theta_err_minus/np.pi, theta_err_plus/np.pi))
print("Δm²_23 = ({:.3e} - {:.3e} + {:.3e})eV²".format(dm_best_2d, dm_err_minus, dm_err_plus))


# section 5: Detector
def gaussian_response_matrix(E_array, mu, sigma):
    """
    Build a simple Gaussian detector response matrix R[i, j].

    Parameters:
    E_array : 1D numpy array
        Bin centres of the (true / reconstructed) energy bins, in GeV.
    mu : float
        Mean shift of the reconstructed energy (in GeV).
        If mu > 0, the reconstructed energy is on average higher.
    sigma : float
        Width of the Gaussian response (in GeV).
        This mimics the detector energy resolution.

    Returns:
    R : 2D numpy array of shape (n_bins, n_bins)
        R[i, j] = probability that an event in true-energy bin i is reconstructed in bin j.
        For each i: sum_j R[i, j] = 1 (row-normalised).
    """
    n = len(E_array)
    R = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dE = E_array[j] - E_array[i] - mu
            R[i, j] = math.exp(-0.5 * (dE / sigma)**2)
        # Normalise each row so that sum_j R[i, j] = 1
        row_sum = np.sum(R[i, :])
        if row_sum > 0.0:
            R[i, :] /= row_sum
    return R


def convolved_spectrum(E_array, lambda_true, mu, sigma):
    """
    Convolve the true predicted spectrum with a Gaussian detector response.

    Parameters:
    E_array : Energy bin centres (GeV).
    lambda_true : Predicted event counts per bin as a function of TRUE energy.
    mu : Mean shift of the reconstructed energy (GeV).
    sigma : Width of the Gaussian response (GeV).

    Returns:
    lambda_reco : Predicted event counts per bin as a function of RECONSTRUCTED energy.
    """
    n = len(E_array)
    R = gaussian_response_matrix(E_array, mu, sigma)

    # Do the discrete convolution: lambda_reco[j] = sum_i R[i, j] * lambda_true[i]
    lambda_reco = np.zeros(n)
    for j in range(n):
        sum_events = 0.0
        for i in range(n):
            sum_events += R[i, j] * lambda_true[i]
        lambda_reco[j] = sum_events
    return lambda_reco


theta_test = theta23_guess
dm_test = dm_23_sq_guess

# True (unsmeared) spectrum
lambda_true = oscillated_prediction(E_center, unosc_array, theta_test, dm_test)

# (A) Fix mu = 0, vary sigma
sigma_list = [0.05, 0.1, 0.20]
mu_fixed = -0.1

plt.figure(figsize=(8, 5))
for sigma in sigma_list:
    lambda_reco_test = convolved_spectrum(E_center, lambda_true, mu_fixed, sigma)
    plt.step(E_center, lambda_reco_test, where="mid", label= "$\\mu = {:.2f}$ GeV, $\\sigma = {:.2f}$ GeV".format(mu_fixed, sigma))

plt.xlabel(r"Reconstructed neutrino energy $E_{\mathrm{reco}}$ (GeV)")
plt.ylabel("Events per bin")
plt.title("Section 5: Effect of detector resolution (varying $\\sigma$, $\\mu=0$)")
plt.legend()
plt.grid(True)
plt.show()


# (B) Fix sigma, vary mu
sigma_fixed = 0.10
mu_list = [-0.10, 0.0, 0.10]

plt.figure(figsize=(8, 5))
for mu in mu_list:
    lambda_reco_test = convolved_spectrum(E_center, lambda_true, mu, sigma_fixed)
    plt.step(E_center, lambda_reco_test, where="mid", label= "$\\mu = {:.2f}$ GeV, $\\sigma = {:.2f}$ GeV".format(mu, sigma_fixed))

plt.xlabel(r"Reconstructed neutrino energy $E_{\mathrm{reco}}$ (GeV)")
plt.ylabel("Events per bin")
plt.title("Section 5: Effect of energy shift (varying $\\mu$, fixed $\\sigma$)")
plt.legend()
plt.grid(True)
plt.show()


def NLL_with_detector(theta23, dm_23_sq, mu, sigma):

    lambda_true = oscillated_prediction(E_center, unosc_array, theta23, dm_23_sq)
    lambda_reco = convolved_spectrum(E_center, lambda_true, mu, sigma)
    n = len(E_center)
    nll_sum = 0.0

    for i in range(n):
        lambda_i = float(lambda_reco[i])
        m_i = float(data_array[i])
        term = lambda_i - m_i * math.log(lambda_i)
        nll_sum += term

    return 2*nll_sum


def parabolic_minimiser_4D(theta_init, dm_init, mu_init, sigma_init, theta_step=0.02*np.pi, dm_step=0.1e-3, mu_step=0.01, sigma_step=0.01, theta_bounds=(0.0, 0.25*np.pi), dm_bounds=(0.0, 1.0e-2), mu_bounds=(-0.3, 0.3), sigma_bounds=(0.01, 0.40), tol_theta=1e-6, tol_dm=1e-7, tol_mu=1e-3, tol_sigma=1e-3, max_iter=100):

    theta = float(np.clip(theta_init, theta_bounds[0], theta_bounds[1]))
    dm = float(np.clip(dm_init, dm_bounds[0], dm_bounds[1]))
    mu = float(np.clip(mu_init, mu_bounds[0], mu_bounds[1]))
    sigma = float(np.clip(sigma_init, sigma_bounds[0], sigma_bounds[1]))

    for it in range(max_iter):

        # fix (dm, mu, sigma)，minimize θ 
        def f_theta(t):
            return NLL_with_detector(t, dm, mu, sigma)

        x0 = max(theta - theta_step, theta_bounds[0])
        x2 = min(theta + theta_step, theta_bounds[1])
        x1 = theta
        xs = sorted(set([x0, x1, x2]))
        theta_new, nll_theta, _ = parabolic_minimiser(f_theta, xs[0], xs[1], xs[2], tol=tol_theta, max_iter=100)


        # fix (theta_new, mu, sigma), minimize dm
        def f_dm(m):
            return NLL_with_detector(theta_new, m, mu, sigma)

        y0 = max(dm - dm_step, dm_bounds[0])
        y2 = min(dm + dm_step, dm_bounds[1])
        y1 = dm
        ys = sorted(set([y0, y1, y2]))
        dm_new, nll_dm, _ = parabolic_minimiser(f_dm, ys[0], ys[1], ys[2], tol=tol_dm, max_iter=100)

        # fix (theta_new, dm_new, sigma), minimize mu
        def f_mu(mu_val):
            return NLL_with_detector(theta_new, dm_new, mu_val, sigma)

        u0 = max(mu - mu_step, mu_bounds[0])
        u2 = min(mu + mu_step, mu_bounds[1])
        u1 = mu
        us = sorted(set([u0, u1, u2]))
        mu_new, nll_mu, _ = parabolic_minimiser(f_mu, us[0], us[1], us[2], tol=tol_mu, max_iter=100)


        # fix (theta_new, dm_new, mu_new), minimize sigma
        def f_sigma(sig_val):
            return NLL_with_detector(theta_new, dm_new, mu_new, sig_val)

        s0 = max(sigma - sigma_step, sigma_bounds[0])
        s2 = min(sigma + sigma_step, sigma_bounds[1])
        s1 = sigma
        ss = sorted(set([s0, s1, s2]))
        sigma_new, nll_sigma, _ = parabolic_minimiser(f_sigma, ss[0], ss[1], ss[2], tol=tol_sigma, max_iter=100)


        # check convergence
        d_theta = abs(theta_new - theta)
        d_dm = abs(dm_new - dm)
        d_mu = abs(mu_new - mu)
        d_sigma = abs(sigma_new - sigma)

        theta, dm, mu, sigma = theta_new, dm_new, mu_new, sigma_new

        if d_theta < tol_theta and d_dm < tol_dm and d_mu < tol_mu and d_sigma < tol_sigma:
            print("Converged after {} iterations.".format(it+1))
            break

    # result
    nll_best = NLL_with_detector(theta, dm, mu, sigma)
    print("θ23/π = {:.6f}".format(theta / np.pi))
    print("Δm²_23 = {:.6e} eV²".format(dm))
    print("mu = {:.3f} GeV".format(mu))
    print("sigma = {:.3f} GeV".format(sigma))
    print("NLL_min = {:.3f}".format(nll_best))

    return theta, dm, mu, sigma, nll_best


theta_start = theta_best_2d
dm_start = dm_best_2d
mu_start = -0.1     
sigma_start = 0.1    

theta_best_4d, dm_best_4d, mu_best_4d, sigma_best_4d, nll_best_4d = parabolic_minimiser_4D(theta_start, dm_start, mu_start, sigma_start)

#%%
lambda_true_best = oscillated_prediction(E_center, unosc_array, theta_best_4d, dm_best_4d)

lambda_reco_best = convolved_spectrum(E_center, lambda_true_best, mu_best_4d, sigma_best_4d)

plt.figure(figsize=(8, 5))

# data
plt.step(E_center, data_array, label="Data", linewidth=1.5)

# 4D best-fit with detector 
plt.step(E_center, lambda_reco_best, label="Prediction with detector effects", linewidth=1.5)

plt.xlabel("Neutrino energy E (GeV)")
plt.ylabel("Events per bin")
plt.title("Best-fit spectrum with detector effects")
plt.legend()
plt.grid(True)
plt.show()




