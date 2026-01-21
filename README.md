# Neutrino Oscillation NLL Fit (T2K-like spectrum)

This project implements a **Poisson negative log-likelihood (NLL)** fit to extract neutrino oscillation parameters from a simulated T2K-like νμ event spectrum binned in energy.

Workflow:
1) Build an oscillated prediction from an unoscillated spectrum and a survival probability model  
2) Compute Poisson NLL across energy bins  
3) Minimise NLL (1D / 2D) to obtain best-fit parameters and uncertainties  
4) (Optional) Include simplified detector effects via a Gaussian response model and perform a 4D fit

---

## Key features

- **Muon neutrino survival probability**
  - Two-flavour survival model P(νμ → νμ) vs energy

- **Oscillated event-rate prediction**
  - Predicts λᵢ(θ23, Δm²23) from an unoscillated spectrum

- **Poisson negative log-likelihood**
  - Standard Poisson NLL for binned counts (constants dropped)

- **Minimisation**
  - 1D parabolic minimiser for θ23 (with fixed Δm²23)
  - 2D minimisation via alternating 1D updates (coordinate-descent style)

- **Uncertainty estimation**
  - ΔNLL = 1 scan-based (1σ) errors
  - curvature-based error estimate from a local quadratic fit

- **Detector effects (optional extension)**
  - Gaussian response matrix R(E_true → E_reco)
  - Spectrum convolution with tunable μ (bias) and σ (resolution)
  - 4D fit: (θ23, Δm²23, μ, σ)

---

## Repository structure

- Main_Code.py        # main code
- Project_1.pdf       # project report
- data                # dataset
