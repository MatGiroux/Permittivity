# Permittivity Calculation


This repository contains a Python script for calculating the permittivity of Si and InAs, as described in [Giroux 2024](
https://doi.org/10.48550/arXiv.2412.10217).


## Features


### Doping Level and Temperature
The user can select the temperature and doping levels by changing the values of the `T`, `NA`, `ND`, `N`, and `N_doped` parameters in the **User Variables** section of `main.py`, located near the top of the file, following the imports and function definitions.

### Model Selection
#### Kramers Kronig Relations
This code implements two models for the dielectric function of InAs. As in [Milovich 2020](https://doi.org/10.1117/1.JPE.10.025503), the interband absorption coefficient $\alpha_\mathrm{IB}(\omega)$  gives the imaginary part of the index of refraction $k_\mathrm{IB}(\omega)$. The Kramers-Kronig relation can then be used to give the real part of the index $n_\mathrm{IB}(\omega)$, which then gives a contribution to the dielectric. 

This model deviates from the model of [Milovich 2020](https://doi.org/10.1117/1.JPE.10.025503) by using an ionized-impurity form for the free-carrier contribution to $\varepsilon''(\omega)$. The contribution of this model to $\varepsilon'(\omega)$ can then be calculated using the Kramers-Kronig relations,

Since the Kramers-Kronig relations are more computationally expensive than the rest of the model, we also implement a simpler approximate model in which $n_\mathrm{IB}(\omega) = \sqrt{\varepsilon_\infty}$ and in which the free-carrier contribution to $\varepsilon'(\omega)$ is simply that from the Drude model.

The choice to use the Kramers-Kronig forms or the approximate forms is controlled by the `model_choice` parameter in `main.py`.

#### Carrier Concentration Calculation

This code provides two options for approximating the carrier concentration. The first option sets the carrier concentration to the doping level, while the second option computes the carrier concentration by considering the contribution of thermally excited carriers.

The choice between computing the carrier concentration or setting it to the doping level is controlled by the `N_calc_choice` parameter in `main.py`.

## Requirements

To run this script, Python 3.7 or a more recent version is required. Additionally, the following Python packages are required:

- `matplotlib`
- `numpy`
- `scipy`
- `sympy`
- `pint`