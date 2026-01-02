# Quantitative-analysis-of-HIV-in-Cuba
This repository contains the implementation of mathematical models to analyze the HIV epidemic in Cuba, including SIR-based extensions (XYZ and SXYZ models). The code explores the limitations of classical compartmental models and evaluates their ability to simultaneously describe HIV and AIDS dynamics.
The main objective of this work is to evaluate the strengths and limitations of deterministic models when applied to long-term infectious diseases, particularly those with slow progression such as HIV.

Motivation:
Modeling HIV dynamics presents unique challenges due to the long latency period between infection and the onset of AIDS, as well as the influence of social behavior, medical treatments, and public health policies.
This project aims to:
Understand how classical epidemiological models behave when applied to HIV.
Analyze why standard models struggle to represent HIV and AIDS simultaneously.
Explore improved modeling approaches by extending the classical SIR structure.

Models Implemented:
1. XYZ Model
A simplified compartmental model where:
X represents individuals infected with HIV.
Y represents individuals who have progressed to AIDS.
Z represents deaths describing AIDS-related mortality.
This model serves as an initial approximation but shows important limitations, particularly in reproducing realistic AIDS dynamics.
2. SXYZ Model
An extended version of the XYZ model that includes:
A susceptible population (S).
Population growth dynamics modeled using a logistic function.
Infection driven by contact between susceptible and infected individuals.
Although this model improves the representation of AIDS cases, it reveals structural limitations that prevent the simultaneous accurate modeling of both HIV and AIDS.

Key Findings
Classical SIR-based models are useful for understanding general epidemic behavior but struggle with long-term diseases such as HIV.
The strong dependency between HIV and AIDS compartments limits the model’s flexibility.
Constant parameters fail to represent real-world changes such as medical advancements or behavioral shifts.
A more realistic approach would require time-dependent parameters or additional disease stages.

Methodology
Ordinary Differential Equations (ODEs) were used to describe compartment interactions.
Parameters were estimated using iterative numerical methods.
Real epidemiological and population data from Cuba were used for calibration.
All simulations were implemented in Python.

Future Work
Possible improvements include:
Introducing time-dependent parameters.
Adding intermediate stages of HIV infection.
Incorporating treatment and behavioral factors.
Exploring hybrid or data-driven modeling approaches.
