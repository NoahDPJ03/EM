## Expectation Maximization: Responsibility Method vs. MCMC Approximation

**Situation**
In statistical modeling, the Expectation Maximization (EM) algorithm is a key tool for estimating parameters of mixture models. However, when the *responsibility* values $P(z_i | x_i, \theta)$ are computationally expensive or analytically intractable, practitioners often turn to Monte Carlo (MC) approximations. I set out to implement and compare the **standard responsibility-based EM** with a **custom MCMC-based EM approximation**, evaluating their performance on mixtures of both continuous and discrete probability distributions.

My goal was to:

1. Implement both EM strategies from scratch in Python.
2. Create controlled synthetic datasets with known distribution parameters and mixture weights.
3. Compare the accuracy, stability, and convergence behavior of the two methods.
4. Explore performance differences between discrete and continuous mixture cases.


* **Implementation**:

  * Built the **Responsibility Method (Standard EM)** using exact probability computations for the E-step and weighted parameter updates in the M-step.
  * Developed the **MCMC Approximation Method** where responsibilities were approximated via **Markov Chain Monte Carlo sampling** with Metropolis-Hastings acceptance.
  * Incorporated safeguards such as **minimum component weights** to prevent component collapse in MCMC updates.
* **Data Generation**:

  * Designed a random mixture generator supporting both continuous (Normal, Uniform, Exponential) and discrete (Poisson, Bernoulli) distributions.
  * Automated random selection of mixture size (2–8 components), distribution types, and parameters within defined ranges.
* **Evaluation**:

  * Ran multiple trials for both continuous and discrete mixtures, tracking log-likelihood progression and absolute differences in estimated mixture weights between methods.
  * Visualized results with convergence plots and difference-scatter charts.


* **Accuracy**: Both methods consistently produced **very similar mixture weights and parameters** across trials, with differences showing no consistent pattern.
* **Performance**: Standard EM converged more quickly and required significantly less computation. MCMC-EM introduced variability, especially in discrete cases, but remained viable when analytic responsibility computation was not feasible.
* **Insights**:

  * Continuous components tended to dominate in mixed continuous–discrete datasets.
  * Discrete-only mixtures often converged faster than continuous ones.
  * MCMC-EM is more computationally intensive and carries a higher risk of instability, but is a reasonable fallback method when the standard EM E-step is intractable.

**Key Takeaways**
This project demonstrated that while Monte Carlo approximations can stand in for the responsibility-based EM, they come at a computational cost and may not always yield better results. Still, having both methods implemented from scratch provides flexibility in handling a wide variety of mixture modeling problems.

Note: This is a work-in-progress project - please excuse minor mistakes, typos, etc. in the code and/or writeup.

This Readme was made with the help of ChatGPT-5.
