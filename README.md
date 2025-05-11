# HMMs-withJumps

This Julia project provides a framework for simulating stock market returns using a hybrid modeling approach that combines Hidden Markov Models (HMMs) with a Poisson jump process. It leverages daily ticker data including Open, High, Low, Close (OHLC), Volume, and Volume Weighted Average Price (VWAP).

## Overview

Financial markets exhibit complex behaviors such as volatility clustering, heavy-tailed distributions, and regime-dependent dynamics, making classical models inadequate. This project addresses these challenges by:

* Implementing discrete-state HMMs to model distinct market regimes (e.g., favorable, unfavorable market conditions).
* Integrating a Poisson jump process to account for sudden, significant market movements.
* Using empirical data calibration for realistic modeling of state transitions.

## Features

* **Modular pipeline** for constructing HMM-based return models for any given ticker.
* Simulation capabilities reflecting real-world statistical features including:

  * Volatility clustering
  * Heavy-tailed return distributions
  * Absence of autocorrelation
* Robust validation framework using Kolmogorov-Smirnov tests.

## Data

The current implementation is validated with a comprehensive dataset covering:

* Over 400 U.S.-listed equities and ETFs
* Historical data spanning 2,515 trading days
* Representative tickers such as NVDA, AAPL, and SPY

## Getting Started

### Prerequisites

* Julia (version 1.9 or higher recommended)
* Required packages:

  ```julia
  using Pkg
  Pkg.add(["CSV", "DataFrames", "Distributions", "StatsBase", "Plots", "HMMBase"])
  ```

### Running Simulations

To run simulations, execute the main script with appropriate parameters:

```julia
include("main.jl")
```

Adjust the parameters in the provided configuration file as needed for your specific tickers and simulation scenarios.

## Validation

In-sample and out-of-sample validations were performed using standard statistical tests. Results confirmed high accuracy in replicating historical data properties.

## Contributing

Contributions to improve functionality or extend capabilities are welcome. Please create pull requests with clear explanations of proposed enhancements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
