# GAN for OHLC Data Generation

This project implements a **Generative Adversarial Network (GAN)** that generates OHLC (Open, High, Low, Close) data for financial markets, designed to mimic the behavior of real market data. The generator produces synthetic financial data, while the discriminator evaluates whether the data is real or generated.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Getting Started](#getting-started)
4. [Model Details](#model-details)
5. [Training](#training)
6. [Usage](#usage)
7. [Results](#results)
8. [License](#license)

## Project Overview

This project uses a GAN to generate synthetic OHLC data. The generator takes random noise as input and learns to produce realistic sequences of OHLC data. The discriminator is trained to distinguish between real and fake data, encouraging the generator to improve. This method can be used to simulate market conditions for testing trading algorithms, backtesting strategies, or generating synthetic data for research purposes.

The data fluctuates between 0.9 and 1.1, reflecting small price changes that commonly occur in real-world markets.

## Requirements

The following libraries are required to run the project:

- Python 3.x
- `torch` (for PyTorch)
- `torchvision`
- `mplfinance` (for plotting candlestick charts)
- `pandas`
- `numpy`

You can install the dependencies using `pip`:

```bash
pip install torch torchvision pandas numpy mplfinance
