# SpaceHub Calc

A Python library for analyzing and visualizing SpaceHub N-body simulation outputs.

## Overview

SpaceHub Calc provides tools for computing Keplerian orbital elements from SpaceHub simulation data, including:

- Two-body orbital analysis (eccentricity, semi-major axis, inclination, etc.)
- N-body trajectory visualization
- Gravitational wave inspiral calculations
- Interactive 3D animations

## Features

- **TwoBodyOrbit**: Compute all Keplerian orbital elements from simulation output
- **NBodyVisualizer**: Visualize multi-body simulations with customizable reference frames
- **Theorize**: Theoretical calculations for gravitational wave decay
- Support for SpaceHub CSV output format
- Interactive matplotlib animations
- Find orbital parameter crossings with high precision

## Installation

```bash
pip install spacehub-calc
```

## Quick Start

```python
from spacehub_calc import TwoBodyOrbit

# Load simulation data and compute orbital elements
orbit = TwoBodyOrbit("simulation_output.csv")

# Plot orbital evolution
fig, axs = orbit.plot_keplerian_evolution(plots=('e', 'a', 'i', 'R'))

# Find when semi-major axis crosses 0.8 AU
time, value = orbit.find_crossing('a', 0.8)
```

## Requirements

- numpy
- pandas
- matplotlib
- scipy
- astropy
- seaborn

## License

MIT License

## Author

Nicholas Marston (ntmarston)
