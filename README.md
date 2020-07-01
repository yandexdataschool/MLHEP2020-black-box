# XFEL

XFEL accelerator tuning competition.

## Installation

### via git

```sh
git clone git@gitlab.com:lambda-hse/mlhep-2020-fel-competition.git fel
cd fel
pip install -e .
```

This package can be easily updated via `git pull` (due to `-e` option).

### one liner

```sh
pip install git+https://gitlab.com/lambda-hse/mlhep-2020-fel-competition.git
```

## Example usage

```python
import numpy as np
from fel import SASE, random_beam, random_geometry

hidden_rng = np.random.RandomState(1111)

sase = SASE(random_beam(hidden_rng), random_geometry(hidden_rng))
epsilon = 1e-12

objective = lambda x: np.log(1e-3) - np.log(sase.rho_int(x) + epsilon)

bounds = np.stack([
    -2 * np.ones(sase.ndim()),
    2 * np.ones(sase.ndim())
], axis=1)

x0 = np.zeros(sase.ndim())
```