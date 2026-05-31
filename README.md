# SimulationCalibration

A neural network-based inverse optimization framework that learns the relationship between input parameters and system outputs, then uses mixed-integer programming to find optimal input configurations that achieve a desired target output.

## Overview

MetaOptimizer combines machine learning with mathematical optimization in a two-phase approach:

1. **Training phase** — A neural network is trained to approximate the forward mapping from input parameters to system outputs.
2. **Optimization phase** — The trained network's weights are embedded into a mixed-integer linear program (MILP), which is then solved to find the input parameters that best reproduce a given target output.

This is useful when you have a complex simulation or black-box system and want to work backwards from a desired outcome to the parameters that produce it.

## How It Works

The neural network uses a single hidden layer with ReLU activations. Once trained, the ReLU activations are linearized using binary indicator variables, converting the network into a MILP that can be solved exactly. The objective minimizes the sum of absolute differences between the network's predicted outputs and the target values.

## Requirements

```
tensorflow
numpy
pandas
gurobipy        # for Gurobi backend
ortools         # for OR-Tools backend
```

A valid [Gurobi license](https://www.gurobi.com/solutions/licensing/) is required if using the Gurobi solver. OR-Tools is open source and free.

## Usage

```python
from meta_trainer import MetaOptimizer, OptimizerType

# Initialize with your preferred solver backend
optimizer = MetaOptimizer(OptimizerType.OR_TOOLS)  # or OptimizerType.GUROBI

# Train on your data
optimizer.train_model(X_training, Y_training, X_validation, Y_validation)

# Define bounds for input parameters
lower_bound = [1.0, 0.5, 5.0, 1.0, 0.4, 390.0]
upper_bound = [2.5, 1.0, 10.0, 5.0, 0.8, 760.0]

# Find inputs that match a target output
parameters, predicted_outputs = optimizer.solve_optimization(
    target_val, lower_bound, upper_bound
)
```

## Data Format

- **Inputs (`X`)** — shape `(n_samples, n_inputs)`, the controllable parameters
- **Outputs (`Y`)** — shape `(n_samples, n_outputs)`, the system responses

The notebook demonstrates a 6-input → 9-output problem with min-max normalization applied to the outputs before training.

## Solver Backends

| Backend | Class | Notes |
|---|---|---|
| OR-Tools (SCIP) | `OptimizerType.OR_TOOLS` | Free, open source |
| Gurobi | `OptimizerType.GUROBI` | Commercial license required, typically faster |

Both backends use a MIP gap of `1e-6` for solution quality.

## File Structure

```
├── 00_meta_trainer.ipynb   # Main notebook with MetaOptimizer class and usage examples
└── README.md
```

## Limitations

- Solve time scales with the number of hidden nodes and input/output dimensions.
- The network must be retrained if the underlying system changes.
