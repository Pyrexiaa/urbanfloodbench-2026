# Solution Write-up

## Overview

My final solution family was based on autoregressive LightGBM models trained separately for each `(model_id, node_type)` combination. I treated the task as a structured spatiotemporal forecasting problem where water-level evolution depends on:

- recent node state
- rainfall history and short-horizon future aggregates
- static hydraulic / terrain context
- graph-neighborhood behavior
- selective 1D-2D coupling

The strongest final variant available locally was `v2h`, which achieved a public leaderboard score of `0.0387`.

## Data Processing

### Node-type-specific handling

I handled `1D` and `2D` nodes separately throughout training and inference. This was important because the two node types exhibited different temporal scales, different variability, and different physically meaningful static attributes.

### Warmup handling

The first 10 timesteps were treated as warmup / observed history. A key improvement came from aligning training with autoregressive inference by excluding pre-warmup target transitions from training targets. This reduced train-inference mismatch and improved public leaderboard performance.

### Static feature preparation

For each node, I used static attributes together with graph-derived neighborhood summaries.

Examples:

- `1D`: position, depth, invert elevation, surface elevation, base area, pipe capacity, elevation range
- `2D`: position, area, roughness, minimum elevation, elevation, aspect, curvature, flow accumulation, elevation above local minimum

Graph features included:

- node degree
- mean / min / max / std of neighboring elevation

## Dynamic Features

For autoregressive prediction, I used the current water level and recent changes as the main state representation.

Key dynamic features included:

- current water level
- water level above elevation / invert elevation
- previous delta
- second previous delta
- change relative to warmup end
- neighboring water-level mean and difference
- warmup summary statistics

Warmup statistics included:

- last warmup water level
- mean
- std
- min
- max
- range

## Rainfall Features

Rainfall was one of the most important drivers in the solution. I engineered both backward-looking and forward-looking rainfall features, including:

- current rainfall
- cumulative rainfall
- rolling sums over multiple windows
- peak-so-far and time-since-peak
- rainfall derivative
- future rainfall sums (`rain_future_*`)

These features were especially important for `2D` behavior.

## Modeling Strategy

### Separate models by subset

I trained separate LightGBM models for:

- `Model 1 / 1D`
- `Model 1 / 2D`
- `Model 2 / 1D`
- `Model 2 / 2D`

This was necessary because the four subsets behaved differently enough that a single shared tabular model was not ideal.

### Target

The model predicted next-step water-level delta autoregressively. During inference, the predicted delta was added to the current predicted level, with clipping and physical lower-bound constraints.

### Noise injection

I added light Gaussian noise during training to improve autoregressive robustness.

### 2D subsampling

For tractability, I subsampled `2D` training rows while keeping validation deterministic.

## Hyperparameters

### Model 1

- learning rate: `0.05`
- num leaves: `127`
- feature fraction: `0.8`
- bagging fraction: `0.8`
- bagging frequency: `5`
- reg alpha: `0.1`
- reg lambda: `1.0`
- min child samples: `50`
- boosting rounds: `5000`

### Model 2

- learning rate: `0.03`
- num leaves: `63`
- feature fraction: `0.8`
- bagging fraction: `0.8`
- bagging frequency: `5`
- reg alpha: `0.5`
- reg lambda: `5.0`
- min child samples: `100`
- boosting rounds: `10000`

Common settings:

- objective: regression
- metric: RMSE
- early stopping: `100`
- GPU training enabled

## 1D-2D Coupling

A late but useful improvement came from using `1d2d_connections.csv` to inject conservative coupling features into `2D` prediction.

I tested multiple variants:

- broad coupling across all `2D`
- `Model 2 / 2D` only coupling
- different coupling feature subsets

The strongest direction was selective coupling on `Model 2 / 2D`.

Useful coupling features:

- connected `1D` current water level
- connected `1D` minus current `2D` gap

This was more reliable than broad inference-time postprocessing.

## Failed Direction

One major failed direction was a physically inspired `2D` inference-only postprocess. Although validation logs looked better, the public leaderboard degraded substantially. The main lesson was that the validation metric I was watching did not evaluate the exact final submitted prediction path once the postprocess was added. This made the apparent local improvement misleading.

## Final Submitted Variant

The strongest final local variant available to me was `v2h`, which combined:

- warmup-aligned training
- autoregressive LightGBM
- node-type / model-specific training
- rainfall history/future features
- graph-neighborhood features
- conservative `1D-2D` coupling only for `Model 2 / 2D`

Public leaderboard progression in the final stage:

- `v2a`: `0.0390`
- `v2g`: `0.0389`
- `v2h`: `0.0387`

## Auxiliary Variables

I did not predict auxiliary variables such as velocity, flow, or volume for submission.
