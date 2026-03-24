## 6th Place Solution — UrbanFloodBench

### Overview

Our final solution was a **stacked autoregressive system** rather than a single end-to-end model. The idea was to keep the long-horizon rollout simple and stable, then add learned corrections only where they were most useful.

The pipeline had four layers:

1. **Linear autoregressive baselines** for both urban models.
2. **Residual GRU models** that corrected those baseline rollouts.
3. **Auxiliary forecasts of hidden hydraulic states** such as inlet flow, local surface exchange, edge flow, and connected storage.
4. A final **gating model** that blended two complementary Model 2 1D stacks.

This design worked well because the competition contained two different problems: a large, smoother **2D surface system** and a smaller but harder **1D drainage system** with strong local coupling to the surface. Most of our gains came from modeling the 1D part more carefully while keeping the rollout engine itself robust.

### Core strategy

We treated the task as a **stable forecasting problem first** and a **representation learning problem second**.

Instead of making one large graph model autoregress over all nodes, we used strong per-node ARX baselines as the backbone. Those baselines already captured much of the temporal structure and were reliable over long rollouts. We then trained residual models to predict the remaining error sequence on top of those trajectories.

This had three advantages:

* it reduced long-horizon drift,
* it made stacking easy,
* and it let us spend model capacity where the baselines were weakest.

In practice, the weak point was **Model 2 1D**, not 2D. That shaped almost every later design choice.

### Baseline models

For **Model 1**, we used a simple per-node **AR(1)+rain+cum_rain** baseline. It was already strong enough that only a light residual stack was needed afterward.

For **Model 2**, we used a family of **regime-specific AR(10)+X** baselines, where the regime was chosen by rainfall bins. We found it useful to split 1D and 2D because they behaved differently.

The most important baselines were:

* **Baseline A:** a non-split regime AR(10)+X over all Model 2 nodes.
* **Baseline B:** separate 1D and 2D models, with event-equalized fitting and a more stable 1D rollout.
* **Baseline D:** a coupled version where 1D prediction used predicted inlet flow.
* **Baseline G:** a locally coupled baseline using the connected 2D cell, nearby 2D neighbors, and inlet behavior.

These baselines were trained in closed form with ridge regularization and became the foundation for all later stacks.

### Data and feature design

We kept the target in the original **water level** units, but most learned features were expressed in more physical **depth-like coordinates**:

* for 1D nodes: water level relative to invert elevation,
* for 2D cells: water level relative to bed elevation.

We also made heavy use of the explicit **1D–2D coupling table**. Many 1D features were built by aggregating predicted 2D states back to each drainage node or by looking at the specific connected 2D cell and its neighborhood.

Although rainfall was effectively spatially uniform, we still used local rainfall and area-weighted rain-volume proxies inside the coupled features.

A key detail was that we matched the competition metric locally: node-wise standardized RMSE, equal weighting across 1D vs 2D, then events, then the two urban models. That mattered for both tuning and baseline fitting.

### Model 1 pipeline

The final **Model 1** branch was straightforward:

1. AR1X baseline,
2. 2D residual GRU,
3. 1D residual GRU conditioned on corrected 2D information aggregated back to the 1D nodes.

Model 1 was not the main source of leaderboard gains, so we kept it simple and stable. We did not ensemble or gate Model 1 in the final submission.

### Model 2: 2D-first, then 1D correction

#### 2D residual model

The 2D branch was corrected first. We trained a GRU to predict the residual between the true 2D water level and the baseline rollout.

The basic 2D features were depth, depth change, rainfall, cumulative rainfall, and normalized time. Our strongest final 2D model used an additional **`v3cpl`** feature set that injected sparse coupling information from the connected 1D system back onto coupled 2D cells. In effect, the 2D model could see a proxy for subsurface pressure differences at the coupled locations.

This final 2D model used a **nodewise head**, **EMA during training**, and no output clamp.

#### 1D residual stack

The 1D part was much more complex. We did not rely on one 1D model. Instead, we used a **stack of residual models**:

* each stage took the current full 1D trajectory,
* predicted a residual sequence,
* added it back,
* and passed the updated trajectory to the next stage.

This kept the system stable while allowing later models to specialize.

The first stage was a broad 1D correction model using 1D depth, connected 2D depth, gap features, rainfall, and pipe-neighbor information. Stronger versions also used post-GRU graph mixing and nodewise output heads.

Later stages added richer hydraulic features, including:

* local 2D KNN summaries,
* weighted incoming and outgoing hydraulic gradients on the pipe graph,
* surcharge-style features,
* rain lags,
* local 2D neighborhood states,
* slot-based local surface-flow features around the coupled 2D cell.

These richer features mattered much more than simply making the network deeper.

### Auxiliary hydraulic-state models

A major part of the solution was predicting hidden states that are only observed during warmup and then reusing them as exogenous features.

We trained lightweight autoregressive auxiliary models for:

* **1D inlet flow**,
* **1D edge flow**,
* **2D surface edge flow**,
* **connected 2D storage volume**.

These models were simple, but they exposed information that is hard to infer from node water levels alone. For example, the 1D models benefited substantially from predicted inlet flow and local surface-transport features near the coupled 2D cell.

### The two final Model 2 branches

Our final submission used two different full Model 2 1D stacks.

#### Branch A: targeted global stack

This was the stronger global branch. It used:

* a weighted mixture of **Baseline A and Baseline B**,
* the strong **2D residual model** with `v3cpl`,
* the best **stage 1 1D residual**,
* the best **rich stage-7 1D residual**,
* and a final **targeted global 1D booster** that used predicted inlet flow (`v12` family).

This branch was globally very strong and became the “best” side of the final gate.

#### Branch B: local G-based stack

The second branch started from **Baseline G**, which had a stronger local coupling bias. On top of that, we trained a 1D residual chain based on the **`v19`** feature family, which adds slot-based local surface-flow features around the coupled 2D cell.

This branch was not better everywhere, but it was better on some nodes and phases of the event where local surface exchange dominated the error.

### Final gate

The final submission did not just average Branch A and Branch B. We trained a small **gate model** for **Model 2 1D** only.

Given the two branch forecasts, the gate predicted:

* a blend weight between them,
* and a small additive correction.

Its inputs included:

* both 1D forecasts,
* their first differences and disagreement,
* aggregated 2D depth from both branches,
* predicted inlet flows,
* normalized fill ratios,
* rainfall features,
* normalized time,
* and simple static node descriptors.

The gate was deliberately tiny. Its job was not to model flood physics from scratch, but to decide which full stack to trust more at each node and timestep. That final learned blend gave us a consistent gain over any single branch or fixed average.

### Training procedure

We first tuned architectures and epoch counts on a held-out split using a local metric aligned to the leaderboard. After selecting the final recipe, we retrained the chosen models on the **full training set**.

Most strong residual models used:

* **EMA = 0.999**,
* **no output clamp** in the final full-train runs,
* and small, specialized architectures rather than one large universal model.

### Final submission

The final submitted system was:

* **Model 1:** baseline A + Model 1 2D residual + Model 1 1D residual.
* **Model 2 2D:** the Branch A 2D prediction.
* **Model 2 1D:** a learned gate between the targeted global Branch A and the local G-based Branch B.

In short, our solution worked because it combined:

* **stable linear autoregressive rollouts**,
* **2D-first correction**,
* **strong local 1D–2D coupling features**,
* **auxiliary predictions of hidden hydraulic states**,
* and a **final gate between two complementary full stacks**.

