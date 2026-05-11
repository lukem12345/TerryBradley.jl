This repo contains different versions of the Bradley-Terry ranking model, and the related Thurstone-Mosteller model, and executes them on MLB data.

In particular, we have:
```julia
@model function bradley_terry_logspace(x, y, d)
    α ~ filldist(truncated(Normal(0.0, 1.0), 0.0, Inf), d)
    for i in 1:length(y)
        θ    = log(α[x[i, 1]]) - log(α[x[i, 2]])
        y[i] ~ BernoulliLogit(θ)
    end
end
```
,
```julia
@model function bradley_terry(x, y, d)
    α ~ filldist(truncated(Normal(0.0, 1.0), 0.0, Inf), d)
    for i in 1:length(y)
        α₁, α₂ = α[x[i, 1]], α[x[i, 2]]
        θ    = α₁ / (α₁ + α₂)
        y[i] ~ Bernoulli(clamp(θ, 0, 1))
    end
end
```

,
```julia
@model function bradley_terry_utility(x, y, d)
    α ~ filldist(truncated(Normal(0.0, 1.0), 0.0, Inf), d)
    for i in 1:length(y)
        θ    = α[x[i, 1]] - α[x[i, 2]]
        y[i] ~ BernoulliLogit(θ)
    end
end
```

, and
```julia
@model function thurstone_mosteller_utility(x, y, d)
    α ~ filldist(truncated(Normal(0.0, 1.0), 0.0, Inf), d)
    for i in 1:length(y)
        θ    = α[x[i, 1]] - α[x[i, 2]]
        y[i] ~ Bernoulli(cdf.(Normal(0.0, 1.0), θ))
    end
end
```

That is,

$\theta_i = \log \alpha_{x_{i,1}} - \log \alpha_{x_{i,2}},$
$y_i \sim \text{Bernoulli}(\text{logit}^{-1}(\theta_i)),$

$\theta_i = \frac{\alpha_{x_{i,1}}}{\alpha_{x_{i,1}} + \alpha_{x_{i,2}}},$
$y_i \sim \text{Bernoulli} \left(\text{clamp}(\theta_i, 0, 1)\right),$

$\theta_i = \alpha_{x_{i,1}} - \alpha_{x_{i,2}}$
$y_i \sim \text{Bernoulli} \left(\text{logit}^{-1}(\theta_i)\right),$

and

$\theta_i = \alpha_{x_{i,1}} - \alpha_{x_{i,2}}$
$y_i \sim \text{Bernoulli} \left(\Phi(\theta_i)\right).$

(The clamp function is not strictly necessarily in the continuous case, but is needed for numerical purposes. The `BernoulliLogit` function interprets its input as log-odds.)

The first of these was written by Damon C. Roberts to replicate a Stan model for benchmarking purposes. The second of these is the result of some experiments I was making on how to improve the sampling speed. It is algebraically equivalent to the first model; the `BernoulliLogit` function takes logodds as its argument.

The second and third models are written using the "utility" formulation of the parameters $a$, as opposed to the first two which interpret the parameters as the "playing strengths"/ "Spielstärken". This utility formulation of the Bradley-Terry model is much faster to sample. The Thurstone-Mosteller model is similar to the Bradley-Terry model, but performs probit regression. These two models should yield the same rankings given sufficient data.

Here are some quick benchmarks. Note that you can compare the effective sample size (ESS) per second and other relevant metrics by examing the output of `fit_model` in your REPL, calling `summarystats(fit)`. For the Bradely-Terry formulations, we use the NUTS sampler, and for the Thurstone-Mosteller model, we use Turing.jl's HMC sampler for autodiff compatibility reasons. The Thurstone-Mosteller model's timings are thus less directly comparable, but I include them here for completeness' sake.

`BTLogSpace:`
```julia
BenchmarkTools.Trial: 2 samples with 1 evaluation per sample.
 Range (min … max):  154.078 s … 175.401 s  ┊ GC (min … max): 0.53% … 0.50%
 Time  (median):     164.740 s              ┊ GC (median):    0.51%
 Time  (mean ± σ):   164.740 s ±  15.077 s  ┊ GC (mean ± σ):  0.51% ± 0.02%

  █                                                         █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  154 s           Histogram: frequency by time          175 s <

 Memory estimate: 11.22 GiB, allocs estimate: 38999567.
```

`BTLinSpace:`
```julia
BenchmarkTools.Trial: 5 samples with 1 evaluation per sample.
 Range (min … max):  25.879 s … 113.660 s  ┊ GC (min … max): 2.00% … 0.84%
 Time  (median):     73.835 s              ┊ GC (median):    0.85%
 Time  (mean ± σ):   67.010 s ±  37.719 s  ┊ GC (mean ± σ):  1.07% ± 0.57%

  █  █                           █          █              █
  █▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  25.9 s         Histogram: frequency by time          114 s <

 Memory estimate: 4.81 GiB, allocs estimate: 18934406.
```

`BTUtility:`
```julia
BenchmarkTools.Trial: 17 samples with 1 evaluation per sample.
 Range (min … max):  16.462 s … 20.888 s  ┊ GC (min … max): 2.16% … 1.99%
 Time  (median):     19.111 s             ┊ GC (median):    2.17%
 Time  (mean ± σ):   18.740 s ±  1.234 s  ┊ GC (mean ± σ):  2.35% ± 0.31%

  ▁ ▁    ▁    ▁     ▁         ▁   ▁▁█ █▁▁   ▁   ▁         ▁
  █▁█▁▁▁▁█▁▁▁▁█▁▁▁▁▁█▁▁▁▁▁▁▁▁▁█▁▁▁███▁███▁▁▁█▁▁▁█▁▁▁▁▁▁▁▁▁█ ▁
  16.5 s         Histogram: frequency by time        20.9 s <

 Memory estimate: 4.01 GiB, allocs estimate: 15994700.
```

`BTThurstoneMosteller:`
```julia
BenchmarkTools.Trial: 15 samples with 1 evaluation per sample.
 Range (min … max):  19.479 s …   21.961 s  ┊ GC (min … max): 1.33% … 1.92%
 Time  (median):     20.995 s               ┊ GC (median):    1.42%
 Time  (mean ± σ):   20.831 s ± 630.154 ms  ┊ GC (mean ± σ):  1.59% ± 0.27%

  ▁      ▁         ▁        ▁▁   ▁▁   ▁▁▁  █▁▁              ▁
  █▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁██▁▁▁██▁▁▁███▁▁███▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  19.5 s          Histogram: frequency by time           22 s <

 Memory estimate: 2.85 GiB, allocs estimate: 11938267.
```

... running on an M2 macbook air with 24 GB of RAM. The variance on these times is large (and the number of samples is small!) so some more benchmarks should be performed. There are also other tricks like swapping out the autodiff backend that can be performed. This is just an internal Turing.jl vs Turing.jl comparison, so it would be interesting to see the same re-parameterization techniques applied to a Stan model.

This original code for project is based off of Damon Roberts' [Python implementation](https://github.com/DamonCharlesRoberts/mlb_pred) of the Bradley-Terry model and the Julia port using Turing.jl discussed in [this blog post](https://blog.damoncroberts.io/posts/julia_nuts/). I am grateful for him open-sourcing the original benchmarking code. This code is released under the same license as the original `DamonCharlesRoberts/mlb_pred` repo.

This package offers a complete Julia pipeline that captures:
- accessing the MLB stats API
- creating the duck db
- learning the Terry-Bradley model
- post-processing ranks
- visualizing distributions.

This repo should not be considered a perfect reproduction of the original environment, e.g., ensuring the same game data is downloaded as in the past benchmarks, same visualization styles are used, etc.

<p align='center'>
<img src='./mlb_rankings_2025.svg' width='512' alt='Histograms of rankings for the 2025 season'>
</p>
