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

$\alpha_k \sim \mathcal{HN}(0, 1, [0, \infty)), \quad k = 1, \ldots, d,$

$\theta_i = \log \alpha_{x_{i,1}} - \log \alpha_{x_{i,2}},$

$y_i \sim \text{Bernoulli}(\text{logit}^{-1}(\theta_i)),$

and

$\alpha_k \sim \mathcal{HN}(0, 1, [0, \infty)), \quad k = 1, \ldots, d,$

$\theta_i = \frac{\alpha_{x_{i,1}}}{\alpha_{x_{i,1}} + \alpha_{x_{i,2}}},$

$y_i \sim \text{Bernoulli} \left(\text{clamp}(\theta_i, 0, 1)\right).$

(The clamp function is not strictly necessarily in the continuous case, but is needed for numerical purposes.)

The first of these was written by Damon C. Roberts to replicate a Stan model for benchmarking purposes. The second of these is the result of some experiments I was making on how to improve the sampling speed. It is algebraically equivalent to the first model; the `BernoulliLogit` function takes logodds as its argument.

The second and third models are written using the "utility" formulation of the parameters $a$, as opposed to the first two which interpret the parameters as the "playing strengths"/ "Spielstärken". This utility formulation of the Bradley-Terry model is much faster to sample. The Thurstone-Mosteller model is similar to the Bradley-Terry model, but performs probit regression. These two models should yield the same rankings given sufficient data.

Some quick benchmarks (just using `@time`):
Original (log-space):
```julia
BenchmarkTools.Trial: 2 samples with 1 evaluation per sample.
 Range (min … max):  184.426 s … 220.521 s  ┊ GC (min … max): 0.67% … 0.87%
 Time  (median):     202.474 s              ┊ GC (median):    0.78%
 Time  (mean ± σ):   202.474 s ±  25.523 s  ┊ GC (mean ± σ):  0.78% ± 0.14%

  █                                                         █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  184 s           Histogram: frequency by time          221 s <

 Memory estimate: 18.62 GiB, allocs estimate: 191333371.
```

New (linear-space):
```julia
BenchmarkTools.Trial: 4 samples with 1 evaluation per sample.
 Range (min … max):  24.974 s … 113.788 s  ┊ GC (min … max): 2.41% … 1.17%
 Time  (median):     86.196 s              ┊ GC (median):    1.24%
 Time  (mean ± σ):   77.788 s ±  39.084 s  ┊ GC (mean ± σ):  1.35% ± 0.60%

  █                              █                █        █
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁█ ▁
  25 s           Histogram: frequency by time          114 s <

 Memory estimate: 9.19 GiB, allocs estimate: 93300551.
```

... running on an M2 macbook pro with 24 GB of RAM. The variance on these times is large (and the number of samples is small!) so some more benchmarks should be performed. There are also other tricks like swapping out the autodiff backend that can be performed. This is just an internal Turing.jl vs Turing.jl comparison, so it would be interesting to see the same re-parameterization applied to a Stan model.

This original code for project is based off of Damon Roberts' [Python implementation](https://github.com/DamonCharlesRoberts/mlb_pred) of the Bradley-Terry model and the Julia port using Turing.jl discussed in [this blog post](https://blog.damoncroberts.io/posts/julia_nuts/). I am grateful for him open-sourcing the original benchmarking code. This code is released under the same license as the original `DamonCharlesRoberts/mlb_pred` repo.

I wanted a complete Julia pipeline that captures:
- accessing the MLB stats API
- creating the duck db
- learning the Terry-Bradley model
- post-processing ranks
- visualizing distributions.

This repo should not be considered a perfect reproduction of the original environment, e.g., ensuring the same game data is downloaded as in the past benchmarks, same visualization styles are used, etc.

<p align='center'>
<img src='./mlb_rankings_2025.svg' width='512' alt='Histograms of rankings for the 2025 season'>
</p>
