"""
    Model

Defines the Bradley-Terry model, fits it via NUTS, and extracts
posterior rankings and alpha samples.
"""
module Model

export fit_model, gen_ids, BTLinSpace, BTLogSpace, BTUtility, BTThurstoneMosteller

using ..MLBDataIngest

using AbstractMCMC
using DataFrames
using DuckDB
using Logging
using MCMCChains
using Turing

# Sampler settings.
const CHAINS    = 4
const ITER_WARM = 1_000
const ITER_SAMP = 3_000

"""
    gen_ids(df)

Returns a `Dict{String, Int}` mapping each team abbreviation to a
stable 1-based integer ID derived from both the home and away columns.
"""
function gen_ids(df)
    teams = unique(vcat(df.home_abbr, df.away_abbr))
    return Dict(t => i for (i, t) in enumerate(teams))
end

abstract type AbstractBradleyTerryModel end

struct BTLinSpace           <: AbstractBradleyTerryModel end
struct BTLogSpace           <: AbstractBradleyTerryModel end
struct BTUtility            <: AbstractBradleyTerryModel end
struct BTThurstoneMosteller <: AbstractBradleyTerryModel end

bradley_terry(::BTLinSpace, args...) =
  bradley_terry_linspace(args...)
bradley_terry(::BTLogSpace, args...) =
  bradley_terry_logspace(args...)
bradley_terry(::BTUtility, args...) =
  bradley_terry_utility(args...)
bradley_terry(::BTThurstoneMosteller, args...) =
  thurstone_mosteller_utility(args...)

"""
    bradley_terry_linspace(x, y, d)

Turing model for a Bradley-Terry paired comparison.

Unlike the reference model, this one operates directly in linear space, and
requires a clamp to keep θ in the range [0,1]. The logit used in the reference
model does this in a smooth fashion, but is slower.

This model and bradley_terry_logspace are algebraically equivalent, aside from the priors. (Recall that BernoulliLogit takes the log-odds as its parameter.)

Args:
    x : N×2 integer matrix of (home_id, away_id) per game.
    y : N-vector of outcomes (1 = home win, 0 = away win).
    d : Number of teams (length of the α vector).
"""
@model function bradley_terry_linspace(x, y, d)
    α ~ filldist(truncated(Normal(0.0, 1.0), 0.0, Inf), d)
    for i in 1:length(y)
        α₁, α₂ = α[x[i, 1]], α[x[i, 2]]
        θ    = α₁ / (α₁ + α₂)
        y[i] ~ Bernoulli(clamp(θ, 0, 1))
    end
end

"""
    bradley_terry_logspace(x, y, d)

Turing model for a Bradley-Terry paired comparison.

This is the reference model given by Damon C. Roberts in his blog post on Stan
vs. Turing.jl benchmarking.

Args:
    x : N×2 integer matrix of (home_id, away_id) per game.
    y : N-vector of outcomes (1 = home win, 0 = away win).
    d : Number of teams (length of the α vector).
"""
@model function bradley_terry_logspace(x, y, d)
    α ~ filldist(truncated(Normal(0.0, 1.0), 0.0, Inf), d)
    for i in 1:length(y)
        θ    = log(α[x[i, 1]]) - log(α[x[i, 2]])
        y[i] ~ BernoulliLogit(θ)
    end
end

# These α are "utilities", unlike the playing "strengths"/ "Spielstaerken" α that the previous
# models were using.
@model function bradley_terry_utility(x, y, d)
    α ~ filldist(truncated(Normal(0.0, 1.0), 0.0, Inf), d)
    for i in 1:length(y)
        θ    = α[x[i, 1]] - α[x[i, 2]]
        y[i] ~ BernoulliLogit(θ)
    end
end

# This also uses utility α.
# It just has a prior that the α are normally distributed,
# i.e., it performs probit regression where BT performs logit regression.
# In the limit of more data, the BT and TM models will converge. This
# choice is only impactful for data-scarce scenarios.
@model function thurstone_mosteller_utility(x, y, d)
    α ~ filldist(truncated(Normal(0.0, 1.0), 0.0, Inf), d)
    for i in 1:length(y)
        θ    = α[x[i, 1]] - α[x[i, 2]]
        y[i] ~ Bernoulli(cdf.(Normal(0.0, 1.0), θ))
    end
end

"""
    rank_teams(fit, ids)

Converts posterior α samples into a long-format DataFrame of per-draw
rankings. Columns: iter, Rank, Team, chain.
"""
function rank_teams(fit, ids)
    iters   = size(fit, 1)
    chains  = size(fit, 3)
    samples = MCMCChains.group(fit, :α).value

    rank_arr = Array{Integer, 3}(undef, iters, length(ids), chains)

    for c in 1:chains
        for i in 1:iters
            current_sample = samples[i, :, c]
            ranked_indices = sortperm(current_sample, rev = true)
            for rank_idx in 1:length(ids)
                rank_arr[i, ranked_indices[rank_idx], c] = rank_idx
            end
        end
    end

    df = DataFrame()
    for c in 1:chains
        for (key, value) in ids
            temp_df = DataFrame(
                iter  = collect(iters:-1:1),
                Rank  = rank_arr[:, value, c],
                Team  = fill(string(key), iters),
                chain = fill(c, iters))
            append!(df, temp_df; promote = true)
        end
    end

    return df
end

"""
    fit_model(df, ids)

Fits the Bradley-Terry model and returns a named tuple with:
    fit    : raw Turing Chains object.
    ranks  : long-format DataFrame of posterior rankings.
"""
function fit_model(df, ids, BTM::T, sampler::U, iter_samp=ITER_SAMP) where {T <: AbstractBradleyTerryModel, U <: AbstractMCMC.AbstractSampler}
    home = [ids[team] for team in df.home_abbr]
    away = [ids[team] for team in df.away_abbr]
    mod  = bradley_terry(BTM, hcat(home, away), df.home_win, length(ids))
    fit  = Turing.sample(mod, sampler, MCMCThreads(), ITER_WARM + iter_samp, CHAINS)
    return (fit = fit, ranks = rank_teams(fit, ids))
end

end # module Model
