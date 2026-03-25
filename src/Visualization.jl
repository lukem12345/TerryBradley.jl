"""
    Visualization

Plotting functions for Bradley-Terry posterior output.

Functions:
    plot_ranks  : Histogram grid of posterior rank distributions.
    summary     : Prints a median rank table with credible intervals.
"""
module Visualization

using DataFrames
using Statistics
using Plots
using Logging

const BG      = RGB(1,    1,    1)
const FG      = RGB(0.25, 0.22, 0.18)
const BARFILL = RGB(0.4,  0.4,  0.4)
const GRID_C  = RGB(0.6,  0.6,  0.6)

"""
    _histogram_grid(data, value_col, ordered_teams; xlabel, bins, xlims, xflip, title)

Internal helper that builds a single-column histogram grid shared across teams.
"""
function _histogram_grid(data, value_col, ordered_teams; xlabel, bins, xlims, xflip, title)
    n_teams   = length(ordered_teams)
    gridlines = range(xlims[1], xlims[2]; step = (xlims[2] - xlims[1]) / 5)

    plots = []
    for (i, team) in enumerate(ordered_teams)
        subset = filter(r -> r.Team == team, data)
        vals   = Float64.(subset[!, value_col])
        med    = median(vals)
        is_last = i == n_teams

        p = histogram(
            vals,
            bins             = bins,
            fillcolor        = BARFILL,
            fillalpha        = 0.85,
            linecolor        = :white,
            linewidth        = 0.4,
            legend           = false,
            ylabel           = team,
            labelfontsize    = 7,
            tickfont         = font("Courier New", 7),
            xticks           = is_last ? (collect(gridlines), string.(round.(Int, gridlines))) : false,
            yticks           = false,
            xlims            = xlims,
            xflip            = xflip,
            grid             = false,
            background_color_inside = BG,
            background_color = BG,
            foreground_color = FG,
            top_margin       = 0Plots.mm,
            bottom_margin    = 0Plots.mm,
            left_margin      = 12Plots.mm,
            right_margin     = 2Plots.mm)

        # Shared reference gridlines — visually continuous across panels.
        vline!(p, collect(gridlines),
            color     = GRID_C,
            linewidth = 0.5,
            linestyle = :dot,
            legend    = false)

        # Median marker on top of gridlines.
        vline!(p, [med],
            color     = FG,
            linewidth = 1.5,
            linestyle = :solid,
            legend    = false)

        push!(plots, p)
    end

    plt = plot(plots...,
               layout           = (n_teams, 1),
               size             = (500, 1800),
               dpi              = 180,
               plot_title       = title,
               plot_titlefont   = font("Times New Roman", 16),
               background_color = BG)
end

"""
    plot_ranks(ranks, season)

Plots a histogram grid of posterior rank distributions, one panel per team,
sorted best to worst. Saves to `mlb_rankings_<season>.png/.svg`.
"""
function plot_ranks(ranks::DataFrame, season::String)
    team_order = combine(groupby(ranks, :Team), :Rank => median => :med)
    sort!(team_order, :med)
    ordered_teams = team_order.Team

    plt = _histogram_grid(ranks, :Rank, ordered_teams;
                          xlabel  = "Rank",
                          bins    = 1:31,
                          xlims   = (1, 30),
                          xflip   = true,
                          title   = "MLB Power Rankings $season")
    outpath = "mlb_rankings_$season"
    savefig(plt, outpath * ".png")
    @info "Saved to $outpath.png"
    savefig(plt, outpath * ".svg")
    @info "Saved to $outpath.svg"
end

"""
    summary(ranks)

Prints a table of median rank with 5th/95th percentile credible intervals,
sorted best to worst.
"""
function summary(ranks::DataFrame)
    tbl = combine(groupby(ranks, :Team)) do g
        r = g.Rank
        (median_rank = median(r), lower_90 = quantile(r, 0.05), upper_90 = quantile(r, 0.95))
    end
    sort!(tbl, :median_rank)
    println(tbl)
end

end # module Visualization
