"""
    MLBDataIngest

Ingests data from the MLB API into a local DuckDB database.
"""
module MLBDataIngest

using DataFrames
using Dates
using DuckDB
using HTTP
using JSON3
using Logging

_dbe = DBInterface.execute

const MLB_BASE = "https://statsapi.mlb.com/api/v1"

"""
    mlb_get(endpoint; params...)

Makes a GET request to the MLB stats API and returns the parsed JSON body.
"""
function mlb_get(endpoint::String; params...)
    url   = "$MLB_BASE/$endpoint"
    query = Dict(string(k) => string(v) for (k, v) in params)
    query["sportId"] = "1"
    resp  = HTTP.get(url; query = query)
    return JSON3.read(resp.body)
end

_seasons_table = """
    create table if not exists seasons (
        season_id              varchar(4)
        , has_wildcard         boolean
        , preseason_start      date
        , season_start         date
        , regular_season_start date
        , regular_season_end   date
        , season_end           date
        , offseason_start      date
        , offseason_end        date
        , primary key (season_id)
    );
"""

_teams_table = """
    create table if not exists teams (
        season_id   varchar(4)
        , team_id   varchar(4)
        , team_name varchar(50)
        , team_abbr varchar(4)
        , foreign key (season_id) references seasons(season_id)
        , primary key (season_id, team_id)
    );
"""

_schedule_table = """
    create table if not exists schedule (
        season_id          varchar(4)
        , game_date        date
        , game_id          varchar(10)
        , double_header    varchar(1)
        , away_team        varchar(4)
        , away_team_wins   integer
        , away_team_losses integer
        , home_team        varchar(4)
        , home_team_wins   integer
        , home_team_losses integer
        , foreign key (season_id) references seasons(season_id)
        , primary key (season_id, game_id)
    );
"""

_score_table = """
    create table if not exists scores (
        game_id     varchar(10)
        , home_runs integer
        , away_runs integer
    );
"""

"""
    initialize_mlb_tables!(db)

Creates all necessary tables. Safe to call on an existing database —
all statements use `create table if not exists`. Accepts a `DuckDB.DB`
object and opens its own connection internally.
"""
function initialize_mlb_tables!(db)
    con = DBInterface.connect(db)
    try
        _dbe(con, _seasons_table);  @info "Initialized the seasons table."
        _dbe(con, _teams_table);    @info "Initialized the teams table."
        _dbe(con, _schedule_table); @info "Initialized the schedule table."
        _dbe(con, _score_table);    @info "Initialized the score table."
        @info "Completed the DB initialization."
    finally
        DBInterface.close!(con)
    end
end

# ============================================================
# Ingestion
# ============================================================

_season_data_keys = [
    "seasonId",
    "hasWildcard",
    "preSeasonStartDate",
    "seasonStartDate",
    "regularSeasonStartDate",
    "regularSeasonEndDate",
    "seasonEndDate",
    "offSeasonStartDate",
    "offSeasonEndDate",
]

function _season_ingest(db)
    con = DBInterface.connect(db)
    try
        for year in 2019:2025
            data = mlb_get("seasons/$year")
            for s in data["seasons"]
                get_key(key) = get(s, key, missing)
                _dbe(con, """
                    insert into seasons values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    on conflict do nothing;
                """, get_key.(_season_data_keys))
            end
        end
    finally
        DBInterface.close!(con)
    end
end

function _teams_ingest(db)
    con = DBInterface.connect(db)
    try
        missing_seasons = _dbe(con, """
            select s.season_id
            from teams as t
                right join seasons as s on t.season_id = s.season_id
            where t.season_id is null
        """) |> DataFrame

        for season_id in missing_seasons.season_id
            data = mlb_get("teams"; season = season_id)
            for t in data["teams"]
                _dbe(con, """
                    insert into teams values (?, ?, ?, ?)
                    on conflict do nothing;
                """, [
                    get(t, "season",       missing),
                    string(get(t, "id",    missing)),
                    get(t, "name",         missing),
                    get(t, "abbreviation", missing),
                ])
            end
        end
    finally
        DBInterface.close!(con)
    end
end

function _schedule_ingest(db)
    con = DBInterface.connect(db)
    try
        stored     = _dbe(con, "select game_id from schedule") |> DataFrame
        stored_set = Set(stored.game_id)

        for season_id in 2019:2025
            data = mlb_get("schedule"; season = season_id)
            for date_entry in data["dates"]
                for game in date_entry["games"]
                    game_id = string(game["gamePk"])
                    game_id in stored_set && continue
                    try
                        away = game["teams"]["away"]
                        home = game["teams"]["home"]
                        _dbe(con, """
                            insert into schedule values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            on conflict do nothing;
                        """, [
                            game["season"],
                            date_entry["date"],
                            string(game["gamePk"]),
                            game["doubleHeader"],
                            string(away["team"]["id"]),
                            away["leagueRecord"]["wins"],
                            away["leagueRecord"]["losses"],
                            string(home["team"]["id"]),
                            home["leagueRecord"]["wins"],
                            home["leagueRecord"]["losses"],
                        ])
                    catch e
                        @warn "Skipping game $game_id: $e"
                    end
                end
            end
            @info "Ingested season $season_id."
        end
    finally
        DBInterface.close!(con)
    end
end

"""
    _fetch_date_scores(game_date, stored_set)
Fetches scores for all games on `game_date` from the MLB API.
Pure HTTP — no database access — so safe to call concurrently.
Returns a vector of (game_id, home_runs, away_runs) tuples,
skipping games already in `stored_set`.
"""
function _fetch_date_scores(game_date, stored_set)
    data = mlb_get("schedule"; date = game_date)
    rows = []
    for date_entry in data["dates"]
        for game in date_entry["games"]
            game_id   = string(game["gamePk"])
            game_id in stored_set && continue
            home_runs = get(get(get(game, "teams", Dict()), "home", Dict()), "score", missing)
            away_runs = get(get(get(game, "teams", Dict()), "away", Dict()), "score", missing)
            (ismissing(home_runs) || ismissing(away_runs)) && continue
            push!(rows, (game_id, home_runs, away_runs))
        end
    end
    return rows
end

function _score_ingest(db)
    today = Dates.today()

    # Read-only setup queries on a dedicated connection, closed before
    # any concurrent work begins.
    con = DBInterface.connect(db)
    dates = _dbe(con, """
        select distinct schedule.game_date
        from schedule
            left join seasons on schedule.season_id = seasons.season_id
        where
            schedule.game_date <= cast('$today' as date)
            and cast(schedule.season_id as integer) >= 2019
            and schedule.game_date
                between seasons.regular_season_start
                    and seasons.regular_season_end
        order by schedule.game_date
    """) |> DataFrame
    stored     = _dbe(con, "select game_id from scores") |> DataFrame
    stored_set = Set(stored.game_id)
    DBInterface.close!(con)

    counter = Threads.Atomic{Int}(0)
    n_dates = length(dates.game_date)

    # Fetch the scores for games on those dates.
    results = asyncmap(dates.game_date; ntasks = 20) do game_date
        try
            return _fetch_date_scores(game_date, stored_set)
        catch e
            @warn "Skipping date $game_date: $e"
            return []
        finally
            n = Threads.atomic_add!(counter, 1) + 1
            n % 50 == 0 && @info "Fetched $n / $n_dates dates..."
        end
    end

    # Insert those scores.
    for day_results in results
        for (game_id, home_runs, away_runs) in day_results
            DuckDB.append(appender, game_id)
            DuckDB.append(appender, home_runs)
            DuckDB.append(appender, away_runs)
            DuckDB.end_row(appender)
        end
    end
    DuckDB.close(appender)

    con = DBInterface.connect(db)
    try
        DBInterface.execute(con, "CHECKPOINT")
    finally
        DBInterface.close!(con)
    end

    @info "Score ingestion complete."
end

"""
    ingest_mlb_data!(db)

Runs the full ingestion pipeline: seasons → teams → schedule → scores.
Accepts a `DuckDB.DB` object. Each stage opens and closes its own
connection so connections are never shared across threads.
"""
function ingest_mlb_data!(db)
    _season_ingest(db);   @info "Ingested the season data."
    _teams_ingest(db);    @info "Ingested the teams data."
    _schedule_ingest(db); @info "Ingested the schedule data."
    _score_ingest(db);    @info "Ingested the score data."
end

"""
    db_to_df(db, season)

Pulls regular-season game results for `season` from the DuckDB database.
Returns a DataFrame with columns: game_id, home_abbr, away_abbr, home_win.
"""
function db_to_df(db, season::String)
    con = DBInterface.connect(db)
    try
        df = _dbe(con, """
            with a as (
                select
                    scores.game_id
                    , schedule.home_team
                    , scores.home_runs
                    , schedule.away_team
                    , scores.away_runs
                from scores
                    left join schedule
                        on scores.game_id = schedule.game_id
                    left join seasons
                        on schedule.season_id = seasons.season_id
                where
                    schedule.season_id = '$season'
                    and schedule.game_date
                        between seasons.regular_season_start
                            and seasons.regular_season_end
            )
            select
                a.game_id
                , ht.team_abbr  as home_abbr
                , awt.team_abbr as away_abbr
                , (case
                    when home_runs > away_runs then 1
                    else 0
                end) as home_win
            from a
                left join teams ht  on a.home_team = ht.team_id  and ht.season_id  = '$season'
                left join teams awt on a.away_team = awt.team_id and awt.season_id = '$season'
        """) |> DataFrame
        return dropmissing(df, [:home_abbr, :away_abbr])
    finally
        DBInterface.close!(con)
    end
end

end # module MLBDataIngest