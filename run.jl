using DataFrames
using DuckDB
using Logging
using TerryBradley

const DB_PATH = "./mlb_pred/data/db_2019_onwards.db"
const SEASON  = "2025"

mkpath(dirname(DB_PATH))

con = DBInterface.connect(DuckDB.DB, DB_PATH)

try
    MLBDataIngest.initialize_mlb_tables!(con)
    MLBDataIngest.ingest_mlb_data!(con)
    df = MLBDataIngest.db_to_df(con, SEASON)

    ids = Model.gen_ids(df)
    @info "Loaded $(DataFrames.nrow(df)) games for season $SEASON with $(length(ids)) teams."

    (; fit, ranks) = Model.fit_model(df, ids)

    Visualization.summary(ranks)
    Visualization.plot_ranks(ranks, SEASON)
finally
    DBInterface.close!(con)
    @info "Database connection closed."
end
