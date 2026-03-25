# Run mlb_ingest.jl to populate the database before running this script.
using DataFrames
using DuckDB
using Logging
using TerryBradley

const DB_PATH = "./mlb_pred/data/db_2019_onwards"
const SEASON  = "2025"

# Load the exported Parquet files back into a fresh in-memory database.
db  = DuckDB.DB()
con = DBInterface.connect(db)

DuckDB.query(con, "IMPORT DATABASE '$DB_PATH'")
@info "Imported database from $DB_PATH"

# Verify.
println(DBInterface.execute(con, "SHOW TABLES") |> DataFrame)

df = MLBDataIngest.db_to_df(db, SEASON)

ids = Model.gen_ids(df)
@info "Loaded $(DataFrames.nrow(df)) games for season $SEASON with $(length(ids)) teams."

@time (; fit, ranks) = Model.fit_model(df, ids)

Visualization.summary(ranks)
Visualization.plot_ranks(ranks, SEASON)
