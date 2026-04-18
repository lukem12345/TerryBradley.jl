# Run mlb_ingest.jl to populate the database before running this script.
using BenchmarkTools
using DataFrames
using DuckDB
using Logging
using TerryBradley
using Turing

const DB_PATH = "./mlb_pred/data/db_2019_onwards"
const SEASON  = "2025"

# Load the exported Parquet files back into a fresh in-memory database.
db  = DuckDB.DB()
con = DBInterface.connect(db)

DuckDB.query(con, "IMPORT DATABASE '$DB_PATH'")
@info "Imported database from $DB_PATH"

# Verify.
println(DBInterface.execute(con, "SHOW TABLES") |> DataFrame)

df = db_to_df(db, SEASON)

ids = gen_ids(df)
@info "Loaded $(DataFrames.nrow(df)) games for season $SEASON with $(length(ids)) teams."

# When changing samplers, check the effective sample size per second.
sampler = Turing.NUTS()
#sampler = Turing.HMC(0.01, 10)
#sampler = Turing.SGLD()
@time (; fit, ranks) = fit_model(df, ids, BTLogSpace(), sampler)

ranks = filter(r -> r.iter > TerryBradley.Model.ITER_WARM, ranks)
rank_summary(ranks)
plot_ranks(ranks, SEASON)

#run(@benchmarkable fit_model($df, $ids, $(BTLogSpace()), $sampler) samples=100 evals=1 seconds=60*5)

