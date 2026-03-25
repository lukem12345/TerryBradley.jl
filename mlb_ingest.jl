using DuckDB
using DataFrames
using TerryBradley

const DB_PATH = "./mlb_pred/data/db_2019_onwards"

mkpath(DB_PATH)

db  = DuckDB.DB()
con = DBInterface.connect(db)

# Initialize and ingest.
MLBDataIngest.initialize_mlb_tables!(db)
MLBDataIngest.ingest_mlb_data!(db)

println("Tables:")
println(DBInterface.execute(con, "SHOW TABLES") |> DataFrame)
println("Seasons:  ", only(only(DBInterface.execute(con, "SELECT COUNT(*) FROM seasons"))))
println("Teams:    ", only(only(DBInterface.execute(con, "SELECT COUNT(*) FROM teams"))))
println("Schedule: ", only(only(DBInterface.execute(con, "SELECT COUNT(*) FROM schedule"))))
println("Scores:   ", only(only(DBInterface.execute(con, "SELECT COUNT(*) FROM scores"))))

# Export to Parquet files on disk.
# Each table becomes a .parquet file inside DB_PATH/.
DBInterface.execute(con, "EXPORT DATABASE '$DB_PATH' (FORMAT PARQUET)")
@info "Exported database to $DB_PATH"

DBInterface.close!(con)
close(db)
@info "Done."
