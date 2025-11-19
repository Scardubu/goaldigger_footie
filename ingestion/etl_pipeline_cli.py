import argparse

from ingestion.etl_pipeline import ingest_from_sources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest historical and upcoming football data.")
    parser.add_argument("--days_back", type=int, default=2, help="Number of days back to ingest historical data.")
    parser.add_argument("--days_ahead", type=int, default=7, help="Number of days ahead to ingest upcoming data.")
    parser.add_argument("--db_uri", type=str, default=None, help="Optional database URI override.")
    args = parser.parse_args()
    upserts = ingest_from_sources(db_uri=args.db_uri, days_back=args.days_back, days_ahead=args.days_ahead)
    print(f"ETL ingest complete: upserted {upserts} matches")
