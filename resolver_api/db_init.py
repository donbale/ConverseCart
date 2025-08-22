import argparse
from db import init_db, DB_PATH, CATALOG_MD_PATH

def main():
    parser = argparse.ArgumentParser(
        description="Initialize the SQLite DB from product-catalog.md"
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help=f"Path to SQLite DB file (default: {DB_PATH})",
    )
    parser.add_argument(
        "--catalog",
        default=CATALOG_MD_PATH,
        help=f"Path to product catalog markdown (default: {CATALOG_MD_PATH})",
    )
    args = parser.parse_args()

    count = init_db(args.db, args.catalog)
    print(f"Initialized DB at: {args.db} with {count} products from {args.catalog}")

if __name__ == "__main__":
    main()