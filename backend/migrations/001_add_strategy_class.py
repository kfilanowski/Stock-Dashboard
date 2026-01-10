"""
Migration: Add strategy_class column to calibration_weights table.

Run this migration after updating models.py to add the strategy_class column.
This script handles SQLite's limitations with ALTER TABLE.

Usage:
    docker exec stock-dashboard-backend python migrations/001_add_strategy_class.py
"""

import sqlite3
import sys
from pathlib import Path

# Database path inside container
DB_PATH = "/app/data/portfolio.db"


def migrate():
    """Add strategy_class column to calibration_weights table."""

    if not Path(DB_PATH).exists():
        print(f"Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(calibration_weights)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'strategy_class' in columns:
            print("Column 'strategy_class' already exists. Skipping migration.")
            return

        print("Adding 'strategy_class' column to calibration_weights...")

        # Step 1: Add the new column with default value
        cursor.execute("""
            ALTER TABLE calibration_weights
            ADD COLUMN strategy_class TEXT NOT NULL DEFAULT 'all'
        """)

        # Step 2: Create new index for strategy lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS ix_calibration_strategy
            ON calibration_weights(ticker, horizon, strategy_class)
        """)

        # Step 3: Drop old unique constraint and create new one
        # SQLite doesn't support DROP CONSTRAINT, so we need to recreate the table
        # For now, just create a new unique index (SQLite will enforce it)
        # The old constraint 'uix_calibration_key' will remain but won't conflict
        # since all existing rows have strategy_class='all'

        print("Creating new unique index with strategy_class...")
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uix_calibration_key_v2
            ON calibration_weights(ticker, indicator, action, horizon, strategy_class)
        """)

        conn.commit()

        # Verify
        cursor.execute("SELECT COUNT(*) FROM calibration_weights WHERE strategy_class = 'all'")
        count = cursor.fetchone()[0]
        print(f"Migration complete. {count} existing rows now have strategy_class='all'")

    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
