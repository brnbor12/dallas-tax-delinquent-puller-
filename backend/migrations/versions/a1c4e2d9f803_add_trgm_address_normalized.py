"""add_trgm_and_address_normalized

Revision ID: a1c4e2d9f803
Revises: 45b89c2b5b2b
Create Date: 2026-03-02

Adds:
  - pg_trgm extension (for similarity() and % operator)
  - address_normalized column on properties (normalized form for fuzzy matching)
  - GIN trigram index on address_normalized (fast similarity search)
  - Backfills address_normalized for all existing rows via SQL UPPER() + regexp_replace()

The normalization in SQL mirrors scrapers/address_utils.py:
  1. Uppercase
  2. Remove punctuation
  3. Strip trailing state + zip
  4. Collapse whitespace
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = "a1c4e2d9f803"
down_revision: Union[str, None] = "45b89c2b5b2b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# SQL normalization expression matching address_utils.normalize_address()
# Applied in-database for the backfill so we don't need to load Python.
_NORMALIZE_SQL = """
    TRIM(
      REGEXP_REPLACE(
        REGEXP_REPLACE(
          REGEXP_REPLACE(
            UPPER(address_raw),
            '[.,#'';]', ' ', 'g'          -- remove punctuation
          ),
          '\\m[A-Z]{2}\\s+\\d{5}(-\\d{4})?\\M\\s*$', '', 'g'   -- strip state+zip at end
        ),
        '\\s+', ' ', 'g'                  -- collapse whitespace
      )
    )
"""


def upgrade() -> None:
    # 1. Enable pg_trgm
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # 2. Add address_normalized column
    op.add_column(
        "properties",
        sa.Column("address_normalized", sa.Text(), nullable=True),
    )

    # 3. Backfill existing rows
    op.execute(
        f"UPDATE properties SET address_normalized = {_NORMALIZE_SQL}"
    )

    # 4. GIN trigram index for fast similarity queries
    op.create_index(
        "idx_properties_addr_trgm",
        "properties",
        ["address_normalized"],
        postgresql_using="gin",
        postgresql_ops={"address_normalized": "gin_trgm_ops"},
    )


def downgrade() -> None:
    op.drop_index("idx_properties_addr_trgm", table_name="properties")
    op.drop_column("properties", "address_normalized")
    # Note: we intentionally do NOT drop pg_trgm here — other code may rely on it
