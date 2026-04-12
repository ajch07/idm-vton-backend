"""
SKIPPED: Add garment attributes to products table.

This migration was prepared for edge case handling but is not needed for initial testing.
We're using a universal prompt approach first, then optimize after testing.
Un-skip this migration if edge case handling becomes necessary.

Revision ID: 001_add_garment_attributes
Revises: 
Create Date: 2026-04-11 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001_add_garment_attributes'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """SKIPPED - Not needed for initial testing."""
    pass


def downgrade() -> None:
    """SKIPPED - Not needed for initial testing."""
    pass
