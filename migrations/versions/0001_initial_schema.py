"""schema inicial

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-09
"""

from alembic import op

from courtvision.data.models import Base

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(bind=bind)
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector(384)")


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
