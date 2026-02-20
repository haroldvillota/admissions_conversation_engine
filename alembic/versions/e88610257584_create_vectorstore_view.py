"""create vectorstore view

Revision ID: e88610257584
Revises: 
Create Date: 2026-02-19 12:12:02.613781

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e88610257584'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE VIEW vectorstore AS
        SELECT
            e.embedding_id AS langchain_id,
            c.text_content AS content,
            e.vector       AS embedding,
            c.metadata     AS langchain_metadata
        FROM embeddings e
        JOIN chunks c
            ON c.chunk_hash = e.chunk_hash;
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS vectorstore;")

