"""exposure_preproc_bitflag

Revision ID: 147e28b89b99
Revises: 75ab6a2da054
Create Date: 2024-09-04 20:05:29.996720

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '147e28b89b99'
down_revision = '75ab6a2da054'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('exposures', sa.Column('preproc_bitflag', sa.SMALLINT(), server_default=sa.text('0'), nullable=False))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('exposures', 'preproc_bitflag')
    # ### end Alembic commands ###