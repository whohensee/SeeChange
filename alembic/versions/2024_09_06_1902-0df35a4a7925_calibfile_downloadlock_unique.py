"""calibfile_downloadlock_unique

Revision ID: 0df35a4a7925
Revises: 140047012e43
Create Date: 2024-09-06 19:02:31.393214

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0df35a4a7925'
down_revision = '140047012e43'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('ix_calibfile_downloadlock__calibrator_set', table_name='calibfile_downloadlock')
    op.drop_index('ix_calibfile_downloadlock__flat_type', table_name='calibfile_downloadlock')
    op.drop_index('ix_calibfile_downloadlock__type', table_name='calibfile_downloadlock')
    op.drop_index('ix_calibfile_downloadlock_instrument', table_name='calibfile_downloadlock')
    op.drop_index('ix_calibfile_downloadlock_sensor_section', table_name='calibfile_downloadlock')
    op.create_unique_constraint('calibfile_downloadlock_unique', 'calibfile_downloadlock', ['_type', '_calibrator_set', '_flat_type', 'instrument', 'sensor_section'], postgresql_nulls_not_distinct=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('calibfile_downloadlock_unique', 'calibfile_downloadlock', type_='unique')
    op.create_index('ix_calibfile_downloadlock_sensor_section', 'calibfile_downloadlock', ['sensor_section'], unique=False)
    op.create_index('ix_calibfile_downloadlock_instrument', 'calibfile_downloadlock', ['instrument'], unique=False)
    op.create_index('ix_calibfile_downloadlock__type', 'calibfile_downloadlock', ['_type'], unique=False)
    op.create_index('ix_calibfile_downloadlock__flat_type', 'calibfile_downloadlock', ['_flat_type'], unique=False)
    op.create_index('ix_calibfile_downloadlock__calibrator_set', 'calibfile_downloadlock', ['_calibrator_set'], unique=False)
    # ### end Alembic commands ###