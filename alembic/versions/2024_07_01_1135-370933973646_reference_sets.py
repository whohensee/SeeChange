"""reference sets

Revision ID: 370933973646
Revises: a375526c8260
Create Date: 2024-06-23 11:35:43.941095

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '370933973646'
down_revision = '7384c6d07485'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('refsets',
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('upstream_hash', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('modified', sa.DateTime(), nullable=False),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_refsets_created_at'), 'refsets', ['created_at'], unique=False)
    op.create_index(op.f('ix_refsets_id'), 'refsets', ['id'], unique=False)
    op.create_index(op.f('ix_refsets_name'), 'refsets', ['name'], unique=True)
    op.create_index(op.f('ix_refsets_upstream_hash'), 'refsets', ['upstream_hash'], unique=False)
    op.create_table('refset_provenance_association',
    sa.Column('provenance_id', sa.Text(), nullable=False),
    sa.Column('refset_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['provenance_id'], ['provenances.id'], name='refset_provenances_association_provenance_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['refset_id'], ['refsets.id'], name='refsets_provenances_association_refset_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('provenance_id', 'refset_id')
    )
    op.drop_index('ix_refs_validity_end', table_name='refs')
    op.drop_index('ix_refs_validity_start', table_name='refs')
    op.drop_column('refs', 'validity_start')
    op.drop_column('refs', 'validity_end')

    op.add_column('images', sa.Column('airmass', sa.REAL(), nullable=True))
    op.create_index(op.f('ix_images_airmass'), 'images', ['airmass'], unique=False)
    op.add_column('exposures', sa.Column('airmass', sa.REAL(), nullable=True))
    op.create_index(op.f('ix_exposures_airmass'), 'exposures', ['airmass'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_images_airmass'), table_name='images')
    op.drop_column('images', 'airmass')
    op.drop_index(op.f('ix_exposures_airmass'), table_name='exposures')
    op.drop_column('exposures', 'airmass')
    op.add_column('refs', sa.Column('validity_end', postgresql.TIMESTAMP(), autoincrement=False, nullable=True))
    op.add_column('refs', sa.Column('validity_start', postgresql.TIMESTAMP(), autoincrement=False, nullable=True))
    op.create_index('ix_refs_validity_start', 'refs', ['validity_start'], unique=False)
    op.create_index('ix_refs_validity_end', 'refs', ['validity_end'], unique=False)
    op.drop_table('refset_provenance_association')
    op.drop_index(op.f('ix_refsets_upstream_hash'), table_name='refsets')
    op.drop_index(op.f('ix_refsets_name'), table_name='refsets')
    op.drop_index(op.f('ix_refsets_id'), table_name='refsets')
    op.drop_index(op.f('ix_refsets_created_at'), table_name='refsets')
    op.drop_table('refsets')
    # ### end Alembic commands ###