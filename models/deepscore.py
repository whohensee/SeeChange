import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, SmartSession, AutoIDMixin


class DeepScore(Base, AutoIDMixin):
    __tablename__ = 'deepscore'

    # table args including unique constraints

    # might be more specifically linked to a cutouts, but for now that will be a measurements
    # might want to consider a unique constraint for this if its important to not redo

    # measurements id
    measurements_id = sa.Column(
        sa.ForeignKey('measurements.id', ondelete='CASCADE', name='deep_score_measurements_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the measurements this DeepScore is associated with. ",
    )
    # measurements (potentially a relationship)
    # investigate the difference btw association_proxy (as in zp) and relationship
    measurements = orm.relationship(
        'Measurements',
        lazy='selectin',
        cascade='save-update, merge, refresh-expire, expunge',
        passive_deletes=True,
        doc="The measurements this DeepScore is associated with. ",
    )

    # provenance_id
    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='deep_score_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this DeepScore. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this DeepScore. "
        )
    )
    # provenance (relationship) described in proj
    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this DeepScore. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this DeepScore. "
        )
    )
    # float score
    # WHPR TODO: improve the docstring here
    score = sa.Column(
        sa.REAL,
        nullable=False,
        doc="The score determined by the ML/DL algorithm used for this object. "
    )
