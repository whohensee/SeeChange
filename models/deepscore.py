import sqlalchemy as sa
from sqlalchemy import orm
import numpy as np

from sqlalchemy.ext.hybrid import hybrid_property

from models.base import Base, SmartSession, AutoIDMixin, SeeChangeBase
from models.measurements import Measurements
from models.enums_and_bitflags import DeepscoreAlgorithmConverter


class DeepScore(Base, AutoIDMixin):
    __tablename__ = 'deepscores'

    # table args including unique constraints

    # might be more specifically linked to a cutouts, but for now that will be a measurements
    # might want to consider a unique constraint for this if its important to not redo

    _algorithm = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=DeepscoreAlgorithmConverter.convert('random'),
        doc=(
            "Integer which represents which of the ML/DL algorithms was used for this object. "
            "Also specifies necessary parameters for a given algorithm."
        )
    )

    @hybrid_property
    def algorithm(self):
        return DeepscoreAlgorithmConverter.convert( self._algorithm )
    

    @algorithm.expression
    def algorithm(cls):
        return sa.case( DeepscoreAlgorithmConverter.dict, value=cls._algorithm )

    @algorithm.setter
    def algorithm( self, value ):
        self._algorithm = DeepscoreAlgorithmConverter.convert( value )

    # measurements id
    measurements_id = sa.Column(
        sa.ForeignKey('measurements.id', ondelete='CASCADE', name='deep_score_measurements_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the measurements this DeepScore is associated with. ",
    )

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
    score = sa.Column(
        sa.REAL,
        nullable=False,
        doc="The score determined by the ML/DL algorithm used for this object. "
    )

    # init
    def __init__(self, *args, **kwargs):
        SeeChangeBase.__init__(self) # don't pass kwargs as they could contain non-column key-values

        # manually set all properties (columns or not)
        self.set_attributes_from_dict(kwargs)
    
    # reconstructor
    @orm.reconstructor
    def init_on_load(self):
        SeeChangeBase.init_on_load(self)

    # consider how much data is useful adding into repr
    def __repr__(self):
        return (
            f"<DeepScore {self.id} "
            f"from Measurements {self.measurements_id} "
            f"with algorithm {self.algorithm}>"
        )

    @staticmethod
    def from_measurements(measurements, provenance=None, **kwargs):
        """
        Create a DeepScore object from a single measurements object, using the parameters
        described in the given provenance.
        """

        score = DeepScore()
        score.measurements = measurements
        score.provenance = provenance
        # score.provenance_id = provenance.id # WHPR check if this should auto-fill somewhere

        return score
    
    def evaluate_scores(self):
        """
        Evaluate the ML/DL score for the associated measurement, using the appropriate
        algorithm.
        """
        # consider doing cases, for now just an if-block

        algo = self.provenance.parameters['algorithm']

        # random ML score
        if algo  == 'random':
            self.score = np.random.default_rng().random()
            self.algorithm = 'random'

        elif algo  == 'allperfect':
            self.score = 1.0
            self.algorithm = 'allperfect'

        elif algo in DeepscoreAlgorithmConverter.dict_inverse:
            raise NotImplementedError(f"ML algorithm {algo} is not implemented yet")
        
        else:
            raise ValueError(f"{algo} is not a valid ML algorithm.")

        return None

    # get upstreams
    def get_upstreams(self, session=None):
        """Get the measurements that was used to make this deepscore. """
        with SmartSession(session) as session:
            return session.scalars(sa.select(Measurements).where(Measurements.id == self.measurementss_id)).all()

    def get_downstreams(self, session=None, siblings=False):
        """Get the downstreams of this DeepScore"""
        return []
