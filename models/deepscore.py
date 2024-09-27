import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects.postgresql import JSONB

import numpy as np

from models.base import Base, UUIDMixin, SeeChangeBase, SmartSession, HasBitFlagBadness
from models.enums_and_bitflags import DeepscoreAlgorithmConverter
from models.provenance import Provenance
from models.measurements import Measurements



class DeepScore(Base, UUIDMixin, HasBitFlagBadness):
    """Contains the Deep Learning/Machine Learning algorithm score assigned to
    the corresponding measurements object. Each DeepScore object contains the
    result of a single algorithm, which is identified by the ___ attribute.
    """

    __tablename__ = 'deepscores'

    # IMPORTANT
    # Do I need to add any table constraints here, and how should I write them?

    _algorithm = sa.Column(
        sa.SMALLINT,
        nullable=False,
        # QUESTION: Definitely never want a deepscore being pushed without an algorithm -
        #    would it be better to have no default, or an obviously bad default, with
        #    nullable=False?
        # server_default=sa.sql.elements.TextClause( '-1' )
        doc=("Integer which represents which of the ML/DL algorithms was used "
        "for this object. Also specifies necessary parameters for a given "
        "algorithm."
        )
    )

    info = sa.Column(
        JSONB,
        nullable=False,
        server_default='{}',
        doc=(
            "Additional information on this DeepScore. "
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

    measurements_id = sa.Column(
        sa.ForeignKey('measurements._id', ondelete='CASCADE',
                      name='deepscore_measurements_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the measurements this DeepScore is associated with."
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances._id', ondelete='CASCADE',
                      name='deepscore_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this DeepScore. "
            "The provenance will contain a record of the code version and the "
            "parameters used to produce this DeepScore."
        )
    )

    score = sa.Column(
        sa.REAL,
        nullable=False,
        doc="The score determined by the ML/DL algorithm used for this object."
    )

    def __init__(self, *args, **kwargs):
        SeeChangeBase.__init__(self) # don't pass kwargs as they could contain
                                     # non-column key-values
        HasBitFlagBadness.__init__(self)

        # manually set all properties (columns or not)
        self.set_attributes_from_dict(kwargs)

    @orm.reconstructor
    def init_on_load(self):
        SeeChangeBase.init_on_load(self)


    def __repr__(self):
        return (
            f"<DeepScore {self.id} "
            f"from Measurements {self.measurements_id} "
            f"with algorithm {self.algorithm}>"
        )

    @staticmethod
    def from_measurements(measurements, provenance=None, **kwargs):
        """Create a DeepScore object from a single measurements object, using 
        the parameters described in the given provenance.
        """

        score = DeepScore()
        score.measurements_id = measurements.id
        score.provenance_id = None if provenance is None else provenance.id

        score._upstream_bitflag = measurements.bitflag

        return score

    def evaluate_scores(self):
        """Evaluate the ML/DL score for the associated measurement, using the 
        appropriate algorithm.
        """
        # consider doing cases or another solution, for now just if-else block

        algo = Provenance.get( self.provenance_id ).parameters['algorithm']

        if algo == 'random':
            self.score = np.random.default_rng().random()
            self.algorithm = algo

        elif algo == 'allperfect':
            self.score = 1.0
            self.algorithm = algo
        
        elif algo in DeepscoreAlgorithmConverter.dict_inverse:
            raise NotImplementedError(f"algorithm {algo} isn't yet implemented")
        
        else:
            raise ValueError(f"{algo} is not a valid ML algorithm.")
        
        return None
    

    def get_upstreams(self, session=None):
        """Get the measurements that was used to make this deepscore. """
        with SmartSession(session) as session:
            return session.scalars(sa.select(Measurements)
                                   .where(Measurements._id ==
                                          self.measurements_id)).all()
        
    def get_downstreams(self, session=None, siblings=False):
        """Get the downstreams of this DeepScore"""
        return []