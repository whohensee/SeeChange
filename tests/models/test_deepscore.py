import uuid
import pytest

import sqlalchemy as sa

from models.base import SmartSession
from models.deepscore import DeepScore
from pipeline.scoring import Scorer

from util.config import Config

def test_deepscore_creation():
    return None

def test_single_algorithm(ptf_datastore, scorers):

    scorer1 = scorers[0]  # 'random'
    scorer2 = scorers[1]  # 'allperfect'

    scorer1.pars.test_parameter = uuid.uuid4().hex
    scorer2.pars.test_parameter = uuid.uuid4().hex

    ds = ptf_datastore

    m_count = len(ds.measurements)

    try:

        # run with the scorer, using algorithm 'random'
        with SmartSession() as session:

            ds = scorer1.run(ds, session)
            ds = scorer2.run(ds, session)

            ds.save_and_commit(session=session)

        # check that we can query the database for the deepscores
        m_ids = [m.id for m in ds.measurements]
        with SmartSession() as session:
            scores1 = session.scalars(
                sa.select(DeepScore)
                .filter(DeepScore._algorithm == 0)
                .filter(DeepScore.measurements_id.in_(m_ids))
            ).all()

            scores2 = session.scalars(
                sa.select(DeepScore)
                .filter(DeepScore._algorithm == 1)
                .filter(DeepScore.measurements_id.in_(m_ids))
            ).all()

        assert len(scores1) == m_count
        assert len(scores2) == m_count

    finally:
        return None
    # cleanup happens in ptf_datastore fixture
        if 'ds' in locals():
            ds.delete_everything()

def test_multiple_algorithms():
    return None

