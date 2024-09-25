import uuid
import pytest

import sqlalchemy as sa

from models.base import SmartSession
from models.deepscore import DeepScore

def test_deepscore_saving(ptf_datastore, scorer):
    # scorer.pars.test_parameter = uuid.uuid4().hex   # this might not be necessary/possible anymore

    ds = ptf_datastore
    ##  delete the scores from the datastore
    ds.scores = None
    ## run the scorer on the measurements in the datastore
    ds = scorer.run(ds)
    ## check the scores are there
    assert len(ds.scores) == len(ds.measurements)
    ## assert that the scores have not recalculated ( thus they were found on DB )
    assert not scorer.has_recalculated
    assert len(ds.scores) == len(ds.measurements)

    ## try and commit
    with SmartSession() as session:
        ## try to commit and confirm there are no errors
        ds.save_and_commit(session=session)

    return None
