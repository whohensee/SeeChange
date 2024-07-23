import time

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.measurements import Measurements

from util.util import parse_session, env_as_bool

class ParsScorer(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        # add the necessary parameters
        self.test_rb_score = self.add_par(
            'test_rb_score',
            5,
            int,
            'A totally fake value for testing. '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'scoring'

class Scorer:
    def __init__(self, **kwargs):
        self.pars = ParsScorer(**kwargs)

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

    # create a run
    def run(self, *args, **kwargs):
        """
        Look at a given Measurements object and assign a score based
        on the chosen ML/DL model. Potentially will include an R/B
        score in addition to other scores.
        """
        self.has_recalculated = False

        try:  # first make sure we get back a datastore, even an empty one
            ds, session = DataStore.from_args(*args, **kwargs)
        except Exception as e:
            return DataStore.catch_failure_to_parse(e, *args)
        
        # run the process
        try:
            #do a thing
            t_start = time.perf_counter()
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # get the provenance for this step:
            # WHPR TODO: as in issue, need capability potentially use multiple provenances
            prov = ds.get_provenance('scoring', self.pars.get_critical_pars(), session=session)

            # find the list of measurements
            measurements = ds.get_measurements(session=session)
            if measurements is None:
                raise ValueError(
                    f'Cannot find a measurements corresponding to the datastore inputs: {ds.get_inputs()}'
                )

            # iterate over the measurements, creating an appropriate DeepScore object for each.

        except Exception as e:
            ds.catch_exception(e)
        finally:  # make sure datastore is returned to be used in the next step
            return ds