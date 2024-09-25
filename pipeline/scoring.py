
import time

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.measurements import Measurements
from models.deepscore import DeepScore

from util.util import env_as_bool
from util.logger import SCLogger


class ParsScorer(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        # would we rather this be the integer in the enum list instead?
        self.algorithm = self.add_par(
            'algorithm',
            'random',
            str,
            'Name of the algorithm used to generate a score for this object.'
            'Valid names can be found in enums_and_bitflags.py.'
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

    def run(self, *args, **kwargs):
        """
        Look at the measurements and assign scores based
        on the chosen ML/DL model. Potentially will include an R/B
        score in addition to other scores.
        """
        self.has_recalculated = False

        try:  # first make sure we get back a datastore, even an empty one
            ds, session = DataStore.from_args(*args, **kwargs)
        except Exception as e:
            return DataStore.catch_failure_to_parse(e, *args)
        
        try:
            t_start = time.perf_counter()
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # get the provenance for this step:
            prov = ds.get_provenance('scoring', self.pars.get_critical_pars(), session=session)

            # find the list of measurements
            measurements = ds.get_measurements(session=session)
            if measurements is None:
                raise ValueError(
                    f'Cannot find a measurements corresponding to '
                    f'the datastore inputs: {ds.get_inputs()}'
                )
            
            # find if these deepscores have already been made
            scores = ds.get_deepscores( prov, session = session, reload=True )

            # compare whether the scores and measurements agree with each other
            # if each score goes to the corresponding measurement, should not need
            # to redo.
            # needs_redo = False
            # if scores is not None and len(scores) != 0:
            #     for i, s in enumerate(scores):
            #         if s.measurements_id != measurements[i].id:
            #             needs_redo = True


            if scores is None or len(scores) == 0: # or needs_redo:
                # recalculate
                self.has_recalculated = True

                # go over each measurements object and produce a DeepScore object
                scorelist = []

                for m in measurements:
                    d = DeepScore.from_measurements( m, provenance=prov )

                    d.evaluate_scores() # calculate the rb and ml scores

                    # add it to the list
                    scorelist.append( d )

                ds.scores = scorelist

            else:
                ds.scores = scores

            ds.runtimes['scoring'] = time.perf_counter() - t_start
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                ds.memory_usages['scoring'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2 # in MB

        except Exception as e:
            ds.catch_exception(e)
        finally:  # make sure datastore is returned to be used in the next step
            return ds