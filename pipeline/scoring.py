import time

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.measurements import Measurements
from models.deepscore import DeepScore

from util.util import parse_session, env_as_bool

class ParsScorer(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        # add the necessary parameters
        self.test_rb_score = self.add_par(
            'test_rb_score',
            5,
            float,
            'A totally fake value for testing. '
        )

        self.algorithm = self.add_par(
            'algorithm',
            'random',
            str,
            'The name of the algorithm used to generate a score for this object.'
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
        Look at the measurements and assign scores based
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
            # For now this is solved by creating multiple scorer objects in top_level, since
            # if we pass all that info down into Scorer, it could cause some provenance trouble.
            # (A 'random' algorithm deepscore created in a pipeline run using only 'random' should
            # have the same provenance as a 'random' algorithm deepscore created in a pipeline run
            # which scored based on 'random' and 'allperfect')
            prov = ds.get_provenance('scoring', self.pars.get_critical_pars(), session=session)

            # find the list of measurements
            measurements = ds.get_measurements(session=session)
            if measurements is None:
                raise ValueError(
                    f'Cannot find a measurements corresponding to the datastore inputs: {ds.get_inputs()}'
                )

            # find if this deepscore object has already been made in the ds
            # scores = ds.get_deepscore(prov, session=session) # WHPR TODO: potentially create this function
            self.has_recalculated = True
            if hasattr(ds, 'scores') and ds.scores is not None:
                # TODO: Note that this will cause issues if, for some reason, you run different versions of the
                # same algorithm in a single run. Consider another solution?
                badscores = [s for s in ds.scores if s.provenance.parameters['algorithm'] == self.pars.algorithm]
                ds.scores = [s for s in ds.scores if s not in badscores]
            else:
                ds.scores = []

            scores = []
            # iterate over the measurements, creating an appropriate DeepScore object for each.
            for m in measurements:

                # make a deepscore object for a specific measurement
                score = DeepScore.from_measurements(m, provenance=prov)

                score.evaluate_scores() # calculate the rb and ml scores

                # add it to the list
                scores.append(score)

            ds.scores.extend(scores)

            ds.runtimes['scoring'] = time.perf_counter() - t_start
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                ds.memory_usages['scoring'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

        except Exception as e:
            ds.catch_exception(e)
        finally:  # make sure datastore is returned to be used in the next step
            return ds
