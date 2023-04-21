

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.preprocessor import Preprocessor
from pipeline.astrometry import Astrometry
from pipeline.calibrator import Calibrator
from pipeline.subtractor import Subtractor
from pipeline.detector import Detector
from pipeline.cutter import Cutter
from pipeline.measurer import Measurer


# should this come from db.py instead?
from models.base import SmartSession

config = {}  # TODO: replace this with Rob's config loader


# put all the top-level pipeline parameters in the init of this class:
class ParsPipeline(Parameters):

    def __init__(self, **kwargs):
            super().__init__()

            self.add_par('example_pipeline_parameter', 1, int, 'an example pipeline parameter')

            self._enforce_no_new_attrs = True  # lock against new parameters

            self.override(kwargs)


class Pipeline:
    def __init__(self, **kwargs):
        # top level parameters
        self.pars = ParsPipeline(**config.get('pipeline', {}))
        self.pars.augment(kwargs.get('pipeline', {}))

        # dark/flat and sky subtraction tools
        preprocessor_config = config.get('preprocessor', {})
        preprocessor_config.update(kwargs.get('preprocessor', {}))
        self.pars.add_defaults_to_dict(preprocessor_config)
        self.preprocessor = Preprocessor(**preprocessor_config)

        # source detection ("extraction" for the regular image!)
        extractor_config = config.get('extractor', {})
        extractor_config.update(kwargs.get('extractor', {}))
        self.pars.add_defaults_to_dict(extractor_config)
        self.extractor = Detector(**extractor_config)

        # astrometric fit using a first pass of sextractor and then astrometric fit to Gaia
        astrometry_config = config.get('astrometry', {})
        astrometry_config.update(kwargs.get('astrometry', {}))
        self.pars.add_defaults_to_dict(astrometry_config)
        self.astrometry = Astrometry(**astrometry_config)

        # photometric calibration:
        calibrator_config = config.get('calibrator', {})
        calibrator_config.update(kwargs.get('calibrator', {}))
        self.pars.add_defaults_to_dict(calibrator_config)
        self.calibrator = Calibrator(**calibrator_config)

        # reference fetching and image subtraction
        subtractor_config = config.get('subtractor', {})
        subtractor_config.update(kwargs.get('subtractor', {}))
        self.pars.add_defaults_to_dict(subtractor_config)
        self.subtractor = Subtractor(**subtractor_config)

        # source detection ("detection" for the subtracted image!)
        detector_config = config.get('detector', {})
        detector_config.update(kwargs.get('detector', {}))
        self.pars.add_defaults_to_dict(detector_config)
        self.detector = Detector(**detector_config)

        # produce cutouts for detected sources:
        cutter_config = config.get('cutter', {})
        cutter_config.update(kwargs.get('cutter', {}))
        self.pars.add_defaults_to_dict(cutter_config)
        self.cutter = Cutter(**cutter_config)

        # measure photometry, analytical cuts, and deep learning models on the Cutouts:
        measurer_config = config.get('measurer', {})
        measurer_config.update(kwargs.get('extractor', {}))
        self.pars.add_defaults_to_dict(measurer_config)
        self.measurer = Measurer(**measurer_config)

    def run(self, *args, **kwargs):
        """
        Run the entire pipeline on a specific CCD in a specific exposure.
        Will open a database session and grab any existing data,
        and calculate and commit any new data that did not exist.
        """

        ds, session = DataStore.from_args(*args, **kwargs)

        # run dark/flat and sky subtraction tools, save the results as Image objects to DB and disk
        ds = self.preprocessor.run(ds, session)

        # extract sources and make a SourceList from the regular image
        ds = self.extractor.run(ds, session)

        # find astrometric solution, save WCS into Image object and FITS headers
        ds = self.astrometry.run(ds, session)

        # cross-match against photometric catalogs and get zero point, save into Image object and FITS headers
        ds = self.calibrator.run(ds, session)

        # fetch reference images and subtract them, save SubtractedImage objects to DB and disk
        ds = self.subtractor.run(ds, session)

        # make cutouts of all the sources in the "detections" source list
        ds = self.cutter.run(ds, session)

        # find sources, generate a source list for detections
        ds = self.detector.run(ds, session)

        # extract photometry, analytical cuts, and deep learning models on the Cutouts:
        ds = self.measurer.run(ds, session)

        return ds

    def run_with_session(self):
        """
        Run the entire pipeline using one session that is opened
        at the beginning and closed at the end of the session,
        just to see if that causes any problems with too many open sessions.
        """
        with SmartSession() as session:
            self.run(session=session)
