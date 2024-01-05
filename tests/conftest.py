import os
import warnings
import pytest
import uuid
import shutil
import pathlib

import numpy as np

import sqlalchemy as sa

from util.config import Config
from models.base import FileOnDiskMixin, SmartSession, CODE_ROOT, get_all_database_objects, _logger
from models.provenance import CodeVersion, Provenance
from models.catalog_excerpt import CatalogExcerpt
from models.exposure import Exposure

from util.archive import Archive

pytest_plugins = [
    'tests.fixtures.simulated',
    'tests.fixtures.decam',
    'tests.fixtures.ztf',
    'tests.fixtures.ptf',
    'tests.fixtures.pipeline_objects',
]


# this fixture should be the first thing loaded by the test suite
# (session is the pytest session, not the SQLAlchemy session)
def pytest_sessionstart(session):
    # Will be executed before the first test
    # print('Initial setup fixture loaded! ')

    # make sure to load the test config
    test_config_file = str((pathlib.Path(__file__).parent.parent / 'tests' / 'seechange_config_test.yaml').resolve())
    Config.get(configfile=test_config_file, setdefault=True)
    FileOnDiskMixin.configure_paths()


# This will be executed after the last test (session is the pytest session, not the SQLAlchemy session)
def pytest_sessionfinish(session, exitstatus):
    # print('Final teardown fixture executed! ')
    with SmartSession() as dbsession:
        # first get rid of any Exposure loading Provenances, if they have no Exposures attached
        provs = dbsession.scalars(sa.select(Provenance).where(Provenance.process == 'load_exposure'))
        for prov in provs:
            exp = dbsession.scalars(sa.select(Exposure).where(Exposure.provenance_id == prov.id)).all()
            if len(exp) == 0:
                dbsession.delete(prov)
        dbsession.commit()

        objects = get_all_database_objects(session=dbsession)
        any_objects = False
        for Class, ids in objects.items():
            # TODO: check that surviving provenances have test_parameter
            if Class.__name__ in ['CodeVersion', 'CodeHash', 'SensorSection', 'CatalogExcerpt', 'Provenance']:
                _logger.info(f'There are {len(ids)} {Class.__name__} objects in the database. These are OK to stay.')
            elif len(ids) > 0:
                print(
                    f'There are {len(ids)} {Class.__name__} objects in the database. Please make sure to cleanup!'
                )
                for id in ids:
                    obj = session.scalars(sa.select(Class).where(Class.id == id)).first()
                    print(f'  {obj}')
                    any_objects = True

        # delete the CodeVersion object (this should remove all provenances as well)
        dbsession.execute(sa.delete(CodeVersion).where(CodeVersion.id == 'test_v1.0.0'))

        dbsession.commit()

        # comment this line out if you just want tests to pass quietly
        if any_objects:
            raise RuntimeError('There are objects in the database. Some tests are not properly cleaning up!')


# data that is included in the repo and should be available for tests
@pytest.fixture(scope="session")
def persistent_dir():
    return os.path.join(CODE_ROOT, 'data')


# this is a cache folder that should survive between test runs
@pytest.fixture(scope="session", autouse=True)
def cache_dir():
    path = os.path.join(CODE_ROOT, 'data/cache')
    os.makedirs(path, exist_ok=True)
    return path


# this will be configured to FileOnDiskMixin.local_path, and used as temporary data location
@pytest.fixture(scope="session")
def data_dir():
    temp_data_folder = FileOnDiskMixin.local_path
    os.makedirs(temp_data_folder, exist_ok=True)
    with open(os.path.join(temp_data_folder, 'placeholder'), 'w'):
        pass  # make an empty file inside this folder to make sure it doesn't get deleted on "remove_data_from_disk"

    # print(f'temp_data_folder: {temp_data_folder}')

    yield temp_data_folder

    # remove all the files created during tests
    # make sure the test config is pointing the data_dir
    # to a different location than the rest of the data
    # shutil.rmtree(temp_data_folder)


@pytest.fixture(scope="session")
def blocking_plots():
    """
    Control how and when plots will be generated.
    There are three options for the environmental variable "INTERACTIVE".
     - It is not set: do not make any plots. blocking_plots returns False.
     - It is set to a False value: make plots, but save them, and do not show on screen/block execution.
       In this case the blocking_plots returns False, but the tests that skip if INTERACTIVE is None will run.
     - It is set to a True value: make the plots, but stop the test execution until the figure is closed.

    If a test only makes plots and does not test functionality, it should be marked with
    @pytest.mark.skipif( os.getenv('INTERACTIVE') is None, reason='Set INTERACTIVE to run this test' )

    If a test makes a diagnostic plot, that is only ever used to visually inspect the results,
    then it should be surrounded by an if blocking_plots: statement. It will only run in interactive mode.

    If a test makes a plot that should be saved to disk, it should either have the skipif mentioned above,
    or have an if os.getenv('INTERACTIVE'): statement surrounding the plot itself.
    You may want to add plt.show(block=blocking_plots) to allow the figure to stick around in interactive mode,
    on top of saving the figure at the end of the test.
    """
    import matplotlib
    backend = matplotlib.get_backend()

    # make sure there's a folder to put the plots in
    if not os.path.isdir(os.path.join(CODE_ROOT, 'tests/plots')):
        os.makedirs(os.path.join(CODE_ROOT, 'tests/plots'))

    inter = os.getenv('INTERACTIVE', False)
    if isinstance(inter, str):
        inter = inter.lower() in ('true', '1', 'on', 'yes')

    if not inter:  # for non-interactive plots, use headless plots that just save to disk
        # ref: https://stackoverflow.com/questions/15713279/calling-pylab-savefig-without-display-in-ipython
        matplotlib.use("Agg")

    yield inter

    matplotlib.use(backend)


def rnd_str(n):
    return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


@pytest.fixture(autouse=True)
def test_config():
    return Config.get()


@pytest.fixture(scope="session", autouse=True)
def code_version():
    with SmartSession() as session:
        cv = session.scalars(sa.select(CodeVersion).where(CodeVersion.id == 'test_v1.0.0')).first()
        if cv is None:
            cv = CodeVersion(id="test_v1.0.0")
            cv.update()
            session.add( cv )
            session.commit()
        cv = session.scalars(sa.select(CodeVersion).where(CodeVersion.id == 'test_v1.0.0')).first()

    yield cv


@pytest.fixture
def provenance_base(code_version):
    with SmartSession() as session:
        code_version = session.merge(code_version)
        p = Provenance(
            process="test_base_process",
            code_version=code_version,
            parameters={"test_parameter": uuid.uuid4().hex},
            upstreams=[],
            is_testing=True,
        )
        p.update_id()
        p = p.recursive_merge(session)

        session.commit()

    yield p

    with SmartSession() as session:
        session.delete(p)
        session.commit()


@pytest.fixture
def provenance_extra( provenance_base ):
    with SmartSession() as session:
        provenance_base = provenance_base.recursive_merge(session)
        p = Provenance(
            process="test_base_process",
            code_version=provenance_base.code_version,
            parameters={"test_parameter": uuid.uuid4().hex},
            upstreams=[provenance_base],
            is_testing=True,
        )
        p.update_id()
        p = p.recursive_merge(session)
        session.commit()

    yield p

    with SmartSession() as session:
        session.delete(p)
        session.commit()


# use this to make all the pre-committed Image fixtures
@pytest.fixture(scope="session")
def provenance_preprocessing(code_version):
    with SmartSession() as session:
        code_version = session.merge(code_version)
        p = Provenance(
            process="preprocessing",
            code_version=code_version,
            parameters={"test_parameter": "test_value"},
            upstreams=[],
            is_testing=True,
        )
        p.update_id()

        p = p.recursive_merge(session)
        session.commit()

    yield p

    with SmartSession() as session:
        session.delete(p)
        session.commit()


@pytest.fixture
def archive(test_config):
    archive_specs = test_config.value('archive')

    if archive_specs is None:
        raise ValueError( "archive in config is None" )
    archive = Archive( **archive_specs )
    yield archive

    try:
        # To tear down, we need to blow away the archive server's directory.
        # For the test suite, we've also mounted that directory locally, so
        # we can do that
        archivebase = f"{test_config.value('archive.local_read_dir')}/{test_config.value('archive.path_base')}"
        try:
            shutil.rmtree( archivebase )
        except FileNotFoundError:
            pass

    except Exception as e:
        warnings.warn(str(e))


@pytest.fixture( scope="module" )
def catexp(data_dir, persistent_dir):
    if not os.path.isfile(os.path.join(data_dir, "Gaia_DR3_151.0926_1.8312_17.0_19.0.fits")):
        shutil.copy2(
            os.path.join(persistent_dir, "test_data/Gaia_DR3_151.0926_1.8312_17.0_19.0.fits"),
            os.path.join(data_dir, "Gaia_DR3_151.0926_1.8312_17.0_19.0.fits")
        )

    filepath = os.path.join(data_dir, "Gaia_DR3_151.0926_1.8312_17.0_19.0.fits")

    yield CatalogExcerpt.create_from_file( filepath, 'GaiaDR3' )

    if os.path.isfile(filepath):
        os.remove(filepath)
