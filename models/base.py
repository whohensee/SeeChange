import os
import inspect

from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy import func, orm

from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.declarative import declared_attr

import util.config as config

utcnow = func.timezone("UTC", func.current_timestamp())


# this is the root SeeChange folder
CODE_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

_engine = None
_Session = None


def Session():
    """
    Make a session if it doesn't already exist.
    Use this in interactive sessions where you don't
    want to open the session as a context manager.
    If you want to use it in a context manager
    (the "with" statement where it closes at the
    end of the context) use SmartSession() instead.

    Returns
    -------
    sqlalchemy.orm.session.Session
        A session object that doesn't automatically close.
    """
    global _Session, _engine

    if _Session is None:
        cfg = config.Config.get()
        url = (f'{cfg.value("db.engine")}://{cfg.value("db.user")}:{cfg.value("db.password")}'
               f'@{cfg.value("db.host")}:{cfg.value("db.port")}/{cfg.value("db.database")}')
        _engine = sa.create_engine(url, future=True, poolclass=sa.pool.NullPool)

        _Session = sessionmaker(bind=_engine, expire_on_commit=True)

    session = _Session()

    return session


@contextmanager
def SmartSession(input_session=None):
    """
    Return a Session() instance that may or may not
    be inside a context manager.

    If the input is already a session, just return that.
    If the input is None, create a session that would
    close at the end of the life of the calling scope.
    """
    global _Session, _engine

    # open a new session and close it when outer scope is done
    if input_session is None:

        with Session() as session:
            yield session

    # return the input session with the same scope as given
    elif isinstance(input_session, sa.orm.session.Session):
        yield input_session

    # wrong input type
    else:
        raise TypeError(
            "input_session must be a sqlalchemy session or None"
        )


def safe_mkdir(path):

    cfg = config.Config.get()
    allowed_dirs = []
    if cfg.value('path.data_root') is not None:
        allowed_dirs.append(cfg.value('path.data_root'))
    if cfg.value('path.data_temp') is not None:
        allowed_dirs.append(cfg.value('path.data_temp'))
    if cfg.value('path.server_data') is not None:
        allowed_dirs.append(cfg.value('path.server_data'))

    ok = False

    for d in allowed_dirs:
        parent = os.path.realpath(os.path.abspath(d))
        child = os.path.realpath(os.path.abspath(path))

        if os.path.commonpath([parent]) == os.path.commonpath([parent, child]):
            ok = True
            break

    if not ok:
        err_str = "Cannot make a new folder not inside the following folders: "
        err_str += "\n".join(allowed_dirs)
        err_str += f"\n\nAttempted folder: {path}"
        raise ValueError(err_str)

    # if the path is ok, also make the subfolders
    os.makedirs(path, exist_ok=True)


class SeeChangeBase:
    """Base class for all SeeChange classes."""

    id = sa.Column(
        sa.BigInteger,
        primary_key=True,
        index=True,
        autoincrement=True,
        doc="Unique identifier for this dataset",
    )

    created_at = sa.Column(
        sa.DateTime,
        nullable=False,
        default=utcnow,
        index=True,
        doc="UTC time of insertion of object's row into the database.",
    )

    modified = sa.Column(
        sa.DateTime,
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
        doc="UTC time the object's row was last modified in the database.",
    )

    def __init__(self, **kwargs):
        self.from_db = False  # let users know this object was newly created
        for k, v in kwargs.items():
            setattr(self, k, v)

    @orm.reconstructor
    def init_on_load(self):
        self.from_db = True  # let users know this object was loaded from the database

    def get_attribute_list(self):
        """
        Get a list of all attributes of this object,
        not including internal SQLAlchemy attributes,
        and database level attributes like id, created_at, etc.
        """
        attrs = [
            a for a in self.__dict__.keys()
            if (
                a not in ['_sa_instance_state', 'id', 'created_at', 'modified', 'from_db']
                and not callable(getattr(self, a))
                and not isinstance(getattr(self, a), (
                    orm.collections.InstrumentedList, orm.collections.InstrumentedDict
                ))
            )
        ]

        return attrs


Base = declarative_base(cls=SeeChangeBase)


class FileOnDiskMixin:
    """
    Mixin for objects that refer to files on disk.

    Files are assumed to live in a remote server or on local disk.
    The path to both these locations is configurable and not stored on DB!
    Once the top level directory is set (locally and remotely),
    the object's path relative to either of those is saved as "filepath".

    If multiple files need to be copied or loaded, we can also append
    the array "filepath_extensions" to the filepath.
    These could be actual extensions as in:
    filepath = 'foo.fits' and filepath_extensions=['.bias', '.dark', '.flat']
    or they could be just a different part of the filepath itself:
    filepath = 'foo_' and filepath_extensions=['bias.fits.gz', 'dark.fits.gz', 'flat.fits.gz']

    If the filepath_extensions array is null, will just load a single file.
    If the filepath_extensions is an array, will load a list of files (even if length 1).

    When calling get_fullpath(), the object will first check if the file exists locally,
    and then it will download it from server if missing.
    If no remote server is defined in the config, this part is skipped.
    If you want to avoid downloading, use get_fullpath(download=False).
    If you want to always get a list of filepaths (even if filepath_extensions=None)
    use get_fullpath(as_list=True).
    If the file is missing locally, and downloading cannot proceed
    (because no server address is defined, or because the download=False flag is used,
    or because the file is missing from server), then the call to get_fullpath() will raise an exception.

    After all the downloading is done and the file(s) exist locally,
    the full path to the local file is returned.
    It is then up to the inheriting object (e.g., the Exposure or Image)
    to actually load the file from disk and figure out what to do with the data.

    The path to the local and server side data folders is saved
    in class variables, and must be initialized by the application
    when the app starts / when the config file is read.
    """
    cfg = config.Config.get()
    server_path = cfg.value('path.server_data')
    local_path = cfg.value('path.data_root')
    if local_path is None:
        local_path = cfg.value('path.data_temp')
    if local_path is None:
        local_path = os.path.join(CODE_ROOT, 'data')
    if not os.path.isdir(local_path):
        os.makedirs(local_path, exist_ok=True)

    filepath = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        unique=True,
        doc="Filename and path (relative to the data root) for a raw exposure. "
    )

    filepath_extensions = sa.Column(
        sa.ARRAY(sa.Text),
        nullable=True,
        doc=(
            "Filename extensions for raw exposure. "
            "Can contain any part of the filepath that isn't shared between files. "
        )
    )

    def __init__(self, *args, **kwargs):
        """
        Initialize an object that is associated with a file on disk.
        If giving a single unnamed argument, will assume that is the filepath.
        Note that the filepath should not include the global data path,
        but only a path relative to that. # TODO: remove the global path if filepath starts with it?

        Parameters
        ----------
        args: list
            List of arguments, should only contain one string as the filepath.
        kwargs: dict
            Dictionary of keyword arguments.
            These include:
            - filepath: str
                Use instead of the unnamed argument.
            - nofile: bool
                If True, will not require the file to exist on disk.
                That means it will not try to download it from server, either.
                This should be used only when creating a new object that will
                later be associated with a file on disk (or for tests).
                This property is NOT SAVED TO DB!
                Saving to DB should only be done when a file exists
                # TODO: add the check that file exists before committing?
        """
        if len(args) == 1 and isinstance(args[0], str):
            self.filepath = args[0]

        self.filepath = kwargs.pop('filepath', self.filepath)
        self.nofile = kwargs.pop('nofile', False)  # do not require a file to exist when making the exposure object

    def get_fullpath(self, download=True, as_list=False):
        """
        Get the full path of the file, or list of full paths
        of files if filepath_extensions is not None.
        If the server_path is defined, and download=True (default),
        the file will be downloaded from the server if missing.
        If the file is not found on server or locally, will
        raise a FileNotFoundError.
        When setting self.nofile=True, will not check if the file exists,
        or try to download it from server. The assumption is that an
        object with self.nofile=True will be associated with a file later on.

        If the file is found on the local drive, under the local_path,
        (either it was there or after it was downloaded)
        the full path is returned.
        The application is then responsible for loading the content
        of the file.

        When the filepath_extensions is None, will return a single string.
        When the filepath_extensions is an array, will return a list of strings.
        If as_list=False, will always return a list of strings,
        even if filepath_extensions is None.

        Parameters
        ----------
        download: bool
            Whether to download the file from server if missing.
            Must have server_path defined. Default is True.
        as_list: bool
            Whether to return a list of filepaths, even if filepath_extensions=None.
            Default is False.

        Returns
        -------
        str or list of str
            Full path to the file(s) on local disk.
        """
        if self.filepath_extensions is None:
            if as_list:
                return [self._get_fullpath_single(download)]
            else:
                return self._get_fullpath_single(download)
        else:
            return [self._get_fullpath_single(download, ext) for ext in self.filepath_extensions]

    def _get_fullpath_single(self, download=True, ext=None):
        """
        Get the full path of a single file.
        Will follow the same logic as get_fullpath(),
        of checking and downloading the file from the server
        if it is not on local disk.

        Parameters
        ----------
        download: bool
            Whether to download the file from server if missing.
            Must have server_path defined. Default is True.
        ext: str
            Extension to add to the filepath. Default is None.

        Returns
        -------
        str
            Full path to the file on local disk.
        """
        if not self.nofile and self.local_path is None:
            raise ValueError("Local path not defined!")

        fname = self.filepath
        if ext is not None:
            fname += ext

        fullname = os.path.join(self.local_path, fname)

        if not self.nofile and not os.path.exists(fullname) and download and self.server_path is not None:
            self._download_file(fname)

        if not self.nofile and not os.path.exists(fullname):
            raise FileNotFoundError(f"File {fullname} not found!")

        return fullname

    def _download_file(self, filepath):
        """
        Search and download the file from a remote server.
        The server_path must be defined on the class
        (e.g., by setting a value for it from the config).
        The download can be a simple copy from an address
        (e.g., a join of server_path and filepath)
        or it can be a more complicated request.
        This depends on the exact configuration.
        """

        # TODO: finish this
        raise NotImplementedError('Downloading files from server is not yet implemented!')


class SpatiallyIndexed:
    """A mixin for tables that have ra and dec fields indexed via q3c."""

    ra = sa.Column(sa.Double, nullable=False, doc='Right ascension in degrees')
    dec = sa.Column(sa.Double, nullable=False, doc='Declination in degrees')

    @declared_attr
    def __table_args__(cls):
        tn = cls.__tablename__
        return (
            sa.Index(f"{tn}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
        )


if __name__ == "__main__":
    pass
