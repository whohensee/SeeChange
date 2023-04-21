import os

from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy import func, orm

from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declared_attr, declarative_base
from sqlalchemy_utils import database_exists, create_database

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
        engine = sa.create_engine(url, future=True, poolclass=sa.pool.NullPool)

        _Session = sessionmaker(bind=engine, expire_on_commit=True)
    return _Session()


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
    allowed_dirs = [
        cfg.value('path.data_root'),
        cfg.value('path.data_temp'),
    ]

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
        super().__init__(**kwargs)
        self.from_db = False  # let users know this object was newly created
        for k, v in kwargs.items():
            setattr(self, k, v)

    @orm.reconstructor
    def init_on_load(self):
        self.from_db = True  # let users know this object was loaded from the database


Base = declarative_base(cls=SeeChangeBase)


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
