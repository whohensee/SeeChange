import os
import git
from collections import defaultdict
import numpy as np
from datetime import datetime

import sqlalchemy as sa

from astropy.time import Time

from models.base import SmartSession


def get_git_hash():
    """
    Get the commit hash of the current git repo.

    If the environmental variable GITHUB_SHA is set,
    use that as the git commit hash.
    If not, try to find the git commit hash of the current repo.
    If all these methods fail, quietly return None.
    """

    git_hash = os.getenv('GITHUB_SHA')
    if git_hash is None:
        try:
            repo = git.Repo(search_parent_directories=True)
            git_hash = repo.head.object.hexsha
        except Exception:
            git_hash = None

    return git_hash


def get_latest_provenance(process_name, session=None):
    """
    Find the provenance object that fits the process_name
    that is the most recent.
    # TODO: we need to think about what "most recent" means.

    Parameters
    ----------
    process_name: str
        Name of the process that created this provenance object.
        Examples can include: "calibration", "subtraction", "source extraction" or just "level1".
    session: sqlalchemy.orm.session.Session or SmartSession

    Returns
    -------
    Provenance
        The most recent provenance object that matches the process_name.
        If not found, returns None.
    """
    # importing the models here to avoid circular imports
    from models.base import SmartSession
    from models.provenance import Provenance

    with SmartSession(session) as session:
        prov = session.scalars(
            sa.select(Provenance).where(
                Provenance.process == process_name
            ).order_by(Provenance.id.desc())
        ).first()

    return prov


def parse_dateobs(dateobs=None, output='datetime'):
    """
    Parse the dateobs, that can be a float, string, datetime or Time object.
    The output is datetime by default, but can be any of the above types.
    If the dateobs is None, the current time will be returned.
    If int or float, will assume MJD (or JD if bigger than 2400000).

    Parameters
    ----------
    dateobs: float, str, datetime, Time or None
        The dateobs to parse.
    output: str
        Choose one of the output formats:
        'datetime', 'Time', 'float', 'str'.

    Returns
    -------
    datetime, Time, float or str
    """
    if dateobs is None:
        dateobs = Time.now()
    elif isinstance(dateobs, (int, float)):
        if dateobs > 2400000:
            dateobs = Time(dateobs, format='jd')
        else:
            dateobs = Time(dateobs, format='mjd')
    elif isinstance(dateobs, str):
        if dateobs == 'now':
            dateobs = Time.now()
        else:
            dateobs = Time(dateobs)
    elif isinstance(dateobs, datetime):
        dateobs = Time(dateobs)
    else:
        raise ValueError(f'Cannot parse dateobs of type {type(dateobs)}')

    if output == 'datetime':
        return dateobs.datetime
    elif output == 'Time':
        return dateobs
    elif output == 'float':
        return dateobs.mjd
    elif output == 'str':
        return dateobs.isot
    else:
        raise ValueError(f'Unknown output type {output}')


def parse_session(*args, **kwargs):
    """
    Parse the arguments and keyword arguments to find a SmartSession or SQLAlchemy session.
    If one of the kwargs is called "session" that value will be returned.
    Otherwise, if any of the unnamed arguments is a session, the last one will be returned.
    If neither of those are found, None will be returned.
    Will also return the args and kwargs with any sessions removed.

    Parameters
    ----------
    args: list
        List of unnamed arguments
    kwargs: dict
        Dictionary of named arguments

    Returns
    -------
    args: list
        List of unnamed arguments with any sessions removed.
    kwargs: dict
        Dictionary of named arguments with any sessions removed.
    session: SmartSession or SQLAlchemy session or None
        The session found in the arguments or kwargs.
    """
    session = None
    sessions = [arg for arg in args if isinstance(arg, sa.orm.session.Session)]
    if len(sessions) > 0:
        session = sessions[-1]
    args = [arg for arg in args if not isinstance(arg, sa.orm.session.Session)]

    for key in kwargs.keys():
        if key in ['session']:
            if not isinstance(kwargs[key], sa.orm.session.Session):
                raise ValueError(f'Session must be a sqlalchemy.orm.session.Session, got {type(kwargs[key])}')
            session = kwargs.pop(key)

    return args, kwargs, session


