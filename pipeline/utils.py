import git
from collections import defaultdict
import numpy as np

import sqlalchemy as sa


def get_git_hash():
    """Get the commit hash of the current git repo. """
    # TODO: what if we are running on a production server
    #  that doesn't have git? Consider replacing this with
    #  an environmental variable that is automatically updated
    #  with the current git hash.

    repo = git.Repo(search_parent_directories=True)
    git_hash = repo.head.object.hexsha

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


def normalize_header_key(key):
    """
    Normalize the header key to be all uppercase and
    remove spaces and underscores.
    """
    return key.upper().replace(' ', '').replace('_', '')