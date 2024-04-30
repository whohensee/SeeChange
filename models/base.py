import sys
import os
import math
import types
import hashlib
import pathlib
import logging
import json
import shutil
import datetime
from uuid import UUID

from contextlib import contextmanager
import numpy as np

from astropy.coordinates import SkyCoord

import sqlalchemy as sa
from sqlalchemy import func, orm

from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.orm.exc import DetachedInstanceError
from sqlalchemy.dialects.postgresql import UUID as sqlUUID
from sqlalchemy.dialects.postgresql import array as sqlarray
from sqlalchemy.schema import CheckConstraint

from models.enums_and_bitflags import (
    data_badness_dict,
    data_badness_inverse,
    string_to_bitflag,
    bitflag_to_string,
)

import util.config as config
from util.archive import Archive

utcnow = func.timezone("UTC", func.current_timestamp())


_logger = logging.getLogger("main")
if len(_logger.handlers) == 0:
    _logout = logging.StreamHandler( sys.stderr )
    _logger.addHandler( _logout )
    _formatter = logging.Formatter( f"[%(asctime)s - %(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S" )
    _logout.setFormatter( _formatter )
# _logout.setLevel( logging.DEBUG )

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

        _Session = sessionmaker(bind=_engine, expire_on_commit=False)

    session = _Session()

    return session


@contextmanager
def SmartSession(*args):
    """
    Return a Session() instance that may or may not
    be inside a context manager.

    If a given input is already a session, just return that.
    If all inputs are None, create a session that would
    close at the end of the life of the calling scope.
    """
    global _Session, _engine

    for arg in args:
        if isinstance(arg, sa.orm.session.Session):
            yield arg
            return
        if arg is None:
            continue
        else:
            raise TypeError(
                "All inputs must be sqlalchemy sessions or None. "
                f"Instead, got {args}"
            )

    # none of the given inputs managed to satisfy any of the conditions...
    # open a new session and close it when outer scope is done
    with Session() as session:
        yield session


def db_stat(obj):
    """Check the status of an object. It can be one of: transient, pending, persistent, deleted, detached."""
    for word in ['transient', 'pending', 'persistent', 'deleted', 'detached']:
        if getattr(sa.inspect(obj), word):
            return word


def get_all_database_objects(display=False, session=None):
    """Find all the objects and their associated IDs in the database.

    WARNING: this is only meant to be used on test databases.
    Calling this on a production database would be very slow!

    Parameters
    ----------
    display: bool (optional)
        If True, print the results to stdout.
    session: sqlalchemy.orm.session.Session (optional)
        The session to use. If None, a new session will be created.

    Returns
    -------
    dict
        A dictionary with the object class names as keys and the IDs list as values.

    """
    from models.provenance import Provenance, CodeVersion, CodeHash
    from models.datafile import DataFile
    from models.exposure import Exposure
    from models.image import Image
    from models.source_list import SourceList
    from models.psf import PSF
    from models.world_coordinates import WorldCoordinates
    from models.zero_point import ZeroPoint
    from models.cutouts import Cutouts
    from models.measurements import Measurements
    from models.calibratorfile import CalibratorFile
    from models.catalog_excerpt import CatalogExcerpt
    from models.reference import Reference
    from models.instrument import SensorSection

    models = [
        CodeHash, CodeVersion, Provenance, DataFile, Exposure, Image,
        SourceList, PSF, WorldCoordinates, ZeroPoint, Cutouts, Measurements,
        CalibratorFile, CatalogExcerpt, Reference, SensorSection
    ]

    output = {}
    with SmartSession(session) as session:
        for model in models:
            object_ids = session.scalars(sa.select(model.id)).all()
            output[model] = object_ids

            if display:
                _logger.debug(f"{model.__name__:16s}: ", end='')
                for obj_id in object_ids:
                    _logger.debug(obj_id, end=', ')
                _logger.debug()

    return output


def safe_merge(session, obj, db_check_att='filepath'):
    """
    Only merge the object if it has a valid ID,
    and if it does not exist on the session.
    Otherwise, return the object itself.

    Parameters
    ----------
    session: sqlalchemy.orm.session.Session
        The session to use for the merge.
    obj: SeeChangeBase
        The object to merge.
    db_check_att: str (optional)
        If given, will check if an object with this attribute
        exists in the DB before merging. If it does, it will
        merge the new object with the existing object's ID.
        Default is to check against the "filepath" attribute,
        which will fail quietly if the object doesn't have
        this attribute.
        This check only occurs for objects without an id.

    Returns
    -------
    obj: SeeChangeBase
        The merged object, or the unmerged object
        if it is already on the session or if it
        doesn't have an ID.
    """
    if obj is None:  # given None, return None
        return None

    # if there is no ID, maybe need to check another attribute
    if db_check_att is not None and hasattr(obj, db_check_att):
        existing = session.scalars(
            sa.select(type(obj)).where(getattr(type(obj), db_check_att) == getattr(obj, db_check_att))
        ).first()
        if existing is not None:  # this object already has a copy on the DB!
            obj.id = existing.id  # make sure to update existing row with new data
            obj.created_at = existing.created_at  # make sure to keep the original creation time
    return session.merge(obj)


class SeeChangeBase:
    """Base class for all SeeChange classes."""

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

    type_annotation_map = { UUID: sqlUUID }

    def __init__(self, **kwargs):
        self.from_db = False  # let users know this object was newly created

        if hasattr(self, '_bitflag'):
            self._bitflag = 0
        if hasattr(self, 'upstream__bitflag'):
            self._upstream_bitflag = 0

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

    def set_attributes_from_dict( self, dictionary ):
        """Set all atributes of self from a dictionary, excepting existing attributes that are methods.

        Parameters
        ----------
        dictionary: dict
          A dictionary of attributes to set in self

        """
        for key, value in dictionary.items():
            if hasattr(self, key):
                if type( getattr( self, key ) ) != types.MethodType:
                    setattr(self, key, value)

    def safe_merge(self, session, db_check_att='filepath'):
        """Safely merge this object into the session. See safe_merge()."""
        return safe_merge(session, self, db_check_att=db_check_att)

    def get_upstreams(self, session=None):
        """Get all data products that were directly used to create this object (non-recursive)."""
        raise NotImplementedError('get_upstreams not implemented for this class')

    def get_downstreams(self, session=None):
        """Get all data products that were created directly from this object (non-recursive)."""
        raise NotImplementedError('get_downstreams not implemented for this class')

    def delete_from_database(self, session=None, commit=True):
        """Remove the object from the database.

        This does not remove any associated files (if this is a FileOnDiskMixin)
        and does not remove the object from the archive.

        Parameters
        ----------
        session: sqlalchemy session
            The session to use for the deletion. If None, will open a new session,
            which will also close at the end of the call.
        commit: bool
            Whether to commit the deletion to the database.
            Default is True. When session=None then commit must be True,
            otherwise the session will exit without committing
            (in this case the function will raise a RuntimeException).
        """
        if session is None and not commit:
            raise RuntimeError("When session=None, commit must be True!")

        with SmartSession(session) as session:
            info = sa.inspect(self)
            need_commit = False
            if info.persistent:
                session.delete(self)
                need_commit = True
            elif info.pending:
                session.expunge(self)
                need_commit = True
            elif info.detached:
                session.execute(sa.delete(self.__class__).where(self.__class__.id == self.id))
                need_commit = True

            if commit and need_commit:
                session.commit()

    def to_dict(self):
        """Translate all the SQLAlchemy columns into a dictionary.

        This can be used, e.g., to cache a row from DB to a file.
        This will include foreign keys, which are not guaranteed
        to remain the same when loading into a new database,
        so all the relationships the object has should be
        reconstructed manually when loading it from the dictionary.

        This will not include any of the attributes of the object
        that are not saved into the database, but those have to
        be lazy loaded anyway, as they are not persisted.

        Will convert non-standard data types:
        UUID will be converted to string (using the .hex attribute).
        Numpy arrays are replaced by lists.

        To reload, use the from_dict() method:
        reloaded_object = MyClass.from_dict( output_dict )
        This will reconstruct the object, including the non-standard
        data types like the UUID.
        """
        output = {}
        for key in sa.inspect(self).mapper.columns.keys():
            value = getattr(self, key)
            # get rid of numpy types
            if isinstance(value, np.number):
                value = value.item()  # convert numpy number to python primitive
            if isinstance(value, list):
                value = [v.item() if isinstance(v, np.number) else v for v in value]
            if isinstance(value, dict):
                value = {k: v.item() if isinstance(v, np.number) else v for k, v in value.items()}

            if key == 'md5sum' and value is not None:
                if isinstance(value, UUID):
                    value = value.hex
            if key == 'md5sum_extensions' and value is not None:
                if isinstance(value, list):
                    value = [v.hex if isinstance(v, UUID) else v for v in value]

            if isinstance(value, np.ndarray) and key in [
                'aper_rads', 'aper_radii', 'aper_cors', 'aper_cor_radii',
                'flux_apertures', 'flux_apertures_err', 'area_apertures',
            ]:
                value = list(value)

            if key in ['modified', 'created_at'] and isinstance(value, datetime.datetime):
                value = value.isoformat()

            if isinstance(value, (datetime.datetime, np.ndarray)):
                raise TypeError('Found some columns with non-standard types. Please parse all columns! ')

            output[key] = value

        return output

    @classmethod
    def from_dict(cls, dictionary):
        """Convert a dictionary into a new object. """
        dictionary.pop('modified')  # we do not want to recreate the object with an old "modified" time

        md5sum = dictionary.get('md5sum', None)
        if md5sum is not None:
            dictionary['md5sum'] = UUID(md5sum)

        md5sum_extensions = dictionary.get('md5sum_extensions', None)
        if md5sum_extensions is not None:
            new_extensions = [UUID(md5) for md5 in md5sum_extensions if md5 is not None]
            dictionary['md5sum_extensions'] = new_extensions

        aper_rads = dictionary.get('aper_rads', None)
        if aper_rads is not None:
            dictionary['aper_rads'] = np.array(aper_rads)

        aper_cors = dictionary.get('aper_cors', None)
        if aper_cors is not None:
            dictionary['aper_cors'] = np.array(aper_cors)

        aper_cor_radii = dictionary.get('aper_cor_radii', None)
        if aper_cor_radii is not None:
            dictionary['aper_cor_radii'] = np.array(aper_cor_radii)

        created_at = dictionary.get('created_at', None)
        if created_at is not None:
            dictionary['created_at'] = datetime.datetime.fromisoformat(created_at)

        return cls(**dictionary)

    def to_json(self, filename):
        """Translate a row object's column values to a JSON file.

        See the description of to_dict() for more details.

        Parameters
        ----------
        filename: str or path
            The path to the output JSON file.
        """
        with open(filename, 'w') as fp:
            try:
                json.dump(self.to_dict(), fp, indent=2)
            except:
                raise

    def copy_to_cache(self, cache_dir, filepath=None):
        """Save a copy of the object (and associated files) into a cache directory.

        If the object is a FileOnDiskMixin, then the file(s) pointed by get_fullpath()
        will be copied to the cache directory with their original names,
        unless filepath is specified, in which case the cached files will
        have a different name than the files in the data folder (and the database filepath).
        The filepath (with optional first extension) will be used to create a JSON file
        which holds the object's column attributes (i.e., only those that are
        database persistent).

        If caching a non-FileOnDiskMixin object, the filepath argument must be given,
        because it is used to name the JSON file.

        Parameters
        ----------
        cache_dir: str or path
            The path to the cache directory.
        filepath: str or path (optional)
            Must be given if the object is not a FileOnDiskMixin.
            If it is a FileOnDiskMixin, it will be used to name
            the data files and the JSON file in the cache folder.

        Returns
        -------
        str
            The full path to the output json file.
        """
        if filepath is not None and filepath.endswith('.json'):  # remove .json if it exists
            filepath = filepath[:-5]

        json_filepath = filepath
        if not isinstance(self, FileOnDiskMixin):
            if filepath is None:
                raise ValueError("filepath must be given when caching a non FileOnDiskMixin object")

        else:  # it is a FileOnDiskMixin
            if filepath is None:  # use the FileOnDiskMixin filepath as default
                filepath = self.filepath  # use this filepath for the data files
                json_filepath = self.filepath  # use the same filepath for the json file too
            if (
                    self.filepath_extensions is not None and
                    len(self.filepath_extensions) > 0 and
                    not json_filepath.endswith(self.filepath_extensions[0])
            ):
                    json_filepath += self.filepath_extensions[0]  # only append this extension to the json filename

            for i, source_f in enumerate(self.get_fullpath(as_list=True)):
                if source_f is None:
                    continue
                target_f = os.path.join(cache_dir, filepath)
                if self.filepath_extensions is not None and i < len(self.filepath_extensions):
                    target_f += self.filepath_extensions[i]
                _logger.debug(f"Copying {source_f} to {target_f}")
                os.makedirs(os.path.dirname(target_f), exist_ok=True)
                shutil.copy2(source_f, target_f)

        # attach the cache_dir and the .json extension if needed
        json_filepath = os.path.join(cache_dir, json_filepath)
        if not json_filepath.endswith('.json'):
            json_filepath += '.json'
        self.to_json(json_filepath)

        return json_filepath

    @classmethod
    def copy_list_to_cache(cls, obj_list, cache_dir, filepath=None):
        """Copy a list of objects to the cache directory.

        The first object on the list will be used to copy any associated files
        (if it is a FileOnDiskMixin). The filepath argument must be given
        if the objects are not FileOnDiskMixin.
        The type and filepath of all objects on the list must be the same!

        The object's column data is saved into the JSON file as a list of dictionaries.

        Parameters
        ----------
        obj_list: list
            A list of objects to save to the cache directory.
        cache_dir: str or path
            The path to the cache directory.
        filepath: str or path (optional)
            Must be given if the objects are not FileOnDiskMixin.
            If it is a FileOnDiskMixin, it will be used to name
            the data files and the JSON file in the cache folder.

        Returns
        -------
        str
            The full path to the output JSON file.
        """
        types = set([type(obj) for obj in obj_list])
        if len(types) != 1:
            raise ValueError("All objects must be of the same type!")

        filepaths = set([getattr(obj, 'filepath', None) for obj in obj_list])
        if len(filepaths) != 1:
            raise ValueError("All objects must have the same filepath!")

        # save the JSON file and copy associated files
        json_filepath = obj_list[0].copy_to_cache(cache_dir, filepath=filepath)

        # overwrite the JSON file with the list of dictionaries
        with open(json_filepath, 'w') as fp:
            json.dump([obj.to_dict() for obj in obj_list], fp, indent=2)

        return json_filepath

    @classmethod
    def copy_from_cache(cls, cache_dir, filepath):
        """Copy and reconstruct an object from the cache directory.

        Will need the JSON file that contains all the column attributes of the file.
        Once those are successfully loaded, and if the object is a FileOnDiskMixin,
        it will be able to figure out where all the associated files are saved
        based on the filepath and extensions in the JSON file.
        Those files will be copied into the current data directory
        (i.e., that pointed to by FileOnDiskMixin.local_path).
        The reconstructed object should be correctly associated
        with its files but will not necessarily have the correct
        relationships to other objects.

        Parameters
        ----------
        cache_dir: str or path
            The path to the cache directory.
        filepath: str or path
            The name of the JSON file that holds the column attributes.

        Returns
        -------
        output: SeeChangeBase
            The reconstructed object, of the same type as the class.
        """
        # allow user to give an absolute path, so long as it is in the cache dir
        if filepath.startswith(cache_dir):
            filepath = filepath[len(cache_dir) + 1:]

        # allow the user to give the filepath with or without the .json extension
        if filepath.endswith('.json'):
            filepath = filepath[:-5]

        full_path = os.path.join(cache_dir, filepath)
        with open(full_path + '.json', 'r') as fp:
            json_dict = json.load(fp)

        output = cls.from_dict(json_dict)

        # copy any associated files
        if isinstance(output, FileOnDiskMixin):
            # if fullpath ends in filepath_extensions[0]
            if (
                    output.filepath_extensions is not None and
                    output.filepath_extensions[0] is not None and
                    full_path.endswith(output.filepath_extensions[0])
            ):
                full_path = full_path[:-len(output.filepath_extensions[0])]

            for i, target_f in enumerate(output.get_fullpath(as_list=True)):
                if target_f is None:
                    continue
                source_f = os.path.join(cache_dir, full_path)
                if output.filepath_extensions is not None and i < len(output.filepath_extensions):
                    source_f += output.filepath_extensions[i]
                _logger.debug(f"Copying {source_f} to {target_f}")
                os.makedirs(os.path.dirname(target_f), exist_ok=True)
                shutil.copyfile(source_f, target_f)

        return output

    @classmethod
    def copy_list_from_cache(cls, cache_dir, filepath):
        """Copy and reconstruct a list of objects from the cache directory.

        Will need the JSON file that contains all the column attributes of the file.
        Once those are successfully loaded, and if the object is a FileOnDiskMixin,
        it will be able to figure out where all the associated files are saved
        based on the filepath and extensions in the JSON file.

        Parameters
        ----------
        cache_dir: str or path
            The path to the cache directory.
        filepath: str or path
            The name of the JSON file that holds the column attributes.

        Returns
        -------
        output: list
            The list of reconstructed objects, of the same type as the class.
        """
        # allow user to give an absolute path, so long as it is in the cache dir
        if filepath.startswith(cache_dir):
            filepath = filepath[len(cache_dir) + 1:]

        # allow the user to give the filepath with or without the .json extension
        if filepath.endswith('.json'):
            filepath = filepath[:-5]

        full_path = os.path.join(cache_dir, filepath)
        with open(full_path + '.json', 'r') as fp:
            json_list = json.load(fp)

        output = []
        for obj_dict in json_list:
            output.append(cls.from_dict(obj_dict))

        if len(output) == 0:
            return []

        if isinstance(output[0], FileOnDiskMixin):
            # if fullpath ends in filepath_extensions[0]
            if (
                    output[0].filepath_extensions is not None and
                    output[0].filepath_extensions[0] is not None and
                    full_path.endswith(output[0].filepath_extensions[0])
            ):
                full_path = full_path[:-len(output[0].filepath_extensions[0])]

            for i, target_f in enumerate(output[0].get_fullpath(as_list=True)):
                if target_f is None:
                    continue
                source_f = os.path.join(cache_dir, full_path)
                if output[0].filepath_extensions is not None and i < len(output[0].filepath_extensions):
                    source_f += output[0].filepath_extensions[i]
                _logger.debug(f"Copying {source_f} to {target_f}")
                os.makedirs(os.path.dirname(target_f), exist_ok=True)
                shutil.copyfile(source_f, target_f)

        return output


Base = declarative_base(cls=SeeChangeBase)

ARCHIVE = None


def get_archive_object():
    """Return a global archive object. If it doesn't exist, create it based on the current config. """
    global ARCHIVE
    if ARCHIVE is None:
        cfg = config.Config.get()
        archive_specs = cfg.value('archive', None)
        if archive_specs is not None:
            ARCHIVE = Archive(**archive_specs)
    return ARCHIVE


class FileOnDiskMixin:
    """Mixin for objects that refer to files on disk.

    Files are assumed to live on the local disk (underneath the
    configured path.data_root), and optionally on a archive server
    (configured through the subproperties of "archive" in the yaml
    config file).  The property filepath is the path relative to the
    root in both cases.

    If there is a single file associated with this entry, then filepath
    has the name of that file.  md5sum holds a checksum for the file,
    *if* it has been correctly saved to the archive.  If the file has
    not been saved to the archive, md5sum is null.  In this case,
    filepath_extensions and md5sum_extensions will be null.  (Exception:
    if you are configured with a null archive (config parameter archive
    in the yaml config file is null), then md5sum will be set when the
    image is saved to disk, instead of when it's saved to the archive.)

    If there are multiple files associated with this entry, then
    filepath is the beginning of the names of all the files.  Each entry
    in filepath_extensions is then appended to filepath to get the
    actual path of the file.  For example, if an image file has the
    image itself, an associated weight, and an associated mask, then
    filepath might be "image" and filepath_extensions might be
    [".fits.fz", ".mask.fits.fz", ".weight.fits.fz"] to indicate that
    the three files image.fits.fz, image.mask.fits.fz, and
    image.weight.fz are all associated with this entry.  When
    filepath_extensions is non-null, md5sum should be null, and
    md5sum_extensions is an array with the same length as
    filepath_extensions.  (For extension files that have not yet been
    saved to the archive, that element of the md5sum_etensions array is
    null.)

    Saving data:

    Any object that implements this mixin must call this class' "save"
    method in order to save the data to disk.  (This may be through
    super() if the subclass has to do custom things.)  The save method
    of this class will save to the local filestore (underneath
    path.data_root), and also save it to the archive.  Once a file is
    saved on the archive, the md5sum (or md5sum_extensions) field in the
    database record is updated.  (If the file has not been saved to the
    archive, then the md5sum and md5sum_extensions fields will be null.)

    Loading data:

    When calling get_fullpath(), the object will first check if the file
    exists locally, and then it will import it from archive if missing
    (and if archive is defined).  If you want to avoid downloading, use
    get_fullpath(download=False) or get_fullpath(nofile=True).  (The
    latter case won't even try to find the file on the local disk, it
    will just tell you what the path should be.)  If you want to always
    get a list of filepaths (even if filepath_extensions=None) use
    get_fullpath(as_list=True).  If the file is missing locally, and
    downloading cannot proceed (because no archive is defined, or
    because the download=False flag is used, or because the file is
    missing from server), then the call to get_fullpath() will raise an
    exception (unless you use download=False or nofile=True).

    After all the pulling from the archive is done and the file(s) exist
    locally, the full (absolute) path to the local file is returned.  It
    is then up to the inheriting object (e.g., the Exposure or Image) to
    actually load the file from disk and figure out what to do with the
    data.

    The path to the local file store and the archive object are saved in
    class variables "local_path" and "archive" that are initialized from
    the config system the first time the class is loaded.

    """
    local_path = None
    temp_path = None

    @classmethod
    def configure_paths(cls):
        cfg = config.Config.get()
        cls.local_path = cfg.value('path.data_root', None)

        if cls.local_path is None:
            cls.local_path = cfg.value('path.data_temp', None)
        if cls.local_path is None:
            cls.local_path = os.path.join(CODE_ROOT, 'data')

        if not os.path.isabs(cls.local_path):
            cls.local_path = os.path.join(CODE_ROOT, cls.local_path)
        if not os.path.isdir(cls.local_path):
            os.makedirs(cls.local_path, exist_ok=True)

        # use this to store temporary files (scratch files)
        cls.temp_path = cfg.value('path.data_temp', None)
        if cls.temp_path is None:
            cls.temp_path = os.path.join(CODE_ROOT, 'data')

        if not os.path.isabs(cls.temp_path):
            cls.temp_path = os.path.join(CODE_ROOT, cls.temp_path)
        if not os.path.isdir(cls.temp_path):
            os.makedirs(cls.temp_path, exist_ok=True)

    @classmethod
    def safe_mkdir(cls, path):
        if path is None or path == '':
            return  # ignore empty paths, we don't need to make them!
        cfg = config.Config.get()

        allowed_dirs = []
        if cls.local_path is not None:
            allowed_dirs.append(cls.local_path)
        temp_path = cfg.value('path.data_temp', None)
        if temp_path is not None:
            allowed_dirs.append(temp_path)

        allowed_dirs = list(set(allowed_dirs))

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

    @declared_attr
    def filepath(cls):
        uniqueness = True
        if cls.__name__ in ['Cutouts']:
            uniqueness = False
        return sa.Column(
            sa.Text,
            nullable=False,
            index=True,
            unique=uniqueness,
            doc="Base path (relative to the data root) for a stored file"
        )


    filepath_extensions = sa.Column(
        sa.ARRAY(sa.Text),
        nullable=True,
        doc="If non-null, array of text appended to filepath to get actual saved filenames."
    )

    md5sum = sa.Column(
        sqlUUID(as_uuid=True),
        nullable=True,
        default=None,
        doc="md5sum of the file, provided by the archive server"
    )

    md5sum_extensions = sa.Column(
        sa.ARRAY(sqlUUID(as_uuid=True)),
        nullable=True,
        default=None,
        doc="md5sum of extension files; must have same number of elements as filepath_extensions"
    )

    # ref: https://docs.sqlalchemy.org/en/20/orm/declarative_mixins.html#creating-indexes-with-mixins
    @declared_attr
    def __table_args__(cls):
        return (
            CheckConstraint(
                sqltext='NOT(md5sum IS NULL AND '
                        '(md5sum_extensions IS NULL OR array_position(md5sum_extensions, NULL) IS NOT NULL))',
                name=f'{cls.__tablename__}_md5sum_check'
            ),
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
                That means it will not try to download it from archive, either.
                This should be used only when creating a new object that will
                later be associated with a file on disk (or for tests).
                This property is NOT SAVED TO DB!
                Saving to DB should only be done when a file exists
                This is True by default, except for subclasses that
                override the _do_not_require_file_to_exist() method.
                # TODO: add the check that file exists before committing?
        """
        if len(args) == 1 and isinstance(args[0], str):
            self.filepath = args[0]

        self.filepath = kwargs.pop('filepath', self.filepath)
        self.nofile = kwargs.pop('nofile', self._do_not_require_file_to_exist())

        self._archive = None

    @orm.reconstructor
    def init_on_load(self):
        self.nofile = self._do_not_require_file_to_exist()
        self._archive = None

    @property
    def archive(self):
        if getattr(self, '_archive', None) is None:
            self._archive = get_archive_object()
        return self._archive

    @archive.setter
    def archive(self, value):
        self._archive = value

    @staticmethod
    def _do_not_require_file_to_exist():
        """
        The default value for the nofile property of new objects.
        Generally it is ok to make new FileOnDiskMixin derived objects
        without first having a file (the file is created by the app and
        saved to disk before the object is committed).
        Some subclasses (e.g., Exposure) will override this method
        so that the default is that a file MUST exist upon creation.
        In either case the caller to the __init__ method can specify
        the value of nofile explicitly.
        """
        return True

    def __setattr__(self, key, value):
        if key == 'filepath' and isinstance(value, str):
            value = self._validate_filepath(value)

        super().__setattr__(key, value)

    def _validate_filepath(self, filepath):
        """
        Make sure the filepath is legitimate.
        If the filepath starts with the local path
        (i.e., an absolute path is given) then
        the local path is removed from the filepath,
        forcing it to be a relative path.

        Parameters
        ----------
        filepath: str
            The filepath to validate.

        Returns
        -------
        filepath: str
            The validated filepath.
        """
        if filepath.startswith(self.local_path):
            filepath = filepath[len(self.local_path) + 1:]

        return filepath

    def get_fullpath(self, download=True, as_list=False, nofile=None, always_verify_md5=False):
        """
        Get the full path of the file, or list of full paths
        of files if filepath_extensions is not None.
        If the archive is defined, and download=True (default),
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
            Must have archive defined. Default is True.
        as_list: bool
            Whether to return a list of filepaths, even if filepath_extensions=None.
            Default is False.
        nofile: bool
            Whether to check if the file exists on local disk.
            Default is None, which means use the value of self.nofile.
        always_verify_md5: bool
            Set True to verify that the file's md5sum matches what's
            in the database (if there is one in the database), and
            raise an exception if it doesn't.  Ignored if nofile=True.

        Returns
        -------
        str or list of str
            Full path to the file(s) on local disk.
        """
        if self.filepath_extensions is None:
            if as_list:
                return [self._get_fullpath_single(download=download, nofile=nofile,
                                                  always_verify_md5=always_verify_md5)]
            else:
                return self._get_fullpath_single(download=download, nofile=nofile,
                                                 always_verify_md5=always_verify_md5)
        else:
            return [
                self._get_fullpath_single(download=download, ext=ext, nofile=nofile,
                                          always_verify_md5=always_verify_md5)
                for ext in self.filepath_extensions
            ]

    def _get_fullpath_single(self, download=True, ext=None, nofile=None, always_verify_md5=False):
        """Get the full path of a single file.
        Will follow the same logic as get_fullpath(),
        of checking and downloading the file from the server
        if it is not on local disk.

        Parameters
        ----------
        download: bool
            Whether to download the file from server if missing.
            Must have archive defined. Default is True.
        ext: str
            Extension to add to the filepath. Default is None
            (indicating that this object is stored in a single file).
        nofile: bool
            Whether to check if the file exists on local disk.
            Default is None, which means use the value of self.nofile.
        always_verify_md5: bool
            Set True to verify that the file's md5sum matches what's
            in the database (if there is one in the database), and
            raise an exception if it doesn't.  Ignored if nofile=True.

        Returns
        -------
        str
            Full path to the file on local disk.

        """
        if self.filepath is None:
            return None

        if nofile is None:
            nofile = self.nofile

        if not nofile and self.local_path is None:
            raise ValueError("Local path not defined!")

        fname = self.filepath
        md5sum = None
        if ext is None:
            md5sum = self.md5sum.hex if self.md5sum is not None else None
        else:
            found = False
            try:
                extdex = self.filepath_extensions.index( ext )
            except ValueError:
                raise ValueError(f"Unknown extension {ext} for {fname}" )
            if (self.md5sum_extensions is None ) or ( extdex >= len(self.md5sum_extensions) ):
                md5sum = None
            else:
                md5sum = self.md5sum_extensions[extdex]
                md5sum = None if md5sum is None else md5sum.hex
            fname += ext

        downloaded = False
        fullname = os.path.join(self.local_path, fname)
        if ( not nofile ) and ( not os.path.exists(fullname) ) and download and ( self.archive is not None ):
            if md5sum is None:
                raise RuntimeError(f"Don't have md5sum in the database for {fname}, can't download")
            self.archive.download( fname, fullname, verifymd5=True, clobbermismatch=False, mkdir=True )
            downloaded = True

        if not nofile:
            if not os.path.exists(fullname):
                raise FileNotFoundError(f"File {fullname} not found!")
            elif always_verify_md5 and not downloaded and md5sum is not None:
                # self.archive.download will have already verified the md5sum
                filemd5 = hashlib.md5()
                with open(fullname, "rb") as ifp:
                    filemd5.update(ifp.read())
                localmd5 = filemd5.hexdigest()
                if localmd5 != md5sum:
                    raise ValueError( f"{fname} has md5sum {localmd5} on disk, which doesn't match the "
                                      f"database value of {md5sum}" )

        return fullname

    def save(self, data, extension=None, overwrite=True, exists_ok=True, verify_md5=True, no_archive=False ):
        """Save a file to disk, and to the archive.

        Parameters
        ---------
        data: bytes, string, or Path
          The data to be saved
        extension: string or None
          The file extension
        overwrite: bool
          True to overwrite existing files (locally and on the archive).
        exists_ok: bool
          Ignored if overwrite is True.  Otherwise: if the file exists
          on disk, and this is False, raise an exception.
        verify_md5: bool
          Used to modify both overwrite and exists_ok
            LOCAL STORAGE
               verify_md5 = True
                  if overwrite = True, check file md5 before actually overwriting
                  if overwrite = False
                      if exists_ok = True, verify existing file
               verify_md5 = False
                  if overwrite = True, always overwrite the file
                  if overwrite = False
                      if exists_ok = True, assume existing file is right
            ARCHIVE
               If self.md5sum (or the appropriate entry in
               md5sum_extensions) is null, then always upload to the
               archive as long as no_archive is False). Otherwise,
               verify_md5 modifies the behavior;
               verify_md5 = True
                 If self.md5sum (or the appropriate entry in
                 md5sum_extensions) matches the md5sum of the passed data,
                 do not upload to the archive.  Otherwise, if overwrite
                 is true, upload to the archive; if overwrite is false,
                 raise an exception.
               verify_md5 = False
                 If overwrite is True, upload to the archive,
                 overwriting what's there.  Otherwise, assume that
                 what's on the archive is right.
        no_archive: bool
          If True, do *not* save to the archive, only to the local filesystem.

        If data is a "bytes" type, then it represents the relevant
        binary data.  It will be written to the right place in the local
        filestore (underneath path.data_root).

        If data is a pathlib.Path or a string, then it is the
        (resolvable) path to a file on disk.  In this case, if the file
        is already in the right place underneath path.data_root, it is
        just left in place (modulo verify_md5).  Otherwise, it is copied
        there.

        Then, in either case, the file is uploaded to the archive (if
        the class property archive is not None, modul verify_md5).  Once
        it's uploaded to the archive, the object's md5sum is set (or
        updated, if overwrite is True and it wasn't null to start with).

        If extension is not None, and it isn't already in the list of
        filepath_extensions, it will be added.

        Performance notes: if you call this with anything other than
        overwrite=False, exists_ok=True, verify_md5=False, you may well
        have redundant I/O.  You may have saved an image before, and
        then (for instance) called
        pipeline.data_store.save_and_commit(), which will at the very
        least read things to verify that md5sums match.

        Of course, not either using overwrite=True or verify_md5=True
        could lead to incorrect files in either the local filestore on
        the server not being detected.

        """

        # (This one is an ugly mess of if statements, done to avoid reading
        # or writing files when not necessary since I/O tends to be much
        # more expensive than processing.)

        # First : figure out if this is an extension or not,
        #   and make sure that's consistent with the object.
        # If it is:
        #   Find the index into the extensions array for
        #   this extension, or append to the array if
        #   it's a new extension that doesn't already exist.
        #   Set the variables curextensions and extmd5s to lists with
        #   extensions and md5sums of extension files,
        #   initially copied from self.filepath_extensions and
        #   self.md5sum_extensions, and modified if necessary
        #   with the saved file.  extensiondex holds the index
        #   into both of these arrays for the current extension.
        # else:
        #   Set curextions, extmd5s, and extensiondex to None.

        # We will either replace these two variables with empty lists,
        #  or make a copy (using list()).  The reason for this: we don't
        #  want to directly modify the lists in self until the saving is
        #  done.  That way, self doesn't get mucked up if this function
        #  exceptions out.
        curextensions = self.filepath_extensions
        extmd5s = self.md5sum_extensions

        extensiondex = None
        if extension is None:
            if curextensions is not None:
                raise RuntimeError( "Tried to save a non-extension file, but this file has extensions" )
            if extmd5s is not None:
                raise RuntimeError( "Data integrity error; filepath_extensions is null, "
                                    "but md5sum_extensions isn't." )
        else:
            if curextensions is None:
                if extmd5s is not None:
                    raise RuntimeError( "Data integrity error; filepath_extensions is null, "
                                        "but md5sum_extensions isn't." )
                curextensions = []
                extmd5s = []
            else:
                if extmd5s is None:
                    raise RuntimeError( "Data integrity error; filepath_extensions is not null, "
                                        "but md5sum_extensions is" )
                curextensions = list(curextensions)
                extmd5s = list(extmd5s)
            if len(curextensions) != len(extmd5s):
                raise RuntimeError( f"Data integrity error; len(md5sum_extensions)={len(extmd5s)}, "
                                    f"but len(filepath_extensions)={len(curextensions)}" )
            try:
                extensiondex = curextensions.index( extension )
            except ValueError:
                curextensions.append( extension )
                extmd5s.append( None )
                extensiondex = len(curextensions) - 1

        # relpath holds the path of the file relative to the data store root
        # origmd5 holds the md5sum (hashlib.hash object) of the original file,
        #   *unless* the original file is already the right file in the local file store
        #   (in which case it's None)
        # localpath holds the absolute path of where the file should be written in the local file store
        relpath = pathlib.Path( self.filepath if extension is None else self.filepath + extension )
        localpath = pathlib.Path( self.local_path ) / relpath
        if isinstance( data, bytes ):
            path = "passed data"
        else:
            if isinstance( data, str ):
                path = pathlib.Path( data )
            elif isinstance( data, pathlib.Path ):
                path = data
            else:
                raise TypeError( f"data must be bytes, str, or Path, not {type(data)}" )
            path = path.absolute()
            data = None

        alreadyinplace = False
        mustwrite = False
        origmd5 = None
        if not localpath.exists():
            mustwrite = True
        else:
            if not localpath.is_file():
                raise RuntimeError( f"{localpath} exists but is not a file!  Can't save." )
            if localpath == path:
                alreadyinplace = True
                _logger.debug( f"FileOnDiskMixin.save: local file store path and original path are the same: {path}" )
            else:
                if ( not overwrite ) and ( not exists_ok ):
                    raise FileExistsError( f"{localpath} already exists, cannot save." )
                if verify_md5:
                    origmd5 = hashlib.md5()
                    if data is None:
                        with open( path, "rb" ) as ifp:
                            data = ifp.read()
                    origmd5.update( data )
                    localmd5 = hashlib.md5()
                    with open( localpath, "rb" ) as ifp:
                        localmd5.update( ifp.read() )
                    if localmd5.hexdigest() != origmd5.hexdigest():
                        if overwrite:
                            _logger.debug( f"Existing {localpath} md5sum mismatch; overwriting." )
                            mustwrite = True
                        else:
                            raise ValueError( f"{localpath} exists, but its md5sum {localmd5.hexdigest()} does not "
                                              f"match md5sum of {path} {origmd5.hexdigest()}" )
                else:
                    if overwrite:
                        _logger.debug( f"Overwriting {localpath}" )
                        mustwrite = True
                    elif exists_ok:
                        _logger.debug( f"{localpath} already exists, not verifying md5 nor overwriting" )
                    else:
                        # raise FileExistsError( f"{localpath} already exists, not saving" )
                        # Logically, should not be able to get here
                        raise RuntimeError( "This should never happen" )

        if mustwrite and not alreadyinplace:
            if data is None:
                with open( path, "rb" ) as ifp:
                    data = ifp.read()
            if origmd5 is None:
                origmd5 = hashlib.md5()
                origmd5.update( data )
            with open( localpath, "wb" ) as ofp:
                ofp.write( data )
            # Verify written file
            with open( localpath, "rb" ) as ifp:
                writtenmd5 = hashlib.md5()
                writtenmd5.update( ifp.read() )
                if writtenmd5.hexdigest() != origmd5.hexdigest():
                    raise RuntimeError( f"Error writing {localpath}; written file md5sum mismatches expected!" )

        # If there is no archive, update the md5sum now
        if self.archive is None:
            if origmd5 is None:
                origmd5 = hashlib.md5()
                with open( localpath, "rb" ) as ifp:
                    origmd5.update( ifp.read() )
            if curextensions is not None:
                extmd5s[ extensiondex ] = UUID( origmd5.hexdigest() )
                self.filepath_extensions = curextensions
                self.md5sum_extensions = extmd5s
            else:
                self.md5sum = UUID( origmd5.hexdigest() )
            return

        # This is the case where there *is* an archive, but the no_archive option was passed
        if no_archive:
            if curextensions is not None:
                self.filepath_extensions = curextensions
                self.md5sum_extensions = extmd5s
            return

        # The rest of this deals with the archive

        archivemd5 = self.md5sum if extension is None else extmd5s[extensiondex]
        logfilepath = self.filepath if extension is None else f'{self.filepath}{extension}'

        mustupload = False
        if archivemd5 is None:
            mustupload = True
        else:
            if not verify_md5:
                if overwrite:
                    _logger.debug( f"Uploading {logfilepath} to archive, overwriting existing file" )
                    mustupload = True
                else:
                    _logger.debug( f"Assuming existing {logfilepath} on archive is correct" )
            else:
                if origmd5 is None:
                    origmd5 = hashlib.md5()
                    if data is None:
                        with open( localpath, "rb" ) as ifp:
                            data = ifp.read()
                    origmd5.update( data )
                if origmd5.hexdigest() == archivemd5.hex:
                    _logger.debug( f"Archive md5sum for {logfilepath} matches saved data, not reuploading." )
                else:
                    if overwrite:
                        _logger.debug( f"Archive md5sum for {logfilepath} doesn't match saved data, "
                                       f"overwriting on archive." )
                        mustupload = True
                    else:
                        raise ValueError( f"Archive md5sum for {logfilepath} does not match saved data!" )

        if mustupload:
            remmd5 = self.archive.upload(
                localpath=localpath,
                remotedir=relpath.parent,
                remotename=relpath.name,
                overwrite=overwrite,
                md5=origmd5
            )
            remmd5 = UUID( remmd5 )
            if curextensions is not None:
                extmd5s[extensiondex] = remmd5
                self.md5sum = None
                self.filepath_extensions = curextensions
                self.md5sum_extensions = extmd5s
            else:
                self.md5sum = remmd5

    def remove_data_from_disk(self, remove_folders=True, remove_downstream_data=False):
        """Delete the data from local disk, if it exists.
        If remove_folders=True, will also remove any folders
        if they are empty after the deletion.
        Use remove_downstream_data=True to also remove any
        downstream data (e.g., for an Image, that would be the
        data for the SourceLists and PSFs that depend on this Image).
        This function will not remove database rows or archive files,
        only cleanup local storage for this object and its downstreams.

        To remove both the files and the database entry, use
        delete_from_disk_and_database() instead.

        Parameters
        ----------
        remove_folders: bool
            If True, will remove any folders on the path to the files
            associated to this object, if they are empty.
        remove_downstream_data: bool
            If True, will also remove any downstream data.
            Will recursively call get_downstreams() and find any objects
            that have remove_data_from_disk() implemented, and call it.
            Default is False.
        """
        if self.filepath is not None:
            # get the filepath, but don't check if the file exists!
            for f in self.get_fullpath(as_list=True, nofile=True):
                if os.path.exists(f):
                    os.remove(f)
                    if remove_folders:
                        folder = f
                        for i in range(10):
                            folder = os.path.dirname(folder)
                            if len(os.listdir(folder)) == 0:
                                os.rmdir(folder)
                            else:
                                break

        if remove_downstream_data:
            try:
                downstreams = self.get_downstreams()
                for d in downstreams:
                    if hasattr(d, 'remove_data_from_disk'):
                        d.remove_data_from_disk(remove_folders=remove_folders, remove_downstream_data=True)

            except NotImplementedError as e:
                pass  # if this object does not implement get_downstreams, it is ok

    def delete_from_archive(self, remove_downstream_data=False):
        """Delete the file from the archive, if it exists.
        This will not remove the file from local disk, nor
        from the database.  Use delete_from_disk_and_database()
        to do that.

        Parameters
        ----------
        remove_downstream_data: bool
            If True, will also remove any downstream data.
            Will recursively call get_downstreams() and find any objects
            that have remove_data_from_disk() implemented, and call it.
            Default is False.
        """
        if remove_downstream_data:
            try:
                downstreams = self.get_downstreams()
                for d in downstreams:
                    if hasattr(d, 'delete_from_archive'):
                        d.delete_from_archive(remove_downstream_data=True)  # TODO: do we need remove_folders?

            except NotImplementedError as e:
                pass  # if this object does not implement get_downstreams, it is ok

        if self.filepath is not None:
            if self.filepath_extensions is None:
                self.archive.delete( self.filepath, okifmissing=True )
            else:
                for ext in self.filepath_extensions:
                    self.archive.delete( f"{self.filepath}{ext}", okifmissing=True )
        # make sure these are set to null just in case we fail
        # to commit later on, we will at least know something is wrong
        self.md5sum = None
        self.md5sum_extensions = None

    def delete_from_disk_and_database(
            self, session=None, commit=True, remove_folders=True, remove_downstream_data=False, archive=True,
    ):
        """
        Delete the data from disk, archive and the database.
        Use this to clean up an entry from all locations.
        Will delete the object from the DB using the given session
        (or using an internal session).
        If using an internal session, commit must be True,
        to allow the change to be committed before closing it.

        This will silently continue if the file does not exist
        (locally or on the archive), or if it isn't on the database,
        and will attempt to delete from any locations regardless
        of if it existed elsewhere or not.

        TODO : this is sometimes broken if you don't pass a session.

        Parameters
        ----------
        session: sqlalchemy session
            The session to use for the deletion. If None, will open a new session,
            which will also close at the end of the call.
        commit: bool
            Whether to commit the deletion to the database.
            Default is True. When session=None then commit must be True,
            otherwise the session will exit without committing
            (in this case the function will raise a RuntimeException).
        remove_folders: bool
            If True, will remove any folders on the path to the files
            associated to this object, if they are empty.
        remove_downstream_data: bool
            If True, will also remove any downstream data.
            Will recursively call get_downstreams() and find any objects
            that have remove_data_from_disk() implemented, and call it.
            Default is False.
        archive: bool
            If True, will also delete the file from the archive.
            Default is True.
        """
        if session is None and not commit:
            raise RuntimeError("When session=None, commit must be True!")

        self.remove_data_from_disk(remove_folders=remove_folders, remove_downstream_data=remove_downstream_data)

        if archive:
            self.delete_from_archive(remove_downstream_data=remove_downstream_data)

        # make sure these are set to null just in case we fail
        # to commit later on, we will at least know something is wrong
        self.md5sum = None
        self.md5sum_extensions = None
        self.filepath_extensions = None
        self.filepath = None

        SeeChangeBase.delete_from_database(self, session=session, commit=commit)


# load the default paths from the config
FileOnDiskMixin.configure_paths()


def safe_mkdir(path):
    FileOnDiskMixin.safe_mkdir(path)


class AutoIDMixin:
    id = sa.Column(
        sa.BigInteger,
        primary_key=True,
        index=True,
        autoincrement=True,
        doc="Autoincrementing unique identifier for this dataset",
    )


class SpatiallyIndexed:
    """A mixin for tables that have ra and dec fields indexed via q3c."""

    ra = sa.Column(sa.Double, nullable=False, doc='Right ascension in degrees')

    dec = sa.Column(sa.Double, nullable=False, doc='Declination in degrees')

    gallat = sa.Column(sa.Double, index=True, doc="Galactic latitude of the target. ")

    gallon = sa.Column(sa.Double, index=False, doc="Galactic longitude of the target. ")

    ecllat = sa.Column(sa.Double, index=True, doc="Ecliptic latitude of the target. ")

    ecllon = sa.Column(sa.Double, index=False, doc="Ecliptic longitude of the target. ")

    @declared_attr
    def __table_args__(cls):
        tn = cls.__tablename__
        return (
            sa.Index(f"{tn}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
        )

    def calculate_coordinates(self):
        """Fill self.gallat, self.gallon, self.ecllat, and self.ecllong based on self.ra and self.dec."""

        if self.ra is None or self.dec is None:
            return

        coords = SkyCoord(self.ra, self.dec, unit="deg", frame="icrs")
        self.gallat = coords.galactic.b.deg
        self.gallon = coords.galactic.l.deg
        self.ecllat = coords.barycentrictrueecliptic.lat.deg
        self.ecllon = coords.barycentrictrueecliptic.lon.deg

    @hybrid_method
    def within( self, fourcorn ):
        """An SQLAlchemy filter to find all things within a FourCorners object

        Parameters
        ----------
          fourcorn: FourCorners
            A FourCorners object

        Returns
        -------
          An expression usable in a sqlalchemy filter

        """

        return func.q3c_poly_query( self.ra, self.dec,
                                    sqlarray( [ fourcorn.ra_corner_00, fourcorn.dec_corner_00,
                                                fourcorn.ra_corner_01, fourcorn.dec_corner_01,
                                                fourcorn.ra_corner_11, fourcorn.dec_corner_11,
                                                fourcorn.ra_corner_10, fourcorn.dec_corner_10 ] ) )


class FourCorners:
    """A mixin for tables that have four RA/Dec corners"""

    ra_corner_00 = sa.Column( sa.REAL, nullable=False, index=True, doc="RA of the low-RA, low-Dec corner (degrees)" )
    ra_corner_01 = sa.Column( sa.REAL, nullable=False, index=True, doc="RA of the low-RA, high-Dec corner (degrees)" )
    ra_corner_10 = sa.Column( sa.REAL, nullable=False, index=True, doc="RA of the high-RA, low-Dec corner (degrees)" )
    ra_corner_11 = sa.Column( sa.REAL, nullable=False, index=True, doc="RA of the high-RA, high-Dec corner (degrees)" )
    dec_corner_00 = sa.Column( sa.REAL, nullable=False, index=True, doc="Dec of the low-RA, low-Dec corner (degrees)" )
    dec_corner_01 = sa.Column( sa.REAL, nullable=False, index=True,
                               doc="Dec of the low-RA, high-Dec corner (degrees)" )
    dec_corner_10 = sa.Column( sa.REAL, nullable=False, index=True,
                               doc="Dec of the high-RA, low-Dec corner (degrees)" )
    dec_corner_11 = sa.Column( sa.REAL, nullable=False, index=True,
                               doc="Dec of the high-RA, high-Dec corner (degrees)" )

    @classmethod
    def sort_radec( cls, ras, decs ):
        """Sort ra and dec lists so they're each in the order in models.base.FourCorners

        Parameters
        ----------
          ras: list of float
             Four ra values in a list. 
          decs: list of float
             Four dec values in a list. 

        Returns
        -------
          Two new lists (ra, dec) sorted so they're in the order:
          (lowRA,lowDec), (lowRA,highDec), (highRA,lowDec), (highRA,highDec)

        """

        if len(ras) != 4:
            raise ValueError(f'ras must be a list/array with exactly four elements. Got {ras}')
            raise ValueError(f'decs must be a list/array with exactly four elements. Got {decs}')

        raorder = list( range(4) )
        raorder.sort( key=lambda i: ras[i] )

        # Of two lowest ras, of those, pick the one with the lower dec;
        #   that's lowRA,lowDec; the other one is lowRA, highDec

        dex00 = raorder[0] if decs[raorder[0]] < decs[raorder[1]] else raorder[1]
        dex01 = raorder[1] if decs[raorder[0]] < decs[raorder[1]] else raorder[0]

        # Same thing, only now high ra

        dex10 = raorder[2] if decs[raorder[2]] < decs[raorder[3]] else raorder[3]
        dex11 = raorder[3] if decs[raorder[2]] < decs[raorder[3]] else raorder[2]

        return ( [  ras[dex00],  ras[dex01],  ras[dex10],  ras[dex11] ],
                 [ decs[dex00], decs[dex01], decs[dex10], decs[dex11] ] )

    @hybrid_method
    def containing( self, ra, dec ):
        """An SQLAlchemy filter for objects that might contain a given ra/dec.

        This will be reliable for objects (i.e. images, or whatever else
        has four corners) that are square to the sky (assuming that the
        ra* and dec* fields are correct).  However, if the object is at
        an angle, it will return objects that have the given ra, dec in
        the rectangle on the sky oriented along ra/dec lines that fully
        contains the four corners of the image.

        Parameters
        ----------
           ra, dec: float
              Position to search (decimal degrees).

        Returns
        -------
           An expression usable in a sqlalchemy filter

        """

        # This query will go through every row of the table it's
        # searching, because q3c uses the index on the first two
        # arguments, not on the array argument.

        # It could probably be made faster by making a first pass doing:
        #   greatest( ra** ) >= ra AND least( ra** ) <= ra AND
        #   greatest( dec** ) >= dec AND least( dec** ) <= dec
        # with indexes in ra** and dec**.  Put the results of that into
        # a temp table, and then do the polygon search on that temp table.
        #
        # I have no clue how to implement that simply here as as an
        # SQLAlchemy filter, so I implement it in find_containing()

        return func.q3c_poly_query( ra, dec, sqlarray( [ self.ra_corner_00, self.dec_corner_00,
                                                         self.ra_corner_01, self.dec_corner_01,
                                                         self.ra_corner_11, self.dec_corner_11,
                                                         self.ra_corner_10, self.dec_corner_10 ] ) )

    @classmethod
    def find_containing( cls, siobj, session=None ):
        """Return all images (or whatever) that contain the given SpatiallyIndexed thing

        Parameters
        ----------
          siobj: SpatiallyIndexed
            A single object that is spatially indexed

        Returns
        -------
           An sql query result thingy.

        """

        # Overabundance of caution to avoid SQL injection
        ra = float( siobj.ra )
        dec = float( siobj.dec )

        with SmartSession( session ) as sess:
            sess.execute( sa.text( f"SELECT i.id, i.ra_corner_00, i.ra_corner_01, i.ra_corner_10, i.ra_corner_11, "
                                   f"       i.dec_corner_00, i.dec_corner_01, i.dec_corner_10, i.dec_corner_11 "
                                   f"INTO TEMP TABLE temp_find_containing "
                                   f"FROM {cls.__tablename__} i "
                                   f"WHERE GREATEST(i.ra_corner_00, i.ra_corner_01, "
                                   f"               i.ra_corner_10, i.ra_corner_11 ) >= :ra "
                                   f"  AND LEAST(i.ra_corner_00, i.ra_corner_01, "
                                   f"            i.ra_corner_10, i.ra_corner_11 ) <= :ra "
                                   f"  AND GREATEST(i.dec_corner_00, i.dec_corner_01, "
                                   f"               i.dec_corner_10, i.dec_corner_11 ) >= :dec "
                                   f"  AND LEAST(i.dec_corner_00, i.dec_corner_01, "
                                   f"            i.dec_corner_10, i.dec_corner_11 ) <= :dec" ),
                          { 'ra': ra, 'dec': dec } )
            query = sa.text( f"SELECT i.id FROM temp_find_containing i "
                             f"WHERE q3c_poly_query( {ra}, {dec}, ARRAY[ i.ra_corner_00, i.dec_corner_00, "
                             f"                                          i.ra_corner_01, i.dec_corner_01, "
                             f"                                          i.ra_corner_11, i.dec_corner_11, "
                             f"                                          i.ra_corner_10, i.dec_corner_10 ])" )
            objs = sess.scalars( sa.select( cls ).from_statement( query ) ).all()
            sess.execute( sa.text( "DROP TABLE temp_find_containing" ) )
            return objs

    @hybrid_method
    def cone_search( self, ra, dec, rad, radunit='arcsec' ):
        """Find all objects of this class that are within a cone.

        Parameters
        ----------
          ra: float
            The central right ascension in decimal degrees
          dec: float
            The central declination in decimal degrees
          rad: float
            The radius of the circle on the sky
          radunit: str
            The units of rad.  One of 'arcsec', 'arcmin', 'degrees', or
            'radians'.  Defaults to 'arcsec'.

        Returns
        -------
          A query with the cone search.

        """

        if radunit == 'arcmin':
            rad /= 60.
        elif radunit == 'arcsec':
            rad /= 3600.
        elif radunit == 'radians':
            rad *= 180. / math.pi
        elif radunit != 'degrees':
            raise ValueError( f'SpatiallyIndexed.cone_search: unknown radius unit {radunit}' )

        return func.q3c_radial_query( self.ra, self.dec, ra, dec, rad )


class HasBitFlagBadness:
    """A mixin class that adds a bitflag marking why this object is bad. """
    _bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        default=0,
        index=True,
        doc='Bitflag for this object. Good objects have a bitflag of 0. '
            'Bad objects are each bad in their own way (i.e., have different bits set). '
            'The bitflag will include this value, bit-wise-or-ed with the bitflags of the '
            'upstream object that were used to make this one. '
    )

    @declared_attr
    def _upstream_bitflag(cls):
        if cls.__name__ != 'Exposure':
            return sa.Column(
                sa.BIGINT,
                nullable=False,
                default=0,
                index=True,
                doc='Bitflag of objects used to generate this object. '
            )
        else:
            return None

    @hybrid_property
    def bitflag(self):
        if self._bitflag is None:
            self._bitflag = 0
        if self._upstream_bitflag is None:
            self._upstream_bitflag = 0
        return self._bitflag | self._upstream_bitflag

    @bitflag.inplace.expression
    @classmethod
    def bitflag(cls):
        return cls._bitflag.op('|')(cls._upstream_bitflag)

    @bitflag.inplace.setter
    def bitflag(self, value):
        allowed_bits = 0
        for i in self._get_inverse_badness().values():
            allowed_bits += 2 ** i
        if value & ~allowed_bits != 0:
            raise ValueError(f'Bitflag value {bin(value)} has bits set that are not allowed.')
        self._bitflag = value

    @property
    def badness(self):
        """
        A comma separated string of keywords describing
        why this data is not good, based on the bitflag.
        This includes all the reasons this data is bad,
        including the parent data models that were used
        to create this data (e.g., the Exposure underlying
        the Image).
        """
        return bitflag_to_string(self.bitflag, data_badness_dict)

    @badness.setter
    def badness(self, value):
        """Set the badness for this image using a comma separated string. """
        self.bitflag = string_to_bitflag(value, self._get_inverse_badness())

    def append_badness(self, value):
        """Add some keywords (in a comma separated string)
        describing what is bad about this image.
        The keywords will be added to the list "badness"
        and the bitflag for this image will be updated accordingly.
        """
        self.bitflag |= string_to_bitflag(value, self._get_inverse_badness())

    description = sa.Column(
        sa.Text,
        nullable=True,
        doc='Free text comment about this data product, e.g., why it is bad. '
    )

    def update_downstream_badness(self, session=None, commit=True):
        """Send a recursive command to update all downstream objects that have bitflags.

        Since this function is called recursively, it always updates the current
        object's _upstream_bitflag to reflect the state of this object's upstreams,
        before calling the same function on all downstream objects.

        Note that this function will session.add() this object and all its
        recursive downstreams (to update the changes in bitflag) and will
        commit the new changes on its own (unless given commit=False)
        but only at the end of the recursion.

        If session=None and commit=False an exception is raised.

        Parameters
        ----------
        session: sqlalchemy session
            The session to use for the update. If None, will open a new session,
            which will also close at the end of the call. In that case, must
            provide a commit=True to commit the changes.
        commit: bool (default True)
            Whether to commit the changes to the database.
        """
        # make sure this object is current:
        with SmartSession(session) as session:
            new_bitflag = 0  # start from scratch, in case some upstreams have lost badness
            for upstream in self.get_upstreams(session):
                if hasattr(upstream, '_bitflag'):
                    new_bitflag |= upstream.bitflag

            if hasattr(self, '_upstream_bitflag'):
                self._upstream_bitflag = new_bitflag
                session.add(self)

            # recursively do this for all the other objects
            for downstream in self.get_downstreams(session):
                if hasattr(downstream, 'update_downstream_badness') and callable(downstream.update_downstream_badness):
                    downstream.update_downstream_badness(session=session, commit=False)

            if commit:
                session.commit()

    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object

        For the base class this is the most inclusive inverse (allows all badness).
        """
        return data_badness_inverse


if __name__ == "__main__":
    pass
