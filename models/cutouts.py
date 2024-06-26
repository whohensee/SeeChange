import os
import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.ext.associationproxy import association_proxy

import h5py

from astropy.table import Table

from models.base import (
    SmartSession,
    Base,
    SeeChangeBase,
    AutoIDMixin,
    FileOnDiskMixin,
    SpatiallyIndexed,
    HasBitFlagBadness,
)
from models.enums_and_bitflags import CutoutsFormatConverter, cutouts_badness_inverse
from models.source_list import SourceList


class Cutouts(Base, AutoIDMixin, FileOnDiskMixin, HasBitFlagBadness):

    __tablename__ = 'cutouts'

    # a unique constraint on the provenance and the source list, but also on the index in the list
    __table_args__ = (
        UniqueConstraint(
            'sources_id', 'provenance_id', name='_cutouts_sources_provenance_uc'
        ),
    )

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=CutoutsFormatConverter.convert('hdf5'),
        doc="Format of the file on disk. Should be fits, hdf5, csv or npy. "
            "Saved as integer but is converted to string when loaded. "
    )

    @hybrid_property
    def format(self):
        return CutoutsFormatConverter.convert(self._format)

    @format.expression
    def format(cls):
        # ref: https://stackoverflow.com/a/25272425
        return sa.case(CutoutsFormatConverter.dict, value=cls._format)

    @format.setter
    def format(self, value):
        self._format = CutoutsFormatConverter.convert(value)

    sources_id = sa.Column(
        sa.ForeignKey('source_lists.id', name='cutouts_source_list_id_fkey', ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of the source list (of detections in the difference image) this cutouts object is associated with. "
    )

    sources = orm.relationship(
        SourceList,
        cascade='save-update, merge, refresh-expire, expunge',
        passive_deletes=True,
        lazy='selectin',
        doc="The source list (of detections in the difference image) this cutouts object is associated with. "
    )

    sub_image_id = association_proxy('sources', 'image_id')
    sub_image = association_proxy('sources', 'image')

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='cutouts_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this cutout. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this cutout. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this cutout. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this cutout. "
        )
    )

    @property
    def new_image(self):
        """Get the aligned new image using the sub_image. """
        return self.sub_image.new_aligned_image

    @property
    def ref_image(self):
        """Get the aligned reference image using the sub_image. """
        return self.sub_image.ref_aligned_image

    def __init__(self, *args, **kwargs):
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        HasBitFlagBadness.__init__(self)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self.format = 'hdf5'  # the default should match the column-defined default above!

        self._source_row = None

        self._sub_data = None
        self._sub_weight = None
        self._sub_flags = None
        self._sub_psfflux = None
        self._sub_psffluxerr = None

        self._ref_data = None
        self._ref_weight = None
        self._ref_flags = None

        self._new_data = None
        self._new_weight = None
        self._new_flags = None

        self._co_dict = {}

        self._bitflag = 0

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        FileOnDiskMixin.init_on_load(self)

        self._source_row = None

        self._sub_data = None
        self._sub_weight = None
        self._sub_flags = None
        self._sub_psfflux = None
        self._sub_psffluxerr = None

        self._ref_data = None
        self._ref_weight = None
        self._ref_flags = None

        self._new_data = None
        self._new_weight = None
        self._new_flags = None

        self._co_dict = {}

    def __repr__(self):
        return (
            f"<Cutouts {self.id} "
            f"from SourceList {self.sources_id} "
            f"from Image {self.sub_image_id} "
        )

    def __setattr__(self, key, value):
        if key in ['x', 'y'] and value is not None:
            value = int(round(value))

        super().__setattr__(key, value)

    @staticmethod
    def get_data_dict_attributes(include_optional=True):  # WHPR could rename get_data_attributes
        names = []
        for im in ['sub', 'ref', 'new']:
            for att in ['data', 'weight', 'flags']:
                names.append(f'{im}_{att}')

        if include_optional:
            names += ['sub_psfflux', 'sub_psffluxerr']

        return names

    @property
    def co_dict( self, ):
        # Ok because of partial lazy loading with hdf5 and measurements only wanting their row,
        # this one is complicated. I have set that if you use this attribute, it will ENSURE
        # that you are given the entire dictionary, including checking it is the proper length
        # using the sourcelist
        if self.sources.num_sources is None:
            raise ValueError("The detections of this cutouts has no num_sources attr")
        proper_length = self.sources.num_sources
        if len(self._co_dict) != proper_length and self.filepath is not None:
            self.load()
        return self._co_dict

    @co_dict.setter
    def co_dict( self, value ):
        self._co_dict = value

    @property
    def co_dict_noload(self):
        return self._co_dict

    @staticmethod
    def from_detections(detections, provenance=None, **kwargs):
        """Create a Cutout object from a row in the SourceList.

        The SourceList must have a valid image attribute, and that image should have exactly two
        upstream_images: the reference and new image. Each Cutout will have three small stamps
        from the new, reference, and subtraction images.

        Parameters
        ----------
        detections: SourceList
            The source list from which to create the cutout.
        provenance: Provenance, optional
            The provenance of the cutout. If not given, will leave as None (to be filled externally).
        kwargs: dict
            Can include any of the following keys, in the format: {im}_{att}, where
            the {im} can be "sub", "ref", or "new", and the {att} can be "data", "weight", or "flags".
            These are optional, to be used to fill the different data attributes of this object.

        Returns
        -------
        cutout: Cutout
            The cutout object.
        """
        cutout = Cutouts()
        cutout.sources = detections
        cutout.provenance = provenance
        cutout.provenance_id = provenance.id # I couldn't find this being set anywhere else?

        # update the bitflag
        cutout._upstream_bitflag = detections.bitflag

        return cutout

    def invent_filepath(self):
        if self.sources is None:
            raise RuntimeError( f"Can't invent a filepath for cutouts without a source list" )
        if self.provenance is None:
            raise RuntimeError( f"Can't invent a filepath for cutouts without a provenance" )

        # base the filename on the image filename, not on the sources filename.
        filename = self.sub_image.filepath
        if filename is None:
            filename = self.sub_image.invent_filepath()

        if filename.endswith(('.fits', '.h5', '.hdf5')):
            filename = os.path.splitext(filename)[0]

        filename += '.cutouts_'
        filename += self.provenance.id[:6]
        if self.format == 'hdf5':
            filename += '.h5'
        elif self.format == ['fits', 'jpg', 'png']:
            filename += f'.{self.format}'
        else:
            raise TypeError( f"Unable to create a filepath for cutouts file of type {self.format}" )

        return filename

    def _save_dataset_dict_to_hdf5(self, co_dict, file, groupname):
        """Save the one co_subdict from the co_dict of this Cutouts
        into an HDF5 group for an open file.

        Parameters
        ----------
        co_dict: dict
            The subdict containing the data for a single cutout
        file: h5py.File
            The open HDF5 file to save to.
        groupname: str
            The name of the group to save into. This should be "source_<number>"
        """
        if groupname in file:
            del file[groupname]

        for key in self.get_data_dict_attributes():
            data = co_dict.get(key)

            if data is not None:
                file.create_dataset(
                    f'{groupname}/{key}',
                    data=data,
                    shape=data.shape,
                    dtype=data.dtype,
                    compression='gzip'
                )

    def save(self, filename=None, overwrite=True, **kwargs):
        """Save the data of this Cutouts object into a file.

        Parameters
        ----------
        filename: str, optional
            The (relative/full path) filename to save to. If not given, will use the default filename.
        kwargs: dict
            Any additional keyword arguments to pass to the FileOnDiskMixin.save method.
        """
        # WHPR ADD A CHECK TO MAKE SURE WE ARE SAVING AN ENTRY FOR EACH ROW
        if self._co_dict == {}:
            return None  # do nothing

        for key, value in self._co_dict.items():
            if not isinstance(value, dict):
                raise TypeError("Each entry of _co_dict must be a dictionary")

        # WHPR consider using the has_data check below, maybe not needed with new structure
        # If we did, I would just check that all 9 arrays have data
        if filename is None:
            filename = self.invent_filepath()

        self.filepath = filename

        fullname = os.path.join(self.local_path, filename)
        self.safe_mkdir(os.path.dirname(fullname))

        if not overwrite and os.path.isfile(fullname):
            raise FileExistsError(f"The file {fullname} already exists and overwrite is False.")

        # WHPR think if I really need to mess with all this format stuff anymore
        if self.format == 'hdf5':
            with h5py.File(fullname, 'a') as file:
                for key, value in self._co_dict.items():
                    self._save_dataset_dict_to_hdf5(value, file, key)
        elif self.format == 'fits':
            raise NotImplementedError('Saving cutouts to fits is not yet implemented.')
        elif self.format in ['jpg', 'png']:
            raise NotImplementedError('Saving cutouts to jpg or png is not yet implemented.')
        else:
            raise TypeError(f"Unable to save cutouts file of type {self.format}")

        # make sure to also save using the FileOnDiskMixin method
        FileOnDiskMixin.save(self, fullname, overwrite=overwrite, **kwargs)

    # this might be able to become a class method
    def _load_dataset_dict_from_hdf5(self, file, groupname):
        """Load the dataset from an HDF5 group into one co_subdict and return.

        Parameters
        ----------
        file: h5py.File
            The open HDF5 file to load from.
        groupname: str
            The name of the group to load from. This should be "source_<number>"
        """

        co_dict = {}
        found_data = False
        for att in self.get_data_dict_attributes(): # remove source index for dict soon
            if att in file[groupname]:
                found_data = True
                co_dict[att] = np.array(file[f'{groupname}/{att}'])
        if found_data:
            return co_dict

        # self.format = 'hdf5' # move this line to load (looks unnecessary as was
        # for making individual new cutouts objects)

    def load_one_co_dict(self, groupname, filepath=None):
        """Load data subdict for a single cutout into this Cutouts co_dict. This allows
        a measurement to request only the information relevant to that object, rather
        than populating the entire dictionary when we only need one subdict.
        """

        if filepath is None:
            filepath = self.get_fullpath()

        with h5py.File(filepath, 'r') as file:
            self._co_dict[groupname] = self._load_dataset_dict_from_hdf5(file, groupname)
        return None

    def load(self, filepath=None):
        """Load the data for this cutout from a file.

        Parameters
        ----------
        filepath: str, optional
            The (relative/full path) filename to load from.
            If not given, will use self.get_fullpath() to get the filename.
        """

        if filepath is None:
            filepath = self.get_fullpath()

        if filepath is None:
            raise ValueError("Could not find filepath to load")

        self._co_dict = {}

        if os.path.exists(filepath): # WHPR revisit this check
            if self.format == 'hdf5':  # consider getting rid of this totally
                with h5py.File(filepath, 'r') as file:
                    for groupname in file:
                        self._co_dict[groupname] = self._load_dataset_dict_from_hdf5(file, groupname)

    def remove_data_from_disk(self, remove_folders=True, remove_local=True,
                              database=True, session=None, commit=True, remove_downstreams=False):
        """WHPR review this docstring and check it works how we want
        Delete the data from local disk, if it exists.
        Will remove the dataset for this specific cutout from the file,
        and remove the file if this is the last cutout in the file.
        If remove_folders=True, will also remove any folders
        if they are empty after the deletion.
        This function will not remove database rows or archive files,
        only cleanup local storage for this object and its downstreams.

        To remove both the files and the database entry, use
        delete_from_disk_and_database() instead.

        Parameters
        ----------
        remove_folders: bool
            If True, will remove any folders on the path to the files
            associated to this object, if they are empty.
        remove_downstreams: bool
            This is not used, but kept here for backward compatibility with the base class.
        """
        if database and session is None and not commit:
            raise ValueError('If session is not given, commit must be True.')

        if remove_local:
            fullpath = self.get_fullpath()
            if self.filepath is not None and os.path.isfile(fullpath):
                os.remove(fullpath)

        if database:
            with SmartSession(session) as session:
                self.delete_from_database(session=session, commit=False)
                if commit:
                    session.commit()

        # I mostly copied from delete_list, which was previously in use. It did not include
        # deletion of downstreams. Should I include that?

    def delete_from_archive(self, remove_downstreams=False):
        """Delete the file from the archive, if it exists.
        Will only
        This will not remove the file from local disk, nor
        from the database.  Use delete_from_disk_and_database()
        to do that.

        Parameters
        ----------
        remove_downstreams: bool
            If True, will also remove any downstream data.
            Will recursively call get_downstreams() and find any objects
            that have remove_data_from_disk() implemented, and call it.
            Default is False.
        """
        if self.filepath is not None:
            self.archive.delete(self.filepath, okifmissing=True)

        # QUESTION is this below still worth doing?
        # # make sure these are set to null just in case we fail
        # # to commit later on, we will at least know something is wrong
        # self.md5sum = None
        # self.md5sum_extensions = None

    def get_upstreams(self, session=None):
        """Get the detections SourceList that was used to make this cutout. """
        with SmartSession(session) as session:
            return session.scalars(sa.select(SourceList).where(SourceList.id == self.sources_id)).all()

    def get_downstreams(self, session=None, siblings=False):
        """Get the downstream Measurements that were made from this Cutouts object. """
        from models.measurements import Measurements

        with SmartSession(session) as session:
            return session.scalars(sa.select(Measurements).where(Measurements.cutouts_id == self.id)).all()

    @classmethod
    def merge_list(cls, cutouts_list, session):
        """Merge (or add) the list of Cutouts to the given session. """
        if cutouts_list is None or len(cutouts_list) == 0:
            return cutouts_list

        sources = session.merge(cutouts_list[0].sources)
        for i, cutout in enumerate(cutouts_list):
            cutouts_list[i].sources = sources
            cutouts_list[i] = session.merge(cutouts_list[i])

        return cutouts_list

    @classmethod
    def delete_list(cls, cutouts_list, remove_local=True, archive=True, database=True, session=None, commit=True):
        """
        Remove a list of Cutouts objects from local disk and/or the archive and/or the database.
        This removes the file that includes all the cutouts.
        Can only delete cutouts that share the same filepath.
        WARNING: this will not check that the file contains ONLY the cutouts on the list!
        So, if the list contains a subset of the cutouts on file, the file is still deleted.

        Parameters
        ----------
        cutouts_list: list of Cutouts
            The list of Cutouts objects to remove.
        remove_local: bool
            If True, will remove the file from local disk.
        archive: bool
            If True, will remove the file from the archive.
        database: bool
            If True, will remove the cutouts from the database.
        session: Session, optional
            The database session to use. If not given, will create a new session.
        commit: bool
            If True, will commit the changes to the database.
            If False, will not commit the changes to the database.
            If session is not given, commit must be True.
        """
        if database and session is None and not commit:
            raise ValueError('If session is not given, commit must be True.')

        filepath = set([c.filepath for c in cutouts_list])
        if len(filepath) > 1:
            raise ValueError(
                f'All cutouts must share the same filepath to be deleted together. Got: {filepath}'
            )

        if remove_local:
            fullpath = cutouts_list[0].get_fullpath()
            if fullpath is not None and os.path.isfile(fullpath):
                os.remove(fullpath)

        if archive:
            if cutouts_list[0].filepath is not None:
                cutouts_list[0].archive.delete(cutouts_list[0].filepath, okifmissing=True)

        if database:
            with SmartSession(session) as session:
                for cutout in cutouts_list:
                    cutout.delete_from_database(session=session, commit=False)
                if commit:
                    session.commit()

    def _get_inverse_badness(self):
        return cutouts_badness_inverse
