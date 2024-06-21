import os
import uuid
import h5py

import numpy as np
import pytest

import sqlalchemy as sa

from models.base import SmartSession
from models.cutouts import Cutouts

def test_make_save_load_cutouts(decam_detection_list, cutter):
    try:
        # blabla
        cutter.pars.test_parameter = uuid.uuid4().hex
        ds = cutter.run(decam_detection_list)

        assert cutter.has_recalculated
        assert isinstance(ds.cutouts, Cutouts)
        assert len(ds.cutouts.co_dict) == ds.cutouts.sources.num_sources

        co_subdict = ds.cutouts.co_dict['source_index_0']

        assert ds.cutouts.sub_image == decam_detection_list.image
        assert ds.cutouts.sub_image == decam_detection_list.image
        assert ds.cutouts.sub_image == decam_detection_list.image

        assert isinstance(co_subdict["sub_data"], np.ndarray)
        assert isinstance(co_subdict["sub_weight"], np.ndarray)
        assert isinstance(co_subdict["sub_flags"], np.ndarray)
        assert isinstance(co_subdict["ref_data"], np.ndarray)
        assert isinstance(co_subdict["ref_weight"], np.ndarray)
        assert isinstance(co_subdict["ref_flags"], np.ndarray)
        assert isinstance(co_subdict["new_data"], np.ndarray)
        assert isinstance(co_subdict["new_weight"], np.ndarray)
        assert isinstance(co_subdict["new_flags"], np.ndarray)
        assert ds.cutouts.bitflag is not None

        # set the bitflag just to see if it is loaded or not
        ds.cutouts.bitflag = 2 ** 41  # should be Cosmic Ray

        # save the Cutouts
        ds.cutouts.save()

        # open the file manually and compare
        with h5py.File(ds.cutouts.get_fullpath(), 'r') as file:
            for im in ['sub', 'ref', 'new']:
                for att in ['data', 'weight', 'flags']:
                    assert f'{im}_{att}' in file['source_index_0']
                    assert np.array_equal(co_subdict.get(f'{im}_{att}'),
                                          file['source_index_0'][f'{im}_{att}'])
                    

        # load a cutouts from file and compare
        c2 = Cutouts()
        c2.filepath = ds.cutouts.filepath
        c2.sources = ds.cutouts.sources  # necessary for co_dict
        c2.load() # explicitly load co_dict

        co_subdict2 = c2.co_dict['source_index_0']

        for im in ['sub', 'ref', 'new']:
            for att in ['data', 'weight', 'flags']:
                assert np.array_equal(co_subdict.get(f'{im}_{att}'),
                                        co_subdict2.get(f'{im}_{att}'))
                
        assert c2.bitflag == 0 # should not load all columns from file

        # change the value of one of the arrays
        ds.cutouts.co_dict['source_index_0']['sub_data'][0, 0] = 100
        co_subdict2['sub_data'][0, 0] = 100 # for comparison later

        # make sure we can re-save
        ds.cutouts.save()

        with h5py.File(ds.cutouts.get_fullpath(), 'r') as file:
            assert np.array_equal(ds.cutouts.co_dict['source_index_0']['sub_data'],
                                file['source_index_0']['sub_data'])
            assert file['source_index_0']['sub_data'][0, 0] == 100 # change has propagated
        
        # check that we can add the cutouts to the database
        with SmartSession() as session:
            ds.cutouts = session.merge(ds.cutouts)
            session.commit() # QUESTION: does this necessitate cleanup in the finally block?

        assert ds.cutouts is not None
        assert len(ds.cutouts.co_dict) > 0

        with SmartSession() as session:
            loaded_cutouts = session.scalars(
                sa.select(Cutouts).where(Cutouts.provenance_id == ds.cutouts.provenance.id)
            ).all()
            assert len(loaded_cutouts) == 1
            loaded_cutouts = loaded_cutouts[0]

            # make sure data is correct
            co_subdict = loaded_cutouts.co_dict['source_index_0']
            for im in ['sub', 'ref', 'new']:
                for att in ['data', 'weight', 'flags']:
                    assert np.array_equal(co_subdict.get(f'{im}_{att}'),
                                            co_subdict2.get(f'{im}_{att}'))


    finally:
        if 'ds' in locals() and ds.cutouts is not None:
            ds.cutouts.remove_data_from_disk()
            ds.cutouts.delete_from_archive()
