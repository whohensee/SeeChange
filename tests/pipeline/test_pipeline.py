import os
import pytest
import shutil
import sqlalchemy as sa

from models.base import SmartSession, FileOnDiskMixin
from models.provenance import Provenance
from models.image import Image, image_upstreams_association_table
from models.source_list import SourceList
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

from pipeline.top_level import Pipeline


def check_datastore_and_database_have_everything(exp_id, sec_id, ref_id, session, ds):
    """
    Check that all the required objects are saved on the database
    and in the datastore, after running the entire pipeline.

    Parameters
    ----------
    exp_id: int
        The exposure ID.
    sec_id: str or int
        The section ID.
    ref_id: int
        The reference image ID.
    session: sqlalchemy.orm.session.Session
        The database session
    ds: datastore.DataStore
        The datastore object
    """
    im = session.scalars(
        sa.select(Image).where(Image.exposure_id == exp_id, Image.section_id == str(sec_id))
    ).first()
    assert im is not None
    assert ds.image.id == im.id

    sl = session.scalars(
        sa.select(SourceList).where(SourceList.image_id == im.id, SourceList.is_sub.is_(False))
    ).first()
    assert sl is not None
    assert ds.sources.id == sl.id

    wcs = session.scalars(
        sa.select(WorldCoordinates).where(WorldCoordinates.source_list_id == sl.id)
    ).first()
    assert wcs is not None
    assert ds.wcs.id == wcs.id

    zp = session.scalars(
        sa.select(ZeroPoint).where(ZeroPoint.source_list_id == sl.id)
    ).first()
    assert zp is not None
    assert ds.zp.id == zp.id

    aliased_table = sa.orm.aliased(image_upstreams_association_table)
    sub = session.scalars(
        sa.select(Image).join(
            image_upstreams_association_table,
            sa.and_(
                image_upstreams_association_table.c.upstream_id == ref_id,
                image_upstreams_association_table.c.downstream_id == Image.id,
            )
        ).join(
            aliased_table,
            sa.and_(
                aliased_table.c.upstream_id == im.id,
                aliased_table.c.downstream_id == Image.id,
            )
        )
    ).first()

    assert sub is not None
    assert ds.sub_image.id == sub.id

    sl = session.scalars(
        sa.select(SourceList).where(SourceList.image_id == sub.id, SourceList.is_sub.is_(True))
    ).first()

    assert sl is not None
    assert ds.detections.id == sl.id

    # TODO: add the cutouts and measurements, but we need to produce them first!


def test_parameters( test_config ):
    """Test that pipeline parameters are being set properly"""

    # Verify that we _enforce_no_new_attrs works
    kwargs = { 'pipeline': { 'keyword_does_not_exist': 'testing' } }
    with pytest.raises( AttributeError, match='object has no attribute' ):
        failed = Pipeline( **kwargs )

    # Verify that we can override from the yaml config file
    pipeline = Pipeline()
    assert not pipeline.preprocessor.pars['use_sky_subtraction']
    assert pipeline.astro_cal.pars['cross_match_catalog'] == 'GaiaDR3'
    assert pipeline.astro_cal.pars['catalog'] == 'GaiaDR3'
    assert pipeline.subtractor.pars['method'] == 'naive'

    # Verify that manual override works for all parts of pipeline
    overrides = { 'preprocessing': { 'steps': [ 'overscan', 'linearity'] },
                  # 'extraction': # Currently has no parameters defined
                  'astro_cal': { 'cross_match_catalog': 'override' },
                  'photo_cal': { 'cross_match_catalog': 'override' },
                  'subtraction': { 'method': 'override' },
                  'detection': { 'threshold': 3.14 },
                  'cutting': { 'cutout_size': 666 },
                  'measurement': { 'photometry_method': 'override' }
                 }
    pipelinemodule = { 'preprocessing': 'preprocessor',
                       'subtraction': 'subtractor',
                       'detection': 'detector',
                       'cutting': 'cutter',
                       'measurement': 'measurer'
                      }

    # TODO: this is based on a temporary "example_pipeline_parameter" that will be removed later
    pipeline = Pipeline( pipeline={ 'example_pipeline_parameter': -999 } )
    assert pipeline.pars['example_pipeline_parameter'] == -999

    pipeline = Pipeline( **overrides )
    for module, subst in overrides.items():
        if module in pipelinemodule:
            pipelinemod = getattr( pipeline, pipelinemodule[module] )
        else:
            pipelinemod = getattr( pipeline, module )
        for key, val in subst.items():
            assert pipelinemod.pars[key] == val


# TODO: need to finish this test (i.e., finish subtraction, source extraction from sub image, etc)
def test_data_flow(decam_exposure, decam_reference, decam_default_calibrators):
    """Test that the pipeline runs end-to-end."""
    exposure = decam_exposure

    ref = decam_reference
    sec_id = ref.section_id
    try:  # cleanup the file at the end

        p = Pipeline()
        assert p.extractor.pars.threshold != 3.14
        assert p.detector.pars.threshold != 3.14

        with pytest.raises(NotImplementedError, match="This needs to be updated for detection on a subtraction."):
            # TODO: failure modes! if the run fails we never get a datastore back, and can't issue a delete_everything!
            ds = p.run(exposure, sec_id)
        return  # TODO: need to finish subtraction and detection etc and bring this back:
        # commit to DB using this session
        with SmartSession() as session:
            ds.save_and_commit(session=session)

        # use a new session to query for the results
        with SmartSession() as session:
            # check that everything is in the database
            provs = session.scalars(sa.select(Provenance)).all()
            assert len(provs) > 0
            prov_processes = [p.process for p in provs]
            expected_processes = ['preprocessing', 'extraction', 'astro_cal', 'photo_cal', 'subtraction', 'detection']
            for process in expected_processes:
                assert process in prov_processes

            check_datastore_and_database_have_everything(exp_id, sec_id, ref_id, session, ds)

        # feed the pipeline the same data, but missing the upstream data
        attributes = ['image', 'sources', 'wcs', 'zp', 'sub_image', 'detections']

        for i in range(len(attributes)):
            for j in range(i):
                setattr(ds, attributes[j], None)  # get rid of all data up to the current attribute

            ds = p.run(ds)

            # commit to DB using this session
            with SmartSession() as session:
                ds.save_and_commit(session=session)

            # use a new session to query for the results
            with SmartSession() as session:
                check_datastore_and_database_have_everything(exp_id, sec_id, ref_id, session, ds)

        # print(ds.image.filepath)
        # print(ds.sub_image.filepath)
        # make sure we can remove the data from the end to the beginning and recreate it
        for i in range(len(attributes)):
            for j in range(i):
                # print(f'i= {i}, j= {j}. Removing attribute: {attributes[-j-1]}')

                obj = getattr(ds, attributes[-j-1])
                with SmartSession() as session:
                    # obj = obj.recursive_merge(session=session)
                    obj = session.merge(obj)
                    if isinstance(obj, FileOnDiskMixin):
                        obj.delete_from_disk_and_database(session=session, commit=True)

                setattr(ds, attributes[-j-1], None)

            ds = p.run(ds)

            # commit to DB using this session
            with SmartSession() as session:
                ds.save_and_commit(session=session)

            # use a new session to query for the results
            with SmartSession() as session:
                check_datastore_and_database_have_everything(exp_id, sec_id, ref_id, session, ds)

    finally:
        if 'ds' in locals():
            ds.delete_everything()
        # added this cleanup to make sure the temp data folder is cleaned up
        # this should be removed after we add datastore failure modes (issue #150)
        shutil.rmtree(os.path.join(os.path.dirname(exposure.get_fullpath()), '115'), ignore_errors=True)

