import os
import warnings

import pytest

import sqlalchemy as sa

from models.base import SmartSession, _logger
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

from pipeline.data_store import DataStore
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from pipeline.subtraction import Subtractor
from pipeline.cutting import Cutter
from pipeline.measurement import Measurer


@pytest.fixture
def preprocessor(test_config):
    prep = Preprocessor(**test_config.value('preprocessing'))
    prep.pars._enforce_no_new_attrs = False
    prep.pars.test_parameter = prep.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    prep.pars._enforce_no_new_attrs = True

    return prep


@pytest.fixture
def extractor(test_config):
    extr = Detector(**test_config.value('extraction'))
    extr.pars._enforce_no_new_attrs = False
    extr.pars.test_parameter = extr.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    extr.pars._enforce_no_new_attrs = True

    return extr


@pytest.fixture
def astrometor(test_config):
    astrom = AstroCalibrator(**test_config.value('astro_cal'))
    astrom.pars._enforce_no_new_attrs = False
    astrom.pars.test_parameter = astrom.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    astrom.pars._enforce_no_new_attrs = True

    return astrom


@pytest.fixture
def photometor(test_config):
    photom = PhotCalibrator(**test_config.value('photo_cal'))
    photom.pars._enforce_no_new_attrs = False
    photom.pars.test_parameter = photom.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    photom.pars._enforce_no_new_attrs = True

    return photom


@pytest.fixture
def subtractor(test_config):
    sub = Subtractor(**test_config.value('subtraction'))
    sub.pars._enforce_no_new_attrs = False
    sub.pars.test_parameter = sub.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    sub.pars._enforce_no_new_attrs = True

    return sub


@pytest.fixture
def detector(test_config):
    det = Detector(**test_config.value('detection'))
    det.pars._enforce_no_new_attrs = False
    det.pars.test_parameter = det.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    det.pars._enforce_no_new_attrs = False

    return det


@pytest.fixture
def cutter(test_config):
    cut = Cutter(**test_config.value('cutting'))
    cut.pars._enforce_no_new_attrs = False
    cut.pars.test_parameter = cut.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    cut.pars._enforce_no_new_attrs = False

    return cut


@pytest.fixture
def measurer(test_config):
    meas = Measurer(**test_config.value('measurement'))
    meas.pars._enforce_no_new_attrs = False
    meas.pars.test_parameter = meas.pars.add_par(
        'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
    )
    meas.pars._enforce_no_new_attrs = False

    return meas


@pytest.fixture
def datastore_factory(
        data_dir,
        preprocessor,
        extractor,
        astrometor,
        photometor,
        detector,
        cutter,
        measurer,
):
    """Provide a function that returns a datastore with all the products based on the given exposure and section ID.

    To use this data store in a test where new data is to be generated,
    simply change the pipeline object's "test_parameter" value to a unique
    new value, so the provenance will not match and the data will be regenerated.

    EXAMPLE
    -------
    extractor.pars.test_parameter = uuid.uuid().hex
    extractor.run(datastore)
    assert extractor.has_recalculated is True
    """
    def make_datastore(*args, cache_dir=None, cache_base_name=None, session=None):
        code_version = args[0].provenance.code_version
        ds = DataStore(*args)  # make a new datastore

        with SmartSession(session) as session:
            code_version = session.merge(code_version)
            if ds.image is not None:  # if starting from an externally provided Image, must merge it first
                ds.image = ds.image.recursive_merge(session)

            ############ preprocessing to create image ############

            if ds.image is None and cache_dir is not None and cache_base_name is not None:
                # check if preprocessed image is in cache
                cache_name = cache_base_name + '.image.fits.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    _logger.debug('loading image from cache. ')
                    ds.image = Image.copy_from_cache(cache_dir, cache_name)
                    upstreams = [ds.exposure.provenance] if ds.exposure is not None else []  # images without exposure
                    prov = Provenance(
                        code_version=code_version,
                        process='preprocessing',
                        upstreams=upstreams,
                        parameters=preprocessor.pars.to_dict(),
                        is_testing=True,
                    )
                    prov.update_id()
                    prov = prov.recursive_merge(session)

                    # if Image already exists on the database, use that instead of this one
                    existing = session.scalars(sa.select(Image).where(Image.filepath == ds.image.filepath)).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        for key in sa.inspect(ds.image).mapper.columns.keys():
                            value = getattr(ds.image, key)
                            if (
                                    key not in ['id', 'image_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.image = existing  # replace with the existing row
                    ds.image.provenance = prov

                    # make sure this is saved to the archive as well
                    ds.image.save(verify_md5=False)

            if ds.image is None:  # make the preprocessed image
                _logger.debug('making preprocessed image. ')
                ds = preprocessor.run(ds)
                ds.image.save()
                output_path = ds.image.copy_to_cache(cache_dir)
                if cache_dir is not None and cache_base_name is not None and output_path != cache_path:
                    warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

            ############# extraction to create sources #############
            if cache_dir is not None and cache_base_name is not None:
                cache_name = cache_base_name + '.sources.fits.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    _logger.debug('loading source list from cache. ')
                    ds.sources = SourceList.copy_from_cache(cache_dir, cache_name)

                    prov = Provenance(
                        code_version=code_version,
                        process='extraction',
                        upstreams=[ds.image.provenance],
                        parameters=extractor.pars.to_dict(),
                        is_testing=True,
                    )
                    prov.update_id()
                    prov = prov.recursive_merge(session)

                    # if SourceList already exists on the database, use that instead of this one
                    existing = session.scalars(
                        sa.select(SourceList).where(SourceList.filepath == ds.sources.filepath)
                    ).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        for key in sa.inspect(ds.sources).mapper.columns.keys():
                            value = getattr(ds.sources, key)
                            if (
                                key not in ['id', 'image_id', 'created_at', 'modified'] and
                                value is not None
                            ):
                                setattr(existing, key, value)
                        ds.sources = existing  # replace with the existing row

                    ds.sources.provenance = prov
                    ds.sources.image = ds.image

                    # make sure this is saved to the archive as well
                    ds.sources.save(verify_md5=False)

                cache_name = cache_base_name + '.psf.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    _logger.debug('loading PSF from cache. ')
                    ds.psf = PSF.copy_from_cache(cache_dir, cache_name)

                    prov = Provenance(
                        code_version=code_version,
                        process='extraction',
                        upstreams=[ds.image.provenance],
                        parameters=extractor.pars.to_dict(),
                        is_testing=True,
                    )
                    prov.update_id()
                    prov = prov.recursive_merge(session)

                    # if PSF already exists on the database, use that instead of this one
                    existing = session.scalars(
                        sa.select(PSF).where(PSF.filepath == ds.psf.filepath)
                    ).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        for key in sa.inspect(ds.psf).mapper.columns.keys():
                            value = getattr(ds.psf, key)
                            if (
                                    key not in ['id', 'image_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.psf = existing  # replace with the existing row

                    ds.psf.provenance = prov
                    ds.psf.image = ds.image

                    # make sure this is saved to the archive as well
                    ds.psf.save(verify_md5=False, overwrite=True)

            if ds.sources is None or ds.psf is None:  # make the source list from the regular image
                _logger.debug('extracting sources. ')
                ds = extractor.run(ds)
                ds.sources.save()
                ds.sources.copy_to_cache(cache_dir)
                ds.psf.save(overwrite=True)
                output_path = ds.psf.copy_to_cache(cache_dir)
                if cache_dir is not None and cache_base_name is not None and output_path != cache_path:
                    warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

            ############## astro_cal to create wcs ################

            if cache_dir is not None and cache_base_name is not None:
                cache_name = cache_base_name + '.wcs.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    _logger.debug('loading WCS from cache. ')
                    ds.wcs = WorldCoordinates.copy_from_cache(cache_dir, cache_name)
                    prov = Provenance(
                        code_version=code_version,
                        process='astro_cal',
                        upstreams=[ds.sources.provenance],
                        parameters=astrometor.pars.to_dict(),
                        is_testing=True,
                    )
                    prov.update_id()
                    prov = prov.recursive_merge(session)

                    # check if WCS already exists on the database
                    existing = session.scalars(
                        sa.select(WorldCoordinates).where(
                            WorldCoordinates.sources_id == ds.sources.id,
                            WorldCoordinates.provenance_id == prov.id
                        )
                    ).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        for key in sa.inspect(ds.wcs).mapper.columns.keys():
                            value = getattr(ds.wcs, key)
                            if (
                                    key not in ['id', 'sources_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.wcs = existing  # replace with the existing row

                    ds.wcs.provenance = prov
                    ds.wcs.sources = ds.sources

            if ds.wcs is None:  # make the WCS
                _logger.debug('Running astrometric calibration')
                ds = astrometor.run(ds)
                if cache_dir is not None and cache_base_name is not None:
                    output_path = ds.wcs.copy_to_cache(cache_dir, cache_name)  # must provide a name because this one isn't a FileOnDiskMixin
                    if output_path != cache_path:
                        warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

            ########### photo_cal to create zero point ############

            if cache_dir is not None and cache_base_name is not None:
                cache_name = cache_base_name + '.zp.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    _logger.debug('loading zero point from cache. ')
                    ds.zp = ZeroPoint.copy_from_cache(cache_dir, cache_name)
                    prov = Provenance(
                        code_version=code_version,
                        process='photo_cal',
                        upstreams=[ds.sources.provenance],
                        parameters=photometor.pars.to_dict(),
                        is_testing=True,
                    )
                    prov.update_id()
                    prov = prov.recursive_merge(session)

                    # check if ZP already exists on the database
                    existing = session.scalars(
                        sa.select(ZeroPoint).where(
                            ZeroPoint.sources_id == ds.sources.id,
                            ZeroPoint.provenance_id == prov.id
                        )
                    ).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        for key in sa.inspect(ds.zp).mapper.columns.keys():
                            value = getattr(ds.zp, key)
                            if (
                                    key not in ['id', 'sources_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.zp = existing  # replace with the existing row

                    ds.zp.provenance = prov
                    ds.zp.sources = ds.sources

            if ds.zp is None:  # make the zero point
                _logger.debug('Running photometric calibration')
                ds = photometor.run(ds)
                if cache_dir is not None and cache_base_name is not None:
                    output_path = ds.zp.copy_to_cache(cache_dir, cache_name)
                    if output_path != cache_path:
                        warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

            # TODO: add the same cache/load and processing for the rest of the pipeline

            ds.save_and_commit(session=session)

            return ds

    return make_datastore
