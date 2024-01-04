import pytest
import hashlib
import uuid

import numpy as np

from astropy.wcs import WCS
from astropy.io import fits

from util.exceptions import BadMatchException
from models.base import SmartSession
from models.image import Image
from models.world_coordinates import WorldCoordinates
from pipeline.astro_cal import AstroCalibrator


def test_solve_wcs_scamp_failures( gaiadr3_excerpt, example_ds_with_sources_and_psf ):
    catexp = gaiadr3_excerpt
    ds = example_ds_with_sources_and_psf

    # Make sure it fails if we give too stringent of a minimum residual
    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[20.], mag_range=4., min_stars=50,
                                  max_resid=0.01 )
    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        wcs = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )

    # Make sure it fails if we give it too small of a crossid radius.
    # Note that this one is passed directly to _solve_wcs_scamp.
    # _solve_wcs_scamp doesn't read what we pass to AstroCalibrator
    # constructor, because that is an array of crossid_rad values to
    # try, whereas _solve_wcs_scamp needs a single value.  (The
    # iteration happens outsice _solve_wcs_scamp.)
    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[20.], mag_range=4., min_stars=50 )
    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        wcs = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp, crossid_rad=0.01 )

    # Make sure it fails if min_frac_matched is too high
    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[20.], mag_range=4., min_stars=50,
                                  min_frac_matched=0.8 )
    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        wcs = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )

    # Make sure it fails if min_matched_stars is too high
    # (For this test image, there's only something like 120 objects in the source list.)
    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[20.], mag_range=4., min_stars=50,
                                  min_matches=50 )
    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        wcs = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )


def test_solve_wcs_scamp( gaiadr3_excerpt, example_ds_with_sources_and_psf ):
    catexp = gaiadr3_excerpt
    ds = example_ds_with_sources_and_psf

    # Make True for visual testing purposes
    if False:
        catexp.ds9_regfile( 'catexp.reg', radius=4 )
        ds.sources.ds9_regfile( 'sources.reg', radius=3 )

    orighdr = ds.image._raw_header.copy()

    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[20.], mag_range=4., min_stars=50 )
    astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )

    # Because this was a ZTF image that had a WCS already, the new WCS
    # should be damn close, but not identical (since there's no way we
    # used exactly the same set of sources and stars, plus this was a
    # cropped ZTF image, not the full image).
    allsame = True
    for i in [ 1, 2 ]:
        for j in range( 17 ):
            diff = np.abs( ( orighdr[f'PV{i}_{j}'] - ds.image._raw_header[f'PV{i}_{j}'] ) / orighdr[f'PV{i}_{j}'] )
            if diff > 1e-6:
                allsame = False
                break
    assert not allsame

    #...but check that they are close
    wcsold = WCS( orighdr )
    scolds = wcsold.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )
    wcsnew = WCS( ds.image._raw_header )
    scnews = wcsnew.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )
    for scold, scnew in zip( scolds, scnews ):
        assert scold.ra.value == pytest.approx( scnew.ra.value, abs=1./3600. )
        assert scold.dec.value == pytest.approx( scnew.dec.value, abs=1./3600. )


def test_run_scamp( decam_example_reduced_image_ds_with_wcs ):
    ds, origwcs, xvals, yvals, origmd5 = decam_example_reduced_image_ds_with_wcs

    # Make sure that the new WCS is different from the original wcs
    # (since we know the one that came in the decam exposure is approximate)
    # BUT, make sure that it's within 40", because the original one, while
    # not great, is *something*
    origscs = origwcs.pixel_to_world( xvals, yvals )
    newscs = ds.wcs.wcs.pixel_to_world( xvals, yvals )
    for origsc, newsc in zip( origscs, newscs ):
        assert not origsc.ra.value == pytest.approx( newsc.ra.value, abs=1./3600. )
        assert not origsc.dec.value == pytest.approx( newsc.dec.value, abs=1./3600. )
        assert origsc.ra.value == pytest.approx( newsc.ra.value, abs=40./3600. )   # cos(dec)...
        assert origsc.dec.value == pytest.approx( newsc.dec.value, abs=40./3600. )

    # These next few lines will need to be done after astrometry is done.  Right now,
    # we don't do saving and committing inside the Astrometor.run method.
    update_image_header = False
    if not ds.image.astro_cal_done:
        ds.image.astro_cal_done = True
        update_image_header = True
    ds.save_and_commit( update_image_header=update_image_header, overwrite=True )

    with SmartSession() as session:
        # Make sure the WCS made it into the databse
        q = ( session.query( WorldCoordinates )
              .filter( WorldCoordinates.source_list_id==ds.sources.id )
              .filter( WorldCoordinates.provenance_id==ds.wcs.provenance.id ) )
        assert q.count() == 1
        dbwcs = q.first()
        dbscs = dbwcs.wcs.pixel_to_world( xvals, yvals )
        for newsc, dbsc in zip( newscs, dbscs ):
            assert dbsc.ra.value == pytest.approx( newsc.ra.value, abs=0.01/3600. )
            assert dbsc.dec.value == pytest.approx( newsc.dec.value, abs=0.01/3600. )

        # Make sure the image got updated properly on the database
        # and on disk
        q = session.query( Image ).filter( Image.id==ds.image.id )
        assert q.count() == 1
        foundim = q.first()
        assert foundim.md5sum_extensions[0] == ds.image.md5sum_extensions[0]
        assert foundim.md5sum_extensions[0] != origmd5
        with open( foundim.get_fullpath()[0], 'rb' ) as ifp:
            md5 = hashlib.md5()
            md5.update( ifp.read() )
            assert uuid.UUID( md5.hexdigest() ) == foundim.md5sum_extensions[0]
        # This is probably redundant given the md5sum test we just did....
        ds.image._raw_header = None
        for kw in foundim.raw_header:
            # SIMPLE can't be an index to a Header.  (This is sort
            # of a weird thing in the astropy Header interface.)
            # BITPIX doesn't match because the ds.image raw header
            # was constructed from the exposure that had been
            # BSCALEd, even though the image we wrote to disk fully
            # a float (BITPIX=-32).
            if kw in [ 'SIMPLE', 'BITPIX' ]:
                continue
            assert foundim.raw_header[kw] == ds.image.raw_header[kw]

        # Make sure the new WCS got written to the FITS file
        with fits.open( foundim.get_fullpath()[0] ) as hdul:
            imhdr = hdul[0].header
        imwcs = WCS( hdul[0].header )
        imscs = imwcs.pixel_to_world( xvals, yvals )
        for newsc, imsc in zip( newscs, imscs ):
            assert newsc.ra.value == pytest.approx( imsc.ra.value, abs=0.01/3600. )
            assert newsc.dec.value == pytest.approx( imsc.dec.value, abs=0.01/3600. )

        # Make sure the archive has the right md5sum
        info = foundim.archive.get_info( f'{foundim.filepath}.image.fits' )
        assert info is not None
        assert uuid.UUID( info['md5sum'] ) == foundim.md5sum_extensions[0]


# TODO : test that it fails when it's supposed to

