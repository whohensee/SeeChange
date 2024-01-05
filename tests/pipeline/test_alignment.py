import pytest
import random

import astropy.wcs

from pipeline.alignment import ImageAligner


def test_warp_decam( decam_datastore, decam_reference ):
    ds = decam_datastore

    try:
        ds.get_reference()
        aligner = ImageAligner()
        warped = aligner.run( ds.reference.image, ds.image )
        warped.filepath = f'warp_test_{"".join(random.choices("abcdefghijklmnopqrstuvwxyz",k=10))}'

        assert warped.data.shape == ds.image.data.shape

        # Check a couple of spots on the image
        # First, around a star:
        assert ds.image.data[ 2223:2237, 545:559 ].sum() == pytest.approx( 58014.1, rel=0.01 )
        assert warped.data[ 2223:2237, 545:559 ].sum() == pytest.approx( 22597.9, rel=0.01 )
        # And a blank spot
        assert ds.image.data[ 2243:2257, 575:589 ].sum() == pytest.approx( 35298.6, rel=0.01 )    # sky not subtracted
        assert warped.data[ 2243:2257, 575:589 ].sum() == pytest.approx( 971.7, rel=0.01 )

        # Make sure the warped image WCS is about right.  We don't
        # expect it to be exactly identical, but it should be very
        # close.
        imwcs = ds.wcs.wcs
        warpwcs = astropy.wcs.WCS( warped.raw_header )
        x = [ 256, 1791, 256, 1791, 1024 ]
        y = [ 256, 256, 3839, 3839, 2048 ]
        imsc = imwcs.pixel_to_world( x, y )
        warpsc = warpwcs.pixel_to_world( x, y )
        assert all( [ i.ra.deg == pytest.approx(w.ra.deg, abs=0.1/3600.) for i, w in zip( imsc, warpsc ) ] )
        assert all( [ i.dec.deg == pytest.approx(w.dec.deg, abs=0.1/3600.) for i, w in zip( imsc, warpsc ) ] )

    finally:
        if 'warped' in locals():
            warped.delete_from_disk_and_database()

