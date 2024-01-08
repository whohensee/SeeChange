import pytest
import random

import numpy as np
import astropy.wcs

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import Provenance
from models.image import Image

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


def test_alignment_in_image( ptf_reference_images, code_version ):
    try:  # cleanup at the end
        images_to_align = ptf_reference_images[:4]  # speed things up using fewer images
        new_image = Image.from_images( images_to_align )
        new_image.provenance = Provenance(
            code_version=code_version,
            parameters={'alignment': {'method': 'swarp', 'to_index': 'last'}, 'test_parameter': 'test_value'},
            upstreams=new_image.get_upstream_provenances(),
            process='coaddition',
            is_testing=True,
        )
        if new_image.provenance.parameters['alignment']['to_index'] == 'last':
            new_image.ref_image_index = len(images_to_align) - 1
        elif new_image.provenance.parameters['alignment']['to_index'] == 'first':
            new_image.ref_image_index = 0
        else:
            raise ValueError(
                f"Unknown alignment reference index: {new_image.provenance.parameters['alignment']['to_index']}"
            )
        new_image.new_image = None
        new_image.data = np.sum([image.data for image in new_image.aligned_images], axis=0)
        new_image.save()
        aligned = new_image.aligned_images

        assert new_image.upstream_images == images_to_align
        assert len(aligned) == len(images_to_align)
        assert np.array_equal(aligned[-1].data, images_to_align[-1].data)
        ref = images_to_align[-1]

        # check that images are aligned properly
        for image in new_image.aligned_images:
            check_aligned(image, ref)

        # check that unaligned images do not pass the check
        for image in new_image.upstream_images:
            if image == ref:
                continue
            with pytest.raises(AssertionError):
                check_aligned(image, ref)

        # add new image to database
        with SmartSession() as session:
            new_image = session.merge(new_image)
            session.commit()

        # should be able to recreate aligned images from scratch
        with SmartSession() as session:
            loaded_image = session.scalars(sa.select(Image).where(Image.id == new_image.id)).first()
            assert loaded_image is not None
            assert len(loaded_image.aligned_images) == len(images_to_align)
            assert np.array_equal(loaded_image.aligned_images[-1].data, images_to_align[-1].data)

            # check that images are aligned properly
            for image in loaded_image.aligned_images:
                check_aligned(image, ref)

    finally:
        ImageAligner.cleanup_temp_images()
        new_image.delete_from_disk_and_database(remove_downstream_data=True)


def check_aligned(image1, image2):
    d1 = image1.data.copy()
    d2 = image2.data.copy()

    row_func1 = np.nansum(d1 - np.nanmedian(d1, axis=1, keepdims=True), axis=1)
    row_func2 = np.nansum(d2 - np.nanmedian(d2, axis=1, keepdims=True), axis=1)

    col_func1 = np.nansum(d1 - np.nanmedian(d1, axis=0, keepdims=True), axis=0)
    col_func2 = np.nansum(d2 - np.nanmedian(d2, axis=0, keepdims=True), axis=0)

    xcorr_rows = np.correlate(row_func1, row_func2, mode='full')
    xcorr_cols = np.correlate(col_func1, col_func2, mode='full')

    assert abs(np.argmax(xcorr_rows) - len(xcorr_rows) // 2) <= 1
    assert abs(np.argmax(xcorr_cols) - len(xcorr_cols) // 2) <= 1
