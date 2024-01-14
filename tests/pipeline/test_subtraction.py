import uuid

import numpy as np

from pipeline.data_store import DataStore


def test_subtraction_data_products(ptf_ref, ptf_supernova_images, subtractor):
    assert len(ptf_supernova_images) == 2
    image1, image2 = ptf_supernova_images

    assert image1.sources is not None
    assert image1.psf is not None
    assert image1.wcs is not None
    assert image1.zp is not None

    # run the subtraction like you'd do in the real pipeline (calls get_reference and get_subtraction internally)
    subtractor.pars.test_parameter = uuid.uuid4().hex
    subtractor.pars.method = 'naive'
    assert subtractor.pars.alignment['to_index'] == 'new'  # make sure alignment is configured to new, not latest image
    ds = subtractor.run(image1)

    # check that we don't lazy load a subtracted image, but recalculate it
    assert subtractor.has_recalculated

    # check that the image is correctly ingested with all products
    assert ds.image is not None
    assert ds.sources is not None
    assert ds.psf is not None
    assert ds.wcs is not None
    assert ds.zp is not None

    # check that reference is loaded automatically when calling the subtraction
    assert ds.reference is not None
    assert ds.reference.id == ptf_ref.id
    assert ds.reference != ptf_ref  # not the same object, it was loaded from DB!
    assert ds.reference.from_db  # it was loaded from DB!

    # check that the diff image is created
    assert ds.sub_image is not None
    assert ds.sub_image.data is not None


