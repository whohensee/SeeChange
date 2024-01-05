import pytest
import os
import shutil
import requests

import scipy
import numpy as np
import sqlalchemy as sa
from bs4 import BeautifulSoup
from astropy.io import fits

from models.base import SmartSession, FileOnDiskMixin, _logger
from models.ptf import PTF  # need this import to make sure PTF is added to the Instrument list
from models.image import Image
from util.retrydownload import retry_download


# TODO: replace this by a proper PTF Instrument class
def read_ptf_data(filepath):
    """Read a PTF image file and return the data
    """
    with fits.open(filepath) as hdu:
        data = hdu[0].data
        return data


# TODO: replace this by a proper PTF Instrument class
def read_ptf_header(filepath):
    """Read a PTF image file and read some header keywords
    """
    output = {}
    with fits.open(filepath) as hdu:
        header = hdu[0].header
        output['raw_header'] = header
        output['ra'] = header['TELRA']  # in degrees
        output['dec'] = header['TELDEC']  # in degrees
        output['exp_time'] = header['AEXPTIME']  # actual exposure time
        output['filter'] = header['FILTER']  # filter name
        output['mjd'] = header['OBSMJD']  # MJD of the observation
        output['end_mjd'] = header['OBSMJD'] + header['AEXPTIME'] / 86400.  # MJD of the end of the observation
        output['project'] = 'PTF'
        output['target'] = header['OBJECT']  # target name
        output['section_id'] = header['CCDID']
        output['telescope'] = 'Palomar 48 inch'
        output['instrument'] = 'PTF'
        output['bkg_mean_estimate'] = header['MEDSKY']
        output['bkg_rms_estimate'] = header['SKYSIG']
        output['type'] = 'Sci'
        output['format'] = 'fits'

        # preproc_bitflag??

        return output


# TODO: replace this by a proper PTF Instrument class
def calc_ptf_weight_flags(data):
    """Calculate the weight and flags for a PTF image
    """
    sz = 3  # size of the box for the background calculation (should this be a tunable parameter?)
    m = scipy.signal.convolve2d(data, np.ones((sz, sz)), mode="same") / sz ** 2
    m2 = scipy.signal.convolve2d( (data - m) ** 2, np.ones((sz, sz)), mode="same") / sz ** 2
    noise_rms = np.sqrt(m2)

    # smooth the noise to avoid unusually high or low values
    noise_rms = scipy.ndimage.median_filter(noise_rms, size=3)
    weight = 1 / noise_rms

    flags = np.zeros_like(data, dtype=np.int32)
    return weight, flags


@pytest.fixture(scope='session')
def ptf_downloader(provenance_preprocessing, cache_dir):
    cache_dir = os.path.join(cache_dir, 'PTF')

    def download_ptf_function(filename='PTF201104234316_2_o_44887_11.w.fits'):
        os.makedirs(cache_dir, exist_ok=True)
        cachedpath = os.path.join(cache_dir, filename)

        # first get the file into cache
        if os.path.isfile(cachedpath):
            _logger.info(f"{cachedpath} exists, not redownloading.")
        else:
            url = f'https://portal.nersc.gov/project/m2218/pipeline/test_images/{filename}'
            retry_download(url, cachedpath)  # make the cached copy

        if not os.path.isfile(cachedpath):
            raise FileNotFoundError(f"Can't read {cachedpath}. It should have been downloaded!")

        # can we load the Image object from the cache?
        if os.path.isfile(cachedpath + '.json'):
            image = Image.copy_from_cache(cache_dir, filename)
            image.provenance = provenance_preprocessing
        else:
            hdr_dict = read_ptf_header(cachedpath)
            image = Image(**hdr_dict)
            image.data = read_ptf_data(cachedpath)
            image.weight, image.flags = calc_ptf_weight_flags(image.data)
            image.provenance = provenance_preprocessing
            image.set_corners_from_header_wcs()
            image.calculate_coordinates()
            # image.invent_filepath()
            image.save()
            # os.makedirs(os.path.dirname(image.get_fullpath()), exist_ok=True)
            # shutil.copy2(cachedpath, image.get_fullpath())
            result = image.copy_to_cache(cache_dir, filename)
            assert result == cachedpath + '.json'

        return image

    return download_ptf_function


@pytest.fixture
def ptf_image(ptf_downloader):

    image = ptf_downloader()
    # check if this Image is already on the database
    # with SmartSession() as session:
    #     existing = session.scalars(sa.select(Image).where(Image.filepath == image.filepath)).first()
    #     if existing is not None:
    #         _logger.info(f"Found existing Image on database: {existing}")
    #         # overwrite the existing row data using the JSON cache file
    #         for key in sa.inspect(image).mapper.columns.keys():
    #             value = getattr(image, key)
    #             if (
    #                     key not in ['id', 'image_id', 'created_at', 'modified'] and
    #                     value is not None
    #             ):
    #                 setattr(existing, key, value)
    #         image = existing  # replace with the existing row
    #     else:
    #         session.add(image)
    #         session.commit()

    yield image

    image.delete_from_disk_and_database()


@pytest.fixture(scope='session')
def all_ptf_example_images(ptf_downloader):
    url = f'https://portal.nersc.gov/project/m2218/pipeline/test_images/'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    links = soup.find_all('a')
    filenames = [link.get('href') for link in links if link.get('href').endswith('.fits')]
    images = []
    for filename in filenames:
        new_image = ptf_downloader(filename=filename)
        images.append(new_image)

    yield images

    with SmartSession() as session:
        for image in images:
            image.delete_from_disk_and_database(session=session, commit=False)
        session.commit()


@pytest.fixture
def ptf_datastore(datastore_factory, ptf_image):
    ds = datastore_factory(ptf_image)
    yield ds
    ds.delete_everything()


def test_get_ptf_image(ptf_image):
    print(ptf_image)


# def test_get_all_ptf_images(all_ptf_example_images):
#     print(all_ptf_example_images)


# TODO: need more work to make the PTF datastore work (sextractor fails)
# def test_ptf_datastore(ptf_datastore):
#     print(ptf_datastore.sources)