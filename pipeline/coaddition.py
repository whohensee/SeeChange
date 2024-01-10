
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

from sep import Background

from models.provenance import Provenance
from models.image import Image

from pipeline.parameters import Parameters

from improc.bitmask_tools import dilate_bitflag
from improc.inpainting import Inpainter
from improc.tools import sigma_clipping


class ParsCoadd(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.method = self.add_par(
            'method',
            'zogy',
            str,
            'Coaddition method.  Currently only "naive" and "zogy" are supported. ',
            critical=True
        )

        self.alignment = self.add_par(
            'alignment',
            {},
            dict,
            'Alignment parameters. ',
            critical=True
        )

        self.inpainting = self.add_par(
            'inpainting',
            {},
            dict,
            'Inpainting parameters. ',
            critical=True
        )

        self.noise_estimator = self.add_par(
            'noise_estimator',
            'sep',
            str,
            'Method to estimate noise (sigma) in the image.  '
            'Use "sep" or "sigma" for sigma clipping. ',
            critical=True,
        )

        self.flag_fwhm_factor = self.add_par(
            'flag_fwhm_factor',
            1.0,
            float,
            'Multiplicative factor for the PSF FWHM (in pixels) to use for dilating the flag maps. ',
            critical=True,
        )

        self.enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name(self):
        return 'coaddition'


class Coadder:
    """Use this class to coadd (stack) images together to make a deeper image.

    Each image should have a PSF and a ZeroPoint associated with it (and loaded!) when running coaddition.

    Images are expected to be aligned (use the Aligner) and should generally be of the same sky region.
    If not already aligned, they need to have the SourceList and WorldCoordinates loaded so that
    alignment can be done on the fly.

    Input images should also have a valid Provenance associated with them. Not that for any set of images
    that share a provenance hash, their respective downstreams (e.g., PSF) should also have a single provenance
    hash for each type of downstream.  This makes it possible to identify the upstream images' associated products
    based solely on the provenance's upstream hashes.

    Areas on the edges where the images are not overlapping (or bad pixels, etc.) are coadded but will
    contribute zero weight, so the total weight of that pixel will be zero (if all input images have a bad pixel)
    or they would have lower weight if only some images had bad pixels there.

    Remember that some coaddition methods use convolution with the PSF, so any effects of individual pixels
    could affect nearby pixels, depending on the size of the PSF.
    """

    def __init__( self, **kwargs ):
        self.pars = ParsCoadd(**kwargs)
        self.inpainter = Inpainter(**self.pars.inpainting)
        # the aligner object is created in the image object

    def _estimate_background(self, data):
        """Get the mean and noise RMS of the background of the given image.

        Parameters
        ----------
        data: ndarray
            The image for which background should be estimated.

        Returns
        -------
        bkg: float
            The mean background in the image.
        sigma: float
            The RMS of the background in the image.
        """
        if self.pars.noise_estimator == 'sep':
            b = Background(data)
            bkg = b.globalback
            sigma = b.globalrms
        elif self.pars.noise_estimator.startswith('sigma'):
            bkg, sigma = sigma_clipping(data)
        else:
            raise ValueError(
                f'Unknown noise estimator: {self.pars.noise_estimator}.  Use "sep" or "sigma_clipping" or "bkg_rms". '
            )

        return bkg, sigma

    def _coadd_naive(self, images, weights=None, flags=None):
        """Simply sum the values in each image on top of each other.

        Parameters
        ----------
        images: list of Image or list of 2D ndarrays
            Images that have been aligned to each other.
        weights: list of 2D ndarrays
            The weights to use for each image.
            If images is given as Image objects, can be left as None.
        flags: list of 2D ndarrays
            The bit flags to use for each image.
            If images is given as Image objects, can be left as None.

        Returns
        -------
        outim: ndarray
            The image data after coaddition.
        outwt: ndarray
            The weight image after coaddition.
        outfl: ndarray
            The bit flags array after coaddition.
        """
        if not all(type(image) == type(images[0]) for image in images):
            raise ValueError('Not all images are of the same type. ')
        if isinstance(images[0], Image):
            data = [image.data for image in images]
            weights = [image.weight for image in images]
            flags = [image.flags for image in images]
        elif isinstance(images[0], np.ndarray):
            data = images
        else:
            raise ValueError('images must be a list of Image objects or 2D arrays. ')

        imcube = np.array(data)
        outim = np.sum(imcube, axis=0)

        wtcube = np.array(weights)
        varflag = wtcube == 0
        wtcube2 = wtcube ** 2
        wtcube2[varflag] = np.nan
        varmap = 1 / wtcube2

        outwt = 1 / np.sqrt(np.sum(varmap, axis=0))

        outfl = np.zeros(outim.shape, dtype='uint16')
        for f in flags:
            outfl |= f

        return outim, outwt, outfl

    def _zogy_core(self, datacube, psfcube, sigmas, flux_zps):
        """Perform the core Zackay & Ofek proper image coaddition on the input data cube.

        Parameters
        ----------
        datacube: ndarray
            The data cube to coadd. Can be images or weight maps (or anything else).
        psfcube: ndarray
            The PSF cube to use for coaddition.
        sigmas: ndarray
            The background noise estimate for each image in the data cube.
            Must be a 1D array with a length equal to the first axis of the data cube.
            It could have additional dimensions, but it will be reshaped to be multiplied
            with the data cube and psf cube.
        flux_zps: ndarray
            The flux zero points for each image in the data cube.
            Must be a 1D array with a length equal to the first axis of the data cube.
            It could have additional dimensions, but it will be reshaped to be multiplied
            with the data cube and psf cube.

        Returns
        -------
        outdata: ndarray
            The coadded 2D data array.
        outpsf: ndarray
            The coadded 2D PSF cube.
        score: ndarray
            The matched-filter result of cross correlating outdata with outpsf.
        """
        # data verification:
        if datacube.shape != psfcube.shape:
            raise ValueError('The data cube and PSF cube must have the same shape. ')
        if len(datacube.shape) != 3:
            raise ValueError('The data cube and PSF cube must have 3 dimensions. ')

        sigmas = np.reshape(np.array(sigmas), (len(sigmas), 1, 1))
        if sigmas.size != datacube.shape[0]:
            raise ValueError('The sigmas array must have the same length as the first axis of the data cube. ')

        flux_zps = np.reshape(np.array(flux_zps), (len(flux_zps), 1, 1))
        if flux_zps.size != datacube.shape[0]:
            raise ValueError('The flux_zps array must have the same length as the first axis of the data cube. ')

        if np.sum(np.isnan(datacube)) > 0:
            raise ValueError('There are NaNs values in the data cube! Use inpainting to remove them... ')

        # calculations:
        datacube_f = fft2(datacube)
        psfcube_f = fft2(psfcube)

        # paper ref: https://ui.adsabs.harvard.edu/abs/2017ApJ...836..188Z/abstract
        score_f = np.sum(flux_zps / sigmas ** 2 * np.conj(psfcube_f) * datacube_f, axis=0)  # eq 7
        psf_f = np.sqrt(np.sum(flux_zps ** 2 / sigmas ** 2 * np.abs(psfcube_f) ** 2, axis=0))  # eq 10
        outdata_f = score_f / psf_f  # eq 8

        outdata = fftshift(ifft2(outdata_f).real)
        score = fftshift(ifft2(score_f).real)
        psf = fftshift(ifft2(psf_f).real)
        psf = psf / np.sum(psf)

        return outdata, psf, score

    def _coadd_zogy(
            self,
            images,
            weights=None,
            flags=None,
            psf_clips=None,
            psf_fwhms=None,
            flux_zps=None,
            bkg_means=None,
            bkg_sigmas=None,
    ):
        """Use Zackay & Ofek proper image coaddition to add the images together.

        This method uses the PSF of each image to coadd images with proper weight
        given to each frequency in Fourier space, such that it preserves information
        even when using images with different PSFs.

        Parameters
        ----------
        images: list of Image or list of 2D ndarrays
            Images that have been aligned to each other.
            Each image must also have a PSF object attached.
        weights: list of 2D ndarrays
            The weights to use for each image.
            If images is given as Image objects, can be left as None.
        flags: list of 2D ndarrays
            The bit flags to use for each image.
            If images is given as Image objects, can be left as None.
        psf_clips: list of 2D ndarrays
            The PSF images to use for each image.
            If images is given as Image objects, can be left as None.
        psf_fwhms: list of floats
            The FWHM of the PSF for each image.
            If images is given as Image objects, can be left as None.
        flux_zps: list of floats
            The flux zero points for each image.
            If images is given as Image objects, can be left as None.
        bkg_means: list of floats
            The mean background for each image.
            If images is given as Image objects, can be left as None.
            This variable can be used to override the background estimation.
            If images are already background subtracted, set these to zeros.
        bkg_sigmas: list of floats
            The RMS of the background for each image.
            If images is given as Image objects, can be left as None.
            This variable can be used to override the background estimation.

        Returns
        -------
        outim: ndarray
            The image data after coaddition.
        outwt: ndarray
            The weight image after coaddition.
        outfl: ndarray
            The bit flags array after coaddition.
        psf: ndarray
            An array with the PSF of the output image.
        score: ndarray
            A matched-filtered score image of the coadded image.
        """
        if not all(type(image) == type(images[0]) for image in images):
            raise ValueError('Not all images are of the same type. ')

        if isinstance(images[0], Image):
            data = []
            flags = []
            weights = []
            psf_clips = []
            psf_fwhms = []
            flux_zps = []
            
            for image in images:
                data.append(image.data)
                flags.append(image.flags)
                weights.append(image.weight)
                psf_clips.append(image.psf.get_clip())
                psf_fwhms.append(image.psf.fwhm_pixels)
                flux_zps.append(10 ** (0.4 * image.zp.zp))

        elif isinstance(images[0], np.ndarray):
            data = images
        else:
            raise ValueError('images must be a list of Image objects or 2D arrays. ')

        # pad the PSFs to the same size as the image data
        psfs = []
        for array, psf in zip(data, psf_clips):
            padsize_x1 = int(np.ceil((array.shape[1] - psf.shape[1]) / 2))
            padsize_x2 = int(np.floor((array.shape[1] - psf.shape[1]) / 2))
            padsize_y1 = int(np.ceil((array.shape[0] - psf.shape[0]) / 2))
            padsize_y2 = int(np.floor((array.shape[0] - psf.shape[0]) / 2))
            psf_pad = np.pad(psf, ((padsize_y1, padsize_y2), (padsize_x1, padsize_x2)))
            psf_pad /= np.sum(psf_pad)
            psfs.append(psf_pad)

        # estimate the background if not given
        if bkg_means is None or bkg_sigmas is None:
            bkg_means = []
            bkg_sigmas = []
            for array in data:
                bkg, sigma = self._estimate_background(array)
                bkg_means.append(bkg)
                bkg_sigmas.append(sigma)

        imcube = np.array(data)
        flcube = np.array(flags)
        wtcube = np.array(weights)
        psfcube = np.array(psfs)
        bkg_means = np.reshape(np.array(bkg_means), (len(bkg_means), 1, 1))
        bkg_sigmas = np.reshape(np.array(bkg_sigmas), (len(bkg_sigmas), 1, 1))
        flux_zps = np.reshape(np.array(flux_zps), (len(flux_zps), 1, 1))

        # subtract the background
        imcube -= bkg_means

        # make sure to inpaint missing data
        # TODO: make sure images are scaled before inpainting, or add that in the inpainting code
        imcube = self.inpainter.run(imcube, flcube, wtcube)

        if np.any(np.isnan(imcube)):
            raise ValueError('There are still NaNs in the image data after inpainting!')

        # This is where the magic happens
        outim, psf, score = self._zogy_core(imcube, psfcube, bkg_sigmas, flux_zps)

        # coadd the variance as well
        varflag = wtcube == 0
        wtcube2 = wtcube ** 2
        wtcube2[varflag] = np.nan
        varmap = 1 / wtcube2
        varmap = self.inpainter.run(varmap, varflag, wtcube)  # wtcube doesn't do anything, maybe put something else?
        outvarmap, _, _ = self._zogy_core(varmap, psfcube, bkg_sigmas, flux_zps)
        outwt = 1 / np.sqrt(np.abs(outvarmap))

        outfl = np.zeros(outim.shape, dtype='uint16')
        for f, p in zip(flags, psf_fwhms):
            splash_pixels = int(np.ceil(p * self.pars.flag_fwhm_factor))
            outfl = outfl | dilate_bitflag(f, iterations=splash_pixels)

        return outim, outwt, outfl, psf, score

    def run(self, images):
        """Run coaddition on the given list of images, and return the coadded image.

        The images should have at least a set of SourceList and WorldCoordinates loaded, so they can be aligned.
        The images must also have a PSF and ZeroPoint loaded for the coaddition process.

        Parameters
        ----------
        images: list of Image objects

        Returns
        -------
        output: Image object
            The coadded image.
        """
        images.sort(key=lambda image: image.mjd)
        if self.pars.alignment['to_index'] == 'last':
            index = len(images) - 1
        elif self.pars.alignment['to_index'] == 'first':
            index = 0
        else:  # TODO: consider allowing a specific index as integer?
            raise ValueError(f"Unknown alignment reference index: {self.pars.alignment['to_index']}")
        output = Image.from_images(images, index=index)
        output.provenance = Provenance(
            code_version=images[0].provenance.code_version,
            parameters=self.pars.get_critical_pars(),
            upstreams=output.get_upstream_provenances(),
            process='coaddition',
        )
        output.is_coadd = True
        output.new_image = None

        if self.pars.method == 'naive':
            outim, outwt, outfl = self._coadd_naive(images)
        elif self.pars.method == 'zogy':
            outim, outwt, outfl, outpsf, outscore = self._coadd_zogy(images)
        else:
            raise ValueError(f'Unknown coaddition method: {self.pars.method}. Use "naive" or "zogy".')

        output.data = outim
        output.weight = outwt
        output.flags = outfl

        if 'outpsf' in locals():
            output.zogy_psf = outpsf  # TODO: do we have a better place to put this?
        if 'outscore' in locals():
            output.zogy_score = outscore

        return output
