
import numpy as np

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.base import SmartSession
from models.provenance import Provenance
from models.image import Image


class ParsSubtractor(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.method = self.add_par(
            'method',
            'hotpants',
            str,
            'Which subtraction method to use. Possible values are: "hotpants", "zogy". '
        )

        self.alignment = self.add_par(
            'alignment',
            {'method:': 'swarp', 'to_index': 'new'},
            dict,
            'How to align the reference image to the new image. This will be ingested by ImageAligner. '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'subtraction'


class Subtractor:
    def __init__(self, **kwargs):
        self.pars = ParsSubtractor(**kwargs)

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

        # TODO: add a reference cache here.

    @staticmethod
    def _subtract_naive(new_image, ref_image):
        """Subtract the reference from the image directly, assuming they are aligned and same shape.

        Doesn't do any fancy PSF matching or anything, just takes the difference of the data arrays.

        Parameters
        ----------
        new_image : Image
            The Image containing the new data, including the data array, weight, and flags
        ref_image : Image
            The Image containing the reference data, including the data array, weight, and flags

        Returns
        -------
        dictionary with the following keys:
            outim : np.ndarray
                The difference between the new and reference images
            outwt: np.ndarray
                The weight image for the difference
            outfl: np.ndarray
                The flag image for the difference
        """
        outim = new_image.data - ref_image.data

        # must add the variance to make a new weight image
        new_mask = new_image.weight <= 0
        new_weight = new_image.weight.copy()
        new_weight[new_mask] = np.nan
        new_var = 1 / new_weight ** 2

        ref_mask = ref_image.weight <= 0
        ref_weight = ref_image.weight.copy()
        ref_weight[ref_mask] = np.nan
        ref_var = 1 / ref_weight ** 2

        outwt = 1 / np.sqrt(new_var + ref_var)
        outwt[new_mask] = 0  # make sure to make zero weight the pixels that started out at zero weight

        outfl = new_image.flags.copy()
        outfl |= ref_image.flags

        return dict(outim=outim, outwt=outwt, outfl=outfl)

    def _subtract_zogy(self, new_image, ref_image):
        """Use ZOGY to subtract the two images.

        This applies PSF matching and uses the ZOGY algorithm to subtract the two images.
        reference: https://ui.adsabs.harvard.edu/abs/2016ApJ...830...27Z/abstract

        Parameters
        ----------
        new_image : Image
            The Image containing the new data, including the data array, weight, and flags.
            Image must also have the PSF and ZeroPoint objects loaded.
        ref_image : Image
            The Image containing the reference data, including the data array, weight, and flags
            Image must also have the PSF and ZeroPoint objects loaded.

        Returns
        -------
        dictionary with the following keys:
            outim : np.ndarray
                The difference between the new and reference images
            outwt: np.ndarray
                The weight image for the difference
            outfl: np.ndarray
                The flag image for the difference
            zogy_score: np.ndarray
                The ZOGY score image (the matched-filter result)
            zogy_psf: np.ndarray
                The ZOGY PSF image (the matched-filter PSF)
        """
        raise NotImplementedError('Not implemented ZOGY subtraction yet')

    def _subtract_hotpants(self, new_image, ref_image):
        """Use Hotpants to subtract the two images.

        This applies PSF matching and uses the Hotpants algorithm to subtract the two images.
        reference: ...

        Parameters
        ----------
        new_image : Image
            The Image containing the new data, including the data array, weight, and flags.
            Image must also have the PSF and ZeroPoint objects loaded.
        ref_image : Image
            The Image containing the reference data, including the data array, weight, and flags
            Image must also have the PSF and ZeroPoint objects loaded.

        Returns
        -------
        dictionary with the following keys:
            outim : np.ndarray
                The difference between the new and reference images
            outwt: np.ndarray
                The weight image for the difference
            outfl: np.ndarray
                The flag image for the difference
        """
        raise NotImplementedError('Not implemented Hotpants subtraction yet')

    def run(self, *args, **kwargs):
        """
        Get a reference image and subtract it from the new image.
        Arguments are parsed by the DataStore.parse_args() method.

        Returns a DataStore object with the products of the processing.
        """
        self.has_recalculated = False
        ds, session = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step:
        with SmartSession(session) as session:
            prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

            # look for a reference that has to do with the current image
            ref = ds.get_reference(session=session)
            if ref is None:
                raise ValueError(
                    f'Cannot find a reference image corresponding to the datastore inputs: {ds.get_inputs()}'
                )

            # manually add the upstreams of reference image's products (SourceList, PSF, WCS, ZP)
            upstreams = prov.upstreams
            upstreams.append(ref.sources.provenance)
            upstreams.append(ref.psf.provenance)
            upstreams.append(ref.wcs.provenance)
            upstreams.append(ref.zp.provenance)
            prov.upstreams = upstreams  # must re-assign to make sure list items are unique
            prov.update_id()
            prov = session.merge(prov)
            sub_image = ds.get_subtraction(prov, session=session)

            if sub_image is None:
                self.has_recalculated = True
                # use the latest image in the data store,
                # or load using the provenance given in the
                # data store's upstream_provs, or just use
                # the most recent provenance for "preprocessing"
                image = ds.get_image(session=session)
                if image is None:
                    raise ValueError(f'Cannot find an image corresponding to the datastore inputs: {ds.get_inputs()}')

                sub_image = Image.from_ref_and_new(ref.image, image)
                sub_image.provenance = prov
                sub_image.provenance_id = prov.id

                # make sure to grab the correct aligned images
                new_image = sub_image.aligned_images[sub_image.new_image_index]
                ref_image = sub_image.aligned_images[sub_image.ref_image_index]

                if self.pars.method.lower() == 'naive':
                    outdict = self._subtract_naive(new_image, ref_image)
                elif self.pars.method.lower() == 'hotpants':
                    outdict = self._subtract_hotpants(new_image, ref_image)
                elif self.pars.method.lower() == 'zogy':
                    outdict = self._subtract_zogy(new_image, ref_image)
                else:
                    raise ValueError(f'Unknown subtraction method {self.pars.method}')

                sub_image.data = outdict['outim']
                sub_image.weight = outdict['outwt']
                sub_image.flags = outdict['outfl']
                if 'score' in outdict:
                    sub_image.zogy_score = outdict['zogy_score']
                if 'psf' in outdict:
                    sub_image.zogy_psf = outdict['psf']  # not saved but can be useful for testing / source detection

        ds.sub_image = sub_image

        # make sure this is returned to be used in the next step
        return ds
