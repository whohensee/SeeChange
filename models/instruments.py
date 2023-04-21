# TODO: make a base Instrument class that has all the common methods.
#  Subclass it for each instrument we want to use, e.g., DECam, LS4, etc.
#  Each one of these classes must implement load and save methods.
#  There are other things we may need from an instrument, like aperture, pixel scale, etc.

import numpy as np


class Instrument:

    def load(self, filename, ccd_id):
        """
        Load a part of an exposure file, based on the ccd_id.
        If the instrument does not have multiple CCDs, set ccd_id=0.

        Parameters
        ----------
        filename: str
            The filename of the exposure file.
        ccd_id: int
            The CCD ID to load.

        Returns
        -------
        data: np.ndarray
            The data from the exposure file.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    def read_header(self, filename):
        """
        Load the FITS header from filename.

        Parameters
        ----------
        filename: str
            The filename of the exposure file.

        Returns
        -------
        header: dict
            The header from the exposure file, as a dictionary.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    def get_num_ccds(self):
        """
        Get the number of CCDs in the instrument.

        Returns
        -------
        num_ccds: int
            The number of CCDs in the instrument.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")


class Demo(Instrument):
    def load(self, filename, ccd_id):
        return np.random.poisson(10, (512, 1024))

    def read_header(self, filename):
        return {}

    def get_num_ccds(self):
        return 7


class DECam(Instrument):
    pass

