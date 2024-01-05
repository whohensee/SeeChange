from models.instrument import Instrument


class PTF(Instrument):

    def __init__(self, **kwargs):
        # TODO: continue this!
        # will apply kwargs to attributes, and register instrument in the INSTRUMENT_INSTANCE_CACHE
        Instrument.__init__(self, **kwargs)

    @classmethod
    def get_section_ids(cls):

        """
        Get a list of SensorSection identifiers for this instrument.
        Includes all 12 CCDs.
        """
        return range(0, 12)

    @classmethod
    def check_section_id(cls, section_id):
        """
        Check that the type and value of the section is compatible with the instrument.
        In this case, it must be an integer in the range [0, 11].
        """
        try:
            section_id = int(section_id)
        except ValueError:
            raise ValueError(
                f"The section_id must be an integer or a string convertible to an integer. Got {section_id}. "
            )

        if not 0 <= section_id <= 11:
            raise ValueError(f"The section_id must be in the range [0, 11]. Got {section_id}. ")

    @classmethod
    def get_filename_regex(cls):
        return [r'PTF.*\.fits']

    @classmethod
    def get_short_instrument_name(cls):
        """
        Get a short name used for e.g., making filenames.
        """
        return 'PTF'

    # TODO: continue this!
