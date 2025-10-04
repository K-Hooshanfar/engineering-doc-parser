"""Configuration for the rotation package.

Defines paths and constants used by the document orientation detector,
including the default EAST text detector model path.
"""


class Config: # pylint: disable=too-few-public-methods
    """
    Configuration settings for the document orientation detector.

    Attributes:
        EAST_TEXT_DETECTOR (str): Filesystem path to the pretrained EAST text detection model.
    """

    EAST_TEXT_DETECTOR: str = "weights/frozen_east_text_detection.pb"
