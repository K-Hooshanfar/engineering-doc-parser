class Config:
    """
    Configuration settings for the document orientation detector.

    Attributes:
        EAST_TEXT_DETECTOR (str): Filesystem path to the pretrained EAST text detection model.
    """
    EAST_TEXT_DETECTOR: str = 'weights/frozen_east_text_detection.pb'
