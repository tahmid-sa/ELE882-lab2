import numpy as np


def apply_lut(img, lut):
    '''Apply a look-up table to an image.

    The look-up table can be be used to quickly adjust the intensities within an
    image.  For colour images, the same LUT can be applied equally to each
    colour channel.

    Parameters
    ----------
    img : numpy.ndarray
        a ``H x W`` greyscale or ``H x W x C`` colour 8bpc image
    lut : numpy.ndarray
        a 256-element, 8-bit array

    Returns
    -------
    numpy.ndarray
        a new ``H x W`` or ``H x W x C`` image derived from applying the LUT

    Raises
    ------
    ValueError
        if the LUT is not 256-elements long
    TypeError
        if either the LUT or images are not 8bpc
    '''
    if lut.size != 256:
        raise ValueError('Invalid LUT Length.')

    if lut.dtype == np.uint8 and img.dtype == np.uint8:
        img_modified = np.array(img.flatten())

        for intensity in range(img.size):
            img_modified[intensity] = lut[img_modified[intensity]]

        if img.ndim == 2:
            height, width = np.shape(img)
            return np.reshape(img_modified, (height, width))
        else:
            height, width, colour = np.shape(img)
            return np.reshape(img_modified, (height, width, colour))
    else:
        raise TypeError('Can only support 8-bit images.')


def adjust_brightness(offset):
    '''Generate a LUT to adjust the image brightness.

    Parameters
    ----------
    offset : int
        the amount to offset brightness values by; this may be negative or
        positive

    Returns
    -------
    numpy.ndarray
        a 256-element LUT that can be provided to ``apply_lut()``
    '''
    lut_brightness = np.arange(256) + offset
    lut_brightness = np.where(lut_brightness > 255, 255, lut_brightness)
    lut_brightness = np.where(lut_brightness < 0, 0, lut_brightness)

    return np.array(lut_brightness, dtype=np.uint8)


def adjust_contrast(scale, hist):
    '''Generate a LUT to adjust contrast without affecting brightness.

    Parameters
    ----------
    scale : float
        the value used to adjust the image contrast; a value greater than 1 will
        increase constrast while a value less than 1 will reduce it
    hist : numpy.ndarray
        a 256-element array containing the image histogram, which is used to
        calculate the image brightness

    Returns
    -------
    numpy.ndarray
        a 256-element LUT that can be provided to ``apply_lut()``

    Raises
    ------
    ValueError
        if the histogram is not 256-elements or if the scale is less than zero
    '''
    if hist.size != 256 or scale < 0:
        raise ValueError('Histogram must be 256-elements long and scale cannot be negative.')

    brightness = int(np.sum([(i * hist[i]) for i in range(256)]) / np.sum(hist))

    lut_contrast = scale * np.arange(256) + (1 - scale) * brightness
    lut_contrast = np.where(lut_contrast > 255, 255, lut_contrast)
    lut_contrast = np.where(lut_contrast < 0, 0, lut_contrast)

    return np.array(lut_contrast, dtype=np.uint8)


def adjust_exposure(gamma):
    '''Generate a LUT that applies a power-law transform to an image.

    Parameters
    ----------
    gamma : float
        the exponent in the power-law transform; must be a positive value

    Returns
    -------
    numpy.ndarray
        a 256-element LUT that can be provided to ``apply_lut()``

    Raises
    ------
    ValueError
        if ``gamma`` is negative
    '''
    if gamma < 0:
        raise ValueError('Gamma must be a positive value.')

    lut_exposure = np.arange(256)
    return np.array(((lut_exposure.astype(float) / 255.0)**gamma)*255, dtype=np.uint8)


def log_transform():
    '''Generate a LUT that applies a log-transform to an image.

    Returns
    -------
    numpy.ndarray
        a 256-element LUT that can be provided to ``apply_lut()``
    '''
    return np.array(106 * np.log10(np.arange(256) + 1), dtype=np.uint8)
