import argparse
import numpy as np

from skimage import io
from matplotlib import pyplot as plt

from assignment.analysis import histogram
from assignment.point_operators import apply_lut

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str)
    parser.add_argument('output_image', type=str)
    return parser.parse_args()

def read_image(input_image, output_image):

    # Read the image file
    img = io.imread(input_image, as_gray=True)
    img = img.astype(np.uint8)

    # # Compute the histogram of the image file and plot it
    hist = histogram(img)
    plt.figure()
    plt.plot([i for i in range(256)], hist)
    plt.title('Histogram of input image')

    # Perform the histogram equalization
    pixels = 0
    for intensity in range(256):
        pixels = pixels + hist[intensity]
    
    pmf = hist / pixels
    cdf = np.cumsum(pmf)

    lut = cdf * 255
    lut = np.where(lut > 255, 255, lut)
    lut = np.where(lut < 0, 0, lut)
    lut = np.array(lut, dtype=np.uint8)

    new_img = apply_lut(img, lut)
    
    hist = histogram(new_img)
    plt.figure()
    plt.plot(hist)
    plt.title('Histogram of output image')
    
    # Display and save the equalized image
    plt.figure()
    plt.imshow(new_img, cmap=plt.get_cmap("gray"))
    plt.title('Equalized image')
    io.imsave(output_image, new_img) 

    plt.show()


def main():

    args = get_args()

    print(f'Input image filename: {args.input_image}')
    input_image = f'{args.input_image}'

    print(f'Output image filename: {args.output_image}')
    output_image = f'{args.output_image}'

    read_image(input_image, output_image)

if __name__ == "__main__":
    main()
