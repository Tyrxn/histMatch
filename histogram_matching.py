from __future__ import print_function
import sys
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import rcParams
from generate_mask import GenerateMask

def calculate_cdf(histogram): # Cumulative distribution function
    cdf = histogram.cumsum()
    cdf_max = cdf.max()
    if cdf_max == 0:
        return np.zeros_like(cdf)
    normalized_cdf = cdf / float(cdf_max)
    return normalized_cdf

def calculate_lookup(src_cdf, ref_cdf):
    lookup_table = np.zeros(256)
    g_j = 0

    for g_i in range(256):
        while g_j < 256 and ref_cdf[g_j] <= src_cdf[g_i]:
            g_j += 1
        lookup_table[g_i] = g_j
    return lookup_table

def white_out_non_roi(img, mask):
    img_out = img.copy()
    img_out[mask == 255] = [255, 255, 255]
    return img_out



def make_plot(src_image, ref_image, output_image, edge_mask):
    plt.style.use('ggplot')  # Use ggplot style for a change
    rcParams['figure.figsize'] = 18, 16

    f, axes = plt.subplots(3, 2)
    image_axes = [axes[0, 0], axes[1, 0], axes[2, 0]]
    for ax in image_axes:
        ax.axis('off')
        ax.grid(False)

    # Ensure data types are uint8 and 3 channels for OpenCV processing and visualization
    src_image = src_image.astype(np.uint8)[:, :, :3]
    ref_image = ref_image.astype(np.uint8)[:, :, :3]
    output_image = output_image.astype(np.uint8)[:, :, :3]

    # Showing images
    axes[0, 0].imshow(cv2.cvtColor(white_out_non_roi(src_image, edge_mask), cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Source Image', fontsize=15)

    # Check if ref_image is already white outside the ROI
    if np.all(ref_image[edge_mask == 0] == 0):
        axes[1, 0].imshow(cv2.cvtColor(white_out_non_roi(ref_image, edge_mask), cv2.COLOR_BGR2RGB))
    else:
        axes[1, 0].imshow(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Reference Image', fontsize=15)

    axes[2, 0].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('Histogram Matched Output', fontsize=15)

    # RGB histograms
    imgs = (src_image, ref_image, output_image)
    titles = ('Source', 'Reference', 'Output')
    bins = 32
    color_palette = {"red": "#E63946", "green": "#0ac400", "blue": "#0394fc"}  # Define a color palette

    for i, axis in enumerate(axes[:, 1]):
        im = imgs[i]
        title = titles[i]

        # Adjust the mask condition for the reference image
        mask_condition = edge_mask == 255 if title == "Reference" else edge_mask == 0

        red, _ = np.histogram(im[mask_condition, 0].flatten(), bins, [0, 256])
        green, _ = np.histogram(im[mask_condition, 1].flatten(), bins, [0, 256])
        blue, _ = np.histogram(im[mask_condition, 2].flatten(), bins, [0, 256])

        # Use the count of pixels in the region of interest for normalization
        roi_pixel_count = np.sum(mask_condition)

        for color, name in ((red, "red"), (green, "green"), (blue, "blue")):
            norm = color / roi_pixel_count
            axis.fill_between([float(x) * 256 / bins for x in range(bins)],
                              norm, facecolor=color_palette[name], alpha=0.6)
        axis.set_title("{} RGB histogram".format(title), fontsize=14)
        axis.set_xlabel("Pixel Value", fontsize=13)
        axis.set_ylabel("Relative Frequency", fontsize=13)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between subplots
    image_src_name = sys.argv[1]
    plt.savefig(image_src_name.split('.')[0] + "_plot.png", bbox_inches='tight')
    plt.show()





def match_histograms(src_image, mask=None):
    
    # Create an empty array to store the output image
    output_image = src_image.copy()



    for color_idx, color_name in enumerate(('blue', 'green', 'red')):
        src_channel = src_image[:, :, color_idx]
        
        # Split the source and reference parts
        src_channel_masked_1 = src_channel[mask == 0]  # Source region (to be adjusted)
        src_channel_masked_0 = src_channel[mask == 255]  # Reference region (to remain unchanged)

        # Calculate histograms for both regions
        hist_masked_1, _ = np.histogram(src_channel_masked_1.flatten(), 256, [0, 256])
        hist_masked_0, _ = np.histogram(src_channel_masked_0.flatten(), 256, [0, 256])

        # Calculate CDFs
        cdf_masked_1 = calculate_cdf(hist_masked_1) # Source CDF
        cdf_masked_0 = calculate_cdf(hist_masked_0) # Reference CDF

        # Generate the lookup table
        lookup_table = calculate_lookup(cdf_masked_1, cdf_masked_0)

        # Apply the lookup table only to the source region
        updated_channel = cv2.LUT(src_channel, lookup_table)
        output_image[:, :, color_idx] = np.where(mask == 0, updated_channel, src_channel)


    output_image = cv2.convertScaleAbs(output_image)
    
    return output_image



def main():
    if len(sys.argv) < 2:
        print("Usage: python histogram_matching.py <source_image>")
        return

    image_src_name = sys.argv[1]

    # Use the GenerateMask class to create a mask
    mask_gen = GenerateMask(image_src_name)
    mask_gen.generate_mask()
    mask_name = mask_gen.output_mask_path  # Getting the mask path after generating
    edge_mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)  # Reading the generated mask as grayscale

    if edge_mask is None:  # Handle potential errors
        print("Error generating or reading mask.")
        return

    try:
        with rasterio.open(image_src_name) as src:
            num_bands = src.count
            if num_bands >= 3:
                src_image = src.read([1, 2, 3])  # Reading only the first three bands
            else:
                print("Error: Source image does not have enough bands (needs at least 3).")
                return
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # Transpose to meet OpenCV's expectation for shape
    src_image = np.transpose(src_image, [1, 2, 0])

    # Split source and reference from the original image using the mask
    original_source = np.where(edge_mask[:, :, np.newaxis] == 0, src_image, 0)
    original_ref = np.where(edge_mask[:, :, np.newaxis] == 255, src_image, [255, 255, 255])

    # Apply histogram matching
    output_image = match_histograms(src_image, edge_mask)

    # Transpose back to meet rasterio's expectation
    transposed_output = np.transpose(output_image, [2, 0, 1])

    # Save the adjusted image
    output_image_name = image_src_name.split('.')[0] + "_balanced.tif"
    with rasterio.open(image_src_name) as src:
        metadata = src.meta.copy()
        metadata['count'] = 3  # Ensure we're writing only 3 bands

    with rasterio.open(output_image_name, 'w', **metadata) as dest:
        dest.write(transposed_output.astype(np.uint8))

    # Display the results
    make_plot(original_source, original_ref, output_image, edge_mask)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()

