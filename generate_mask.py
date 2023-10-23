import cv2
import numpy as np
import rasterio
import sys

class GenerateMask:

    def __init__(self, input_image_path, output_mask_path="mask.tif"):
        self.input_image_path = input_image_path
        self.output_mask_path = output_mask_path
        self.roi = []
        self.scale_factor = 0.35
        self.input_image = None
        self.input_image_meta = None

    def draw_polygon(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x = max(0, min(x, param.shape[1] - 1))
            y = max(0, min(y, param.shape[0] - 1))
            self.roi.append((x, y))
            cv2.circle(param, (x, y), 3, (0, 0, 255), -1)
            if len(self.roi) > 1:
                cv2.line(param, self.roi[-1], self.roi[-2], (0, 255, 0), 2)
            cv2.imshow("Draw ROI", param)

    def load_image(self):
        with rasterio.open(self.input_image_path) as src:
            self.input_image_meta = src.meta.copy()
            self.input_image = src.read([1, 2, 3])
            self.input_image = cv2.normalize(self.input_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.input_image = np.transpose(self.input_image, [1, 2, 0])

    def generate_mask(self):
        self.load_image()
        small = cv2.resize(self.input_image, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        cv2.namedWindow("Draw ROI")
        cv2.setMouseCallback("Draw ROI", self.draw_polygon, small)

        while True:
            cv2.imshow("Draw ROI", small)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        roi_actual = [(int(pt[0] / self.scale_factor), int(pt[1] / self.scale_factor)) for pt in self.roi]
        mask = np.zeros(self.input_image.shape[:2], dtype=np.uint8)

        if len(roi_actual) > 2:
            roi_array = np.array([roi_actual], np.int32)
            cv2.fillPoly(mask, roi_array, 255)
        else:
            print("ROI was not set properly.")

        transposed_mask = np.expand_dims(mask, axis=0)
        self.input_image_meta['count'] = 1

        with rasterio.open(self.output_mask_path, 'w', **self.input_image_meta) as dest:
            dest.write(transposed_mask.astype(np.uint8))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_mask.py <input_image>")
        sys.exit(1)
    mask_gen = GenerateMask(sys.argv[1])
    mask_gen.generate_mask()
