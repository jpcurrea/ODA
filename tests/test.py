from analysis_tools import *

# test out Layer object using a stored image file
img = Layer("./002.jpg")
img.load()
# test by importing directly from a numpy array
img_arr = np.copy(img.image)
img = Layer(arr=img_arr)
img.load()
img.get_gradient()
img.color_key()

# test out the Eye object using the same eye image
PIXEL_SIZE = (488.84896*2)**-1  # mm
eye = Eye("./002.jpg", mask_fn="mask.jpg", pixel_size=PIXEL_SIZE)
eye.get_eye_outline()
eye.get_eye_dimensions(display=False)
eye.crop_eye(use_ellipse_fit=False)
eye.crop_eye(use_ellipse_fit=True)
# use the cropped image
cropped_eye = eye.eye
# run the ommatidia_detecting_algorithm
cropped_eye.oda(bright_peak=False, min_count=100, max_count=1500,
                high_pass=True, plot=True, plot_fn='./002_ommatidia.svg')
 
