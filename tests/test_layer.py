sys.path.append("g:/.shortcut-targets-by-id/1as1j06Ce9KOwHXHd8ujnPqBpqZGP503z/backup/Desktop/programs/eye_tools/src/eye_tools")
from analysis_tools import *



# test out Layer object using a stored image file
img = Layer("./002.jpg")
img.load_memmap()
# test by importing directly from a numpy array
img_arr = np.copy(img.image)
img = Layer(arr=img_arr)
img.load_memmap()
img.get_gradient()
img.color_key()
# mask_img = img.mask.astype('uint8')
# mask_img = np.repeat(mask_img[..., np.newaxis], 3, axis=-1)
# mask_img *= 255 
# save_image("mask.jpg", 255 * mask_img)

# test out the Eye object using the same eye image
PIXEL_SIZE = (488.84896*2)**-1  # mm
eye = Eye("./002.jpg", mask_fn="./002/mask.jpg", pixel_size=PIXEL_SIZE)
eye.load_memmap()
eye.get_eye_outline()
eye.get_eye_dimensions(display=False)
eye.crop_eye(use_ellipse_fit=False)
eye.crop_eye(use_ellipse_fit=True)
# use the cropped image
cropped_eye = eye.eye
# run the ommatidia_detecting_algorithm
cropped_eye.oda(bright_peak=False, high_pass=True, plot=True,
                plot_fn='./002_ommatidia.svg')
 
