from analysis_tools import *

PIXEL_SIZE = (488.84896*2)**-1
DEPTH_SIZE = .004*3            # from the manual; we used to use .0042
# import an image stack
# stack = Stack(dirname="002", f_type='.JPG',
#               pixel_size=PIXEL_SIZE, depth_size=DEPTH_SIZE)
# stack.load()
# stack.get_focus_stack()
# plt.imshow(stack.stack)
# plt.show()
# import an eyestack
eye_stack = EyeStack(dirname="002", img_extension='.JPG',
                     pixel_size=PIXEL_SIZE, depth_size=DEPTH_SIZE)
eye_stack.oda_3d(bright_peak=False, plot=True)

