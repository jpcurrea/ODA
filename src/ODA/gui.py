"""Here is the start of the full GUI version of this software.

TODO:
1) make a GUI replacing the plot_data_3d using a Volume instead of scatterplots
2) make 3 different GUIs for running each of the 3 main classes:
    - Eye: file(s) selector, integrated color selection with slider for working on multiple images, and manual selection GUI
    - EyeStack: folder(s) selector, [Eye GUI], 3 plotter
    - CTStack: folder selector, 

"""
from collections import OrderedDict
from datetime import datetime
import sys
sys.path.append("h:\\My Drive\\backup\\Desktop\\programs\\ODA\\ODA\\src\\ODA\\")
from analysis_tools import *
# import widgets for a GUI
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSlider,
    QCheckBox,
    QComboBox,
    )
from functools import partial

class Volume():
    def __init__(self, dirname, center=(0, 0, 0), pixel_length=1, 
                 depth_length=1, volume=None, scale=None, alpha=20,
                 vmin=0, vmax=255, remappable=True, gui_depth=0):
        """Control the display an arbitrary volume.
        
        Parameters
        ----------
        dirname : str
            The directory containing the stack.
        center : array-like, len=3
            Specifies the 3D center point of the volume for rotations.
        pixel_length, depth_length : float, default=1
            The length of the side of each pixel or the interval between stack layers.
        volume : np.ndarray, default=None
            Optionally, you can provide the numpy array used as the 3D volume.
        scale : int, default=None
            Optionally, you can specify the scale or bin length you want to use for 
            bin averaging.
        alpha : int or float, default=20
            The default opacity of the volume.
        vmin, vmax : float, default=0, 255
            The minimum and maximum values to include in the volume.
        remappable : bool, default=True
            Whether to allow selecting the colormap range. False prevents the colormap
            from updating.
        """
        # store important parameters
        self.dirname = dirname
        self.center = np.asarray(center)
        self.pixel_length = pixel_length
        self.depth_length = depth_length
        self.volume = volume
        self.gui_depth = gui_depth
        self.scale = scale
        self.alpha = alpha
        self.vmin = vmin
        self.vmax = vmax
        self.remappable = remappable
        # store an h5 file in this directory for easier access to the full volume
        self.h5_filename = os.path.join(self.dirname, '_volume_data.h5')
        if not os.path.exists(self.h5_filename):
            self.database = h5py.File(self.h5_filename, 'w')
            del self.database            
        self.database = h5py.File(self.h5_filename, 'r+')

    def load(self, **stack_kwargs):
        """Load the dataset into an h5 file to offload RAM."""
        # load the raw data into a 4D h5 dataset (x, y, z, RGBA)
        stack = Stack(dirname=self.dirname, **stack_kwargs)
        # go through each layer and load the pixel values into 
        first_layer = Layer(stack.fns[0])
        img = first_layer.load()
        if img.ndim == 2:
            height, width = img.shape
            num_channels = 1
        elif img.ndimg == 3:
            height, width, num_channels = img.shape
        num_layers = len(stack.fns)
        if num_channels > 1:
            shape = (num_layers, height, width, num_channels)
        else:
            shape = (num_layers, height, width)        
        loaded = False
        if 'original_volume' in self.database.keys():
            resp = input("This volume has been loaded previously. Enter <1> to use the pre-loaded volume or <enter> to import from scratch: ")
            loaded = resp == '1'
            if not loaded:
                del self.database['original_volume']
        if not loaded:
            if num_channels > 1:
                self.original_volume = self.database.create_dataset('original_volume', shape, chunks=(1, height, width, num_channels), dtype=np.uint8)
            else:
                self.original_volume = self.database.create_dataset('original_volume', shape, chunks=(1, height, width), dtype=np.uint8)
            # use chunking to speed up reading and writing to the dataset
            for layer_num, (layer, chunk) in enumerate(zip(stack.iter_layers(), self.original_volume.iter_chunks())):
                img = layer.load()
                if img.dtype == np.uint16:
                    img = (img // (2 ** 8)).astype('uint8')
                # store in the corresponding layer of the dataset
                self.original_volume[chunk] = img[np.newaxis]
                print_progress(layer_num, len(stack.fns))

    def bin_average(self, var='original_volume', bin_length=1):
        """Generate a numpy array from the original volume by using cube bins."""
        original_arr = self.database[var]
        if original_arr.ndim == 3:
            depth, height, width = original_arr.shape
            num_channels = 1
        if original_arr.ndim == 4:
            depth, height, width, num_channels = original_arr.shape
        # calculate the new dimensions based on the bin_length
        num_layers = math.ceil(depth / bin_length)
        new_height, new_width = int(height / bin_length), int(width / bin_length)
        # volume = np.zeros((num_layers, new_height, new_width, num_channels), dtype=img.dtype)
        new_shape = (num_layers + 1, new_height, new_width)
        # check if the bin averaged volume has already been stored
        loaded = False
        result_var = f"{var}_scale={bin_length}"
        if result_var in self.database.keys():
            resp = input("This volume was stored previously at the same bin size. Enter <1> to use that dataset or <enter> to bin average from scratch: ")
            loaded = resp == '1'
            if loaded:
                self.volume = self.database[result_var]
                self.scale = bin_length
            else:
                del self.database[result_var]
        if not loaded:
            if num_channels > 1:
                binned_volume = self.database.create_dataset(result_var, (num_layers+1, new_height, new_width, num_channels), 
                                                             chunks=(1, new_height, new_width, num_channels), dtype=np.uint8)
            else:
                binned_volume = self.database.create_dataset(result_var, (num_layers+1, new_height, new_width), 
                                                             chunks=(1, new_height, new_width), dtype=np.uint8)
            # bin average the layers
            layer_num = 0
            height_trunc, width_trunc = bin_length * (height // bin_length), bin_length * (width // bin_length)
            img_avg = []
            for num, chunk_og in enumerate(original_arr.iter_chunks()):
                img = original_arr[chunk_og][0]
                if img.dtype == np.uint16:
                    img = (img // (2 ** 8)).astype('uint8')
                img_avg += [img]
                if len(img_avg) == bin_length or num == len(original_arr)-1:
                    num_sub_layers = len(img_avg)
                    img_avg = np.asarray(img_avg)
                    # bin average the image
                    img_avg = img_avg[:, :height_trunc, :width_trunc]
                    img_avg = img_avg.reshape((num_sub_layers, height_trunc // bin_length, bin_length, width_trunc // bin_length, bin_length))
                    img_avg = img_avg.mean((0, 2, 4))
                    # set the average in the volume
                    if num_channels > 1:
                        binned_volume[num // bin_length, ..., 0] = img_avg
                    else:
                        binned_volume[num // bin_length] = img_avg
                    layer_num += 1
                    # reset img_avg
                    img_avg = []
                print_progress(num, len(original_arr))
        self.scale = bin_length

    def volume_to_rgb(self, key='original_volume', cmap='gray', vmin=0, vmax=255, alpha=20):
        """Convert a full volume dataset into RGBA voxel values.
        
        We use a system of variable names to store and retrieve specific rgb volumes.
        Each volume has a name. Every time we update the colormap settings, we have
        to first update the full res. volume, stored in the database, and then bin
        average the color volume to get the right results. So first load the volume, 
        convert it to RGBA values and store as a dataset. Then, to plot, we update the 
        color volume and then bin average. 

        Notes:
        -rgb versions of the volume are stored under the name {self.volume_name}_rgb.
        -the outcome will generate the full 0 to 255 color range unless you choose the 
        specific minimum and maximum of the colormap.

        Parameters
        ----------
        key : str, default='original_volume'
            The volume to use for generating the full resolution color volume.
        cmap : str, default='gray'
            The name of the colormap to use based on the list of pyplot colormaps.
        """
        # set the default volume
        self.set_display_volume(key)
        # todo: make a function to generate an RGBA volume as a stored h5 dataset using self.volume
        saved = False
        rgb_name = f"{key}_rgb"
        if rgb_name in self.database.keys():
            self.rgb = self.database[rgb_name]
            resp = input("This volume has been converted to color already. Enter <1> to use the pre-loaded color volume or <enter> to import from scratch: ")
            saved = resp == '1'
        else:
            depth, height, width = self.volume.shape[:3]
            self.rgb = self.database.create_dataset(rgb_name, (depth, height, width, 4), dtype='uint8', chunks=(1, height, width, 4))
        if not saved:
            # graph the specific cmap
            cmap = plt.get_cmap(cmap)
            # go through the full and color volumes one layer at a time to avoid loading everything to RAM
            for layer_num, (chunk_vol, chunk_rgb) in enumerate(zip(self.volume.iter_chunks(), self.rgb.iter_chunks())):
                # reduce the volume along the first dimension by averaging
                alpha_vals = np.copy(self.volume[chunk_vol])
                # todo: make a temporary rgba array with the appropriate values and then store all at the same time
                colorvals = (alpha_vals - vmin) / (vmax - vmin)
                rgb = (255 * cmap(colorvals)).astype('uint8')
                # set the alpha of all non-zero values in the volume to 0
                non_zero = alpha_vals > 0
                alpha_vals[non_zero] = alpha
                rgb[..., -1] = alpha_vals
                self.rgb[chunk_rgb] = rgb
                print_progress(layer_num, self.rgb.shape[0])

    def control_panel(self, title, layout, preview_button=True, alpha=20):
        """Plot controls for displaying the 3D volume.
        
        Parameters
        ----------
        title : str
            The title for this control panel.
        layout : QHBoxLayout or QVBoxLayout
            The qt layout to add the panel to.
        preview_button : bool, default=True
        """
        self.control_layout = layout
        # make a control panel for the visibility, opacity, and colormapping
        self.rows = QVBoxLayout()
        ## first row is just text
        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignBottom)
        self.row0 = QHBoxLayout()
        self.row0.addWidget(self.label)
        self.rows.addLayout(self.row0)
        # make a checkbox for displaying the 3D volume
        self.row1 = QHBoxLayout()
        self.visible_check = QCheckBox("show")
        self.visible_check.setChecked(True)
        self.visible_check.stateChanged.connect(partial(self.toggle, var='gui'))
        self.row1.addWidget(self.visible_check)
        # make a second checkbox for previewing the 3D volume in ImageView
        self.preview_check = QCheckBox("2D preview")
        self.preview_check.setChecked(False)
        self.preview_check.stateChanged.connect(
            partial(self.toggle, var='preview'))
        self.row1.addWidget(self.preview_check)
        self.rows.addLayout(self.row1)
        ## and a button to update the colormap of the volume_raw_rgb
        self.row2 = QHBoxLayout()
        update_button = QPushButton("update")
        # update_button.released.connect(self.update_raw_volume)
        update_button.released.connect(self.update_plot)
        self.row2.addWidget(update_button)
        self.rows.addLayout(self.row2)
        ## add a second row with a slider to specify the alpha value of the volume
        self.row3 = QHBoxLayout()
        label_alpha = QLabel("opacity: ")
        self.row3.addWidget(label_alpha)
        # make a slider for setting the alpha channel
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.setValue(alpha)
        self.row3.addWidget(self.alpha_slider)
        self.rows.addLayout(self.row3)
        ## add a second row with a slider to specify the alpha value of the volume
        self.row4 = QHBoxLayout()
        label = QLabel("color scheme: ")
        self.row4.addWidget(label)
        # make a dropdown menu for choosing the color scheme
        self.color_dropdown = QComboBox()
        for key in self.database.keys():
            arr = self.database[key]
            if 'rgb' in key and arr.ndim == 4:
                self.color_dropdown.addItem(key)
        # find the index of the current volume name
        items = np.array([self.color_dropdown.itemText(i) for i in range(self.color_dropdown.count())])
        ind = np.where(items == self.rgb_name)[0][0]
        self.color_dropdown.setCurrentIndex(ind)
        self.row4.addWidget(self.color_dropdown)
        self.rows.addLayout(self.row4)
        # add the rows to the control panel layout
        self.control_layout.addLayout(self.rows)

    def set_display_volume(self, key='original_volume'):
        """Set the volume for displaying."""
        self.volume = self.database[key]
        self.volume_name = key
        rgb_name = f"{key}_rgb"
        self.rgb_name = rgb_name
        if self.rgb_name in self.database.keys():
            self.rgb = self.database[self.rgb_name]
        if 'scale' in self.rgb_name:
            self.scale = int(self.rgb_name.split("scale=")[-1].split("_")[0])
        else:
            self.scale = 1
        # change the dropdown value
        if 'color_dropdown' in dir(self):
            items = np.array([self.color_dropdown.itemText(i) for i in range(self.color_dropdown.count())])
            ind = np.where(items == self.rgb_name)[0][0]
            self.color_dropdown.setCurrentIndex(ind)

    def update_rgb(self, cmap='gray', alpha=20, vmin=0, vmax=255):
        """Generate the 3D volume of RGBA values for the whole stack.
        
        Parameters
        ----------
        cmap : str, default='gray'
            The matplotlib colormap string for converting the stack's density measurements
            into RGB values 
        alpha : int, default=20
            The alpha value to use for all non-zero elements of the volume.
        vmin, vmax : float, default=0, 255
            The minimum and maximum values for the colormap.
        """
        # assume that there's already an rgb array
        assert 'rgb' in dir(self), f"Can't find a color array. Try running {self.volume_to_rgb} first."
        # store the current settings
        alpha_changed, cmap_changed = False, False
        if alpha != self.alpha:
            alpha_changed = True
            self.alpha = alpha
        if vmin != self.vmin or vmax != self.vmax:
            cmap_changed = True
            self.vmin = vmin
            self.vmax = vmax
        # if alpha_changed or (cmap_changed and self.remappable):
        #     alpha_vals = np.copy(self.volume[:])
        # todo: go through the database by layers instead of all at once
        # graph the specific cmap
        self.cmap_raw = plt.get_cmap(cmap)
        for chunk_vol, chunk_rgb in zip(self.volume.iter_chunks(), self.rgb.iter_chunks()):
            alpha_vals = self.volume[chunk_vol]
            rgb_vals = self.rgb[chunk_rgb]
            if cmap_changed and self.remappable:
                # reduce the volume along the first dimension by averaging
                colorvals = (alpha_vals - vmin) / (vmax - vmin)
                rgb_vals[:] = (255 * self.cmap_raw(colorvals)).astype('uint8')
            if alpha_changed or (cmap_changed and self.remappable):
                # set the alpha of all non-zero values in the volume to 0
                non_zero = alpha_vals > 0
                rgb_vals[non_zero][..., -1] = self.alpha
                self.rgb[chunk_rgb] = rgb_vals
            # report changes 
        print(f"updated colormap:\nvmin={self.vmin}\tvmax={self.vmax}\talpha={self.alpha}")

    def plot(self, window, **update_kwargs):
        """Plot the volume using pyqtgraph."""
        self.window = window
        if 'preview' not in dir(self):
            self.plot_preview()
        if 'rgb' not in dir(self):
            self.update_rgb(**update_kwargs)
        # draw a 3D GLVolumeItem
        print(f"alpha = {self.alpha_slider.value()}")
        rgb = np.copy(self.rgb[:])
        self.gui = gl.GLVolumeItem(rgb, sliceDensity=1)
        self.gui.setGLOptions('translucent')
        self.gui.setDepthValue(self.gui_depth)
        self.gui.scale(self.depth_length * self.scale, self.pixel_length * self.scale, self.pixel_length * self.scale)
        # now center using the center point from the fitted spherical model
        dz, dy, dx = self.center
        self.gui.translate(-dz, -dy, -dx)
        window.addItem(self.gui)
        self.gui_hidden = False

    # make a function to update the raw_scan alpha channel based on a slider 
    def update_plot(self):
        """Update the plotted volume using the alpha and cmap sliders."""
        # check if the color scheme has changed
        volume_name_new = self.color_dropdown.currentText()
        if volume_name_new != self.rgb_name and self.remappable:
            self.volume_to_rgb(key=volume_name_new.replace("_rgb", ""))
        # grab the minimum, maximum, and alpha values from the corresponding sliders
        vmin, vmax = self.preview.getLevels()
        alpha = self.alpha_slider.value()
        # delete the current plot of the rgb volume
        if self.gui in self.window.items:
            self.window.removeItem(self.gui)
            del self.gui
        # update the volume of colors
        self.update_rgb(vmin=vmin, vmax=vmax, alpha=alpha)
        self.plot(window=self.window)
        # initialize 
        self.window.initializeGL()
        self.window.update()

    def toggle(self, var='gui'):
        """Toggle to display/hide the volume gui or other gui objects."""
        gui = self.__getattribute__(var)
        hidden = self.__getattribute__(f"{var}_hidden")
        if hidden:
            gui.show()
            hidden = False
        else:
            gui.hide()
            hidden = True
        self.__setattr__(f"{var}_hidden", hidden)

    def plot_preview(self, ):
        """Use ImageView to preview the stack in slices."""
        self.preview = pg.ImageView()
        self.preview.setImage(self.volume[:], levels=(1, np.iinfo(self.volume.dtype).max))
        self.preview_hidden = True


# todo: make a GUI replacing the plot_data_3d using a Volume instead of scatterplots
class VolumeGUI():
    def __init__(self, raw_stack_folder=None, bin_length=2, raw_bin_length=8, zoffset=0,
                 **CTStack_kwargs):
        """A GUI for plotting the 3D data as a volume.
        
        Parameters
        ----------
        raw_stack_folder : path, default=None
            The path of the directory with the raw stack before the initial filter.
        bin_length : int, default=4
            The length of cube bins used for bin-averaging the volume and re-scaling
            the account for this.
        zoffset : int, default=0
            The offset of additional layers applied to the z-stack to account for 
            omitted layers. 
        **CTStack_kwargs
            Keywards for instantiating the CTStack.
        """
        # load the CTStack dataset
        self.stack = CTStack(**CTStack_kwargs)
        self.stack.load_database()
        # update the eye center based on the provided zoffset
        center = self.stack.database.attrs['center']
        raw_center = np.copy(center)
        self.zoffset = zoffset
        raw_center[0] += zoffset * self.stack.depth_size
        # make two volumes: 1) the full stack if available and 2) the prefiltered stack
        # folders = [self.stack.dirname, raw_stack_folder]
        folders = [raw_stack_folder, self.stack.dirname]
        self.volumes = {}
        # import the raw stack, if provided
        self.pixel_length = self.stack.pixel_size
        self.img_extension = self.stack.img_extension
        self.volume_raw = Volume(
            raw_stack_folder, center=raw_center, pixel_length=self.pixel_length, 
            depth_length=self.pixel_length, gui_depth=2, alpha=20)
        self.volumes['Raw Volume'] = self.volume_raw
        # load the full stack
        self.volume_raw.load(img_extension=self.img_extension)
        self.volume_prefiltered = Volume(self.stack.dirname, center, 
                                         pixel_length=self.pixel_length, depth_length=self.pixel_length,
                                         remappable=False, gui_depth=1, alpha=80)
        self.volumes['Filtered Volume'] = self.volume_prefiltered
        self.volume_prefiltered.load(img_extension=self.img_extension)
        self.volume_raw.database['original_volume'].shape
        # todo: remove voxels from the raw scan that are in the filtered stack
        # after accounting for the z-offset, go through each layer of the raw volume
        # removing any non-zero coordinates from the filtered volume
        volume_raw, volume_filtered = self.volume_raw.database['original_volume'], self.volume_prefiltered.database['original_volume']
        for chunk_raw, chunk_filtered in zip(volume_raw.iter_chunks(), volume_filtered.iter_chunks()):
            layer_raw = volume_raw[chunk_raw]
            layer_filtered = volume_filtered[chunk_filtered]
            # find non-zero voxels in the filtered
            non_zero = layer_filtered > 0
            layer_raw[non_zero] *= 0
            volume_raw[chunk_raw] = layer_raw
        self.volume_raw.bin_average(bin_length=raw_bin_length)
        key = f"original_volume_scale={raw_bin_length}"
        # for density volumes, we can use the binned data to generate the color volume
        self.volume_raw.volume_to_rgb(key=key)
        # note: for the label volume or the ommatidial data volume, we will change the non-zero
        # values of the array to the appropriate values, convert to color, and then bin average
        # import the prefiltered volume
        self.volume_prefiltered.bin_average(bin_length=bin_length)
        key = f"original_volume_scale={bin_length}"
        self.volume_prefiltered.volume_to_rgb(key=key)
        # todo: import the labels volume
        coords = (self.stack.points_original[:] / self.pixel_length).astype(int)
        labels = self.stack.labels[:]
        # copy the prefiltered volume but replace coords with label values
        frame_shape = list(self.volume_prefiltered.database['original_volume'].shape)
        frame_shape[0] = 1
        lbls = ['labels', 'labels_rgb']
        loaded = all([lbl in self.volume_prefiltered.database.keys() for lbl in lbls])
        for lbl in ['labels', 'labels_rgb']:
            if lbl in self.volume_prefiltered.database.keys():
                if lbl not in dir(self):
                    self.__setattr__(lbl, self.volume_prefiltered.database[lbl])
        if loaded:
            resp = input("The labels were loaded previously. Enter <1> to use the pre-loaded volume or <enter> to import from scratch: ")
            loaded = resp == '1'
        if not loaded:
            for lbl in lbls:
                if lbl in dir(self):
                    self.__delattr__(lbl)
                if lbl in self.volume_prefiltered.database.keys():
                    del self.volume_prefiltered.database[lbl]
            self.labels = self.volume_prefiltered.database.create_dataset("labels", self.volume_prefiltered.database['original_volume'].shape, dtype=int, chunks=tuple(frame_shape))
            frame_shape = frame_shape + [4]
            num_layers, height, width = self.volume_prefiltered.database['original_volume'].shape
            self.labels_rgb = self.volume_prefiltered.database.create_dataset("labels_rgb", (num_layers, height, width, 4), dtype=np.uint8, chunks=tuple(frame_shape))
            # go through the labels volume layer by layer, checking which coords are in there and replacing them
            # then, generate the rgba volume by creating a randomized 
            label_max = self.stack.ommatidial_data.label.max()
            labels_conv = np.arange(label_max + 1)
            np.random.shuffle(labels_conv)
            # make a colormap for converting label numbers to random colors
            cmap = partial(linear_cmap, vmin=1, vmax=label_max)
            frame_shape[-1] = 1
            frame = np.zeros(frame_shape[:3], dtype=int)
            frame_shape[-1] = 4
            frame_rgb = np.zeros(frame_shape, dtype=np.uint8)
            for layer_num, (chunk, chunk_rgb) in enumerate(zip(self.labels.iter_chunks(), self.labels_rgb.iter_chunks())):
                in_layer = coords[:, 0] == layer_num
                if np.any(in_layer):
                    coords_in_layer = coords[in_layer]
                    labels_in_layer = labels_conv[labels[in_layer]]
                    # convert to RGBA array
                    # store the labels of this layer
                    new_lbls = labels_conv[labels_in_layer]
                    cvals = cmap(new_lbls)
                    cvals = np.round(255 * cvals).astype('uint8')
                    cvals[..., -1] = self.volume_prefiltered.alpha
                    frame[0, coords_in_layer[:, 1], coords_in_layer[:, 2]] = new_lbls
                    self.labels[chunk] = frame
                    frame.fill(0)
                    # and now store the corresponding colors
                    frame_rgb[0, coords_in_layer[:, 1], coords_in_layer[:, 2]] = cvals
                    self.labels_rgb[chunk_rgb] = frame_rgb
                    frame_rgb.fill(0)
                    # for coord, label in zip(coords_in_layer, labels_in_layer):
                    #     self.labels[layer_num, coord[1], coord[2]] = label
                print_progress(layer_num, self.labels.shape[0])
        # make an rgb volume for the labels dataset
        # todo: make a function for converting non-zero values in the rgb dataset with ommatidial data
        # todo: check if the raw volume is already stored in the stack 
        # setup the PyQtGraph window
        self.gui_setup()

    def gui_setup(self, app=None, main_window=None, display_window=None):
        """Setup the PyQt5 application and window.

        Parameters
        ----------
        app : QApplication, default=None
            Option to pass an already instantiated QApplication.
        main_window : QMainWindow, default=None
            Option to pass an already instantiated main window.
        display_window : GLViewWidget, default=None
            Option to pass an already instantiated pyqtgraph viewing window for plotting
            3D data.
        """
        # what datasets are available?
        conditions = {'cluster labels':'labels' in dir(self.stack), 
                      'ommatidial data':'ommatidial_data' in dir(self.stack)}
        # these are the functions
        functions = [
            self.plot_ommatidial_clusters,
            self.plot_ommatidial_data]
        # make a main window with options for displaying different graphs
        if app is None:
            self.app = QApplication([])
        if main_window is None:
            self.main_window = QMainWindow()
            self.main_window.resize(800, 800)
        # make a GLViewWidget for displaying 3D volumes
        if display_window is None:
            self.display_window = gl.GLViewWidget()
        self.display_window.setBackgroundColor('w')
        # self.display_window.setBackgroundColor(128, 128, 128, 255)
        # make a layout to place the display window inside the main window
        self.main_layout = QHBoxLayout()
        self.display_window.show()
        # make a frame for controlling the alpha and vmin/vmax of the colormap
        self.display_controls = QVBoxLayout()
        for num, ((lbl, volume), alpha) in enumerate(zip(self.volumes.items(), [20, 128])):
            # add the control panel
            volume.control_panel(title=lbl, layout=self.display_controls)
            # plot the 3D
            volume.plot(window=self.display_window)
        # todo: add a capture window button with a horizontal resolution fill box
        self.capture_row = QHBoxLayout()
        # first, the capture button:
        self.capture_button = QPushButton("capture")
        self.capture_button.released.connect(self.capture_plot)
        self.capture_row.addWidget(self.capture_button)    
        self.display_controls.addLayout(self.capture_row)
        # then the resolution box
        label = QLabel("x-resolution: ")
        self.capture_row.addWidget(label)
        # and finally the box for entering the x resolution
        self.resolution_box = QLineEdit()
        self.resolution_box.setValidator(QtGui.QIntValidator())
        self.resolution_box.setText('1000')
        self.capture_row.addWidget(self.resolution_box)
        # adjust the margins for all of the display_controls contents
        self.display_controls.setContentsMargins(10, 30, 10, 30)
        # vertically center the display controls
        self.display_controls.setAlignment(Qt.AlignCenter)
        # change the GLOption for the filtered volume
        if conditions['cluster labels']:
            self.plot_ommatidial_clusters()
        # add display_controls
        self.main_layout.addLayout(self.display_controls, 1)
        # place GLViewWidget inside main window too
        self.main_layout.addWidget(self.display_window, 5)
        self.widget = QWidget()
        self.widget.setLayout(self.main_layout)
        self.main_window.setCentralWidget(self.widget)
        # resize window to show both the buttons and GLViewWidget
        # run exec loop
        # test the capture functions
        # self.capture_plot()
        self.main_window.show()
        # test: try capturing the window
        self.app.exec()

    def capture_plot(self):
        """Use QWidget.grab() to export the main window as a .png. 
        
        Parameters
        ----------
        xres : int, default=1000
            The resolution of the image along the x- and y-axes in terms of the 
            number of pixels.
        """
        xres = int(self.resolution_box.text())
        size = self.display_window.size()
        ratio = math.ceil(xres / size.width())
        width = ratio * size.width() 
        height = ratio * size.height() 
        # make a timestamp for the filename
        now = datetime.now()
        fn = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}_screenshot.png"
        # generate the numpy array from the image
        image = self.display_window.renderToArray((width, height))
        image[..., -1] = 255
        image = image[..., [2, 1, 0, 3]]
        # plt.imshow(image.astype('uint8'))
        # plt.show()
        # make alpha channel max
        save_image(fn, image)
        print(f"image save in {fn}.")

    def plot_ommatidial_clusters(self):
        self.volume_prefiltered.set_display_volume(key='labels')
        self.volume_prefiltered.update_plot()
        # self.volume_prefiltered.gui.setGLOptions('opaque')
        # self.volume_prefiltered.gui.setGLOptions('additive')

    def plot_ommatidial_data(self):
        breakpoint()

# def linear_cmap(vals, cmap='rainbow', vmin=0, vmax=None):
def linear_cmap(vals, cmap='viridis', vmin=0, vmax=None):
    vals = (np.copy(vals) - vmin)/(vmax - vmin)
    cmap = plt.get_cmap(cmap)
    return cmap(vals)


# folder = 'H:\\My Drive\\backup\\Desktop\\research\\method__compound_eye_tools\\benchmark_3D_data\\manduca_sexta\\_prefiltered_stack\\'
# raw_folder = 'H:\\My Drive\\backup\\Desktop\\research\\method__compound_eye_tools\\benchmark_3D_data\\manduca_sexta\\'
# pixel_length = 4.50074
# zoffset = 762
folder = 'H:\\My Drive\\backup\\Desktop\\research\\method__compound_eye_tools\\benchmark_3D_data\\deilephila_elpenor\\_prefiltered_stack\\'
raw_folder = 'H:\\My Drive\\backup\\Desktop\\research\\method__compound_eye_tools\\benchmark_3D_data\\deilephila_elpenor\\'
pixel_length = 3.325
zoffset = 0
#window_length = np.pi/3
#polar_clustering = True
#stack = CTStack(dirname=folder, img_extension='.tif', bw=True,
#                pixel_size=pixel_length, depth_size=pixel_length)
#stack.ommatidia_detecting_algorithm(neighborhood_smoothing=5, window_length=window_length,
#                                    polar_clustering=polar_clustering, prefiltered=True, stage=0)
pg.setConfigOptions(antialias=True)

gui = VolumeGUI(raw_stack_folder=raw_folder, dirname=folder, img_extension='.tif', bw=True,
                pixel_size=pixel_length, depth_size=pixel_length, zoffset=zoffset,
                bin_length=1, raw_bin_length=2)
# todo: test: make sure everything still works for the raw and prefiltered stacks 
