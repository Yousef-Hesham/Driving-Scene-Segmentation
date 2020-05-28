# Tensorflow
import tensorflow as tf

# I/O libraries
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

# Helper libraries
import matplotlib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import cv2
import pathlib


# Model
from DeepLabModel import DeepLabModel

# Comment this out if you want to see Deprecation warnings
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

def create_label_colormap():
    """Creates a label colormap used in Cityscapes segmentation benchmark.

    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.array([
        [128,  64, 128],
        [244,  35, 232],
        [ 70,  70,  70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170,  30],
        [220, 220,   0],
        [107, 142,  35],
        [152, 251, 152],
        [ 70, 130, 180],
        [220,  20,  60],
        [255,   0,   0],
        [  0,   0, 142],
        [  0,   0,  70],
        [  0,  60, 100],
        [  0,  80, 100],
        [  0,   0, 230],
        [119,  11,  32],
        [  0,   0,   0]], dtype=np.uint8)
    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(20, 4))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()


#Define All Labels for the Model
LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle', 'void'])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

#Define the model path directory to be loaded 
ModelPath = os.path.join( (pathlib.Path(__file__).parent.absolute()) , "Model/deeplab_model.tar.gz")

#Create instance of DeepLabModel Class and loading the model
MODEL = DeepLabModel(ModelPath)
print('model loaded successfully!')

cap = cv2.VideoCapture("Video/driveseg_sample.mp4")
if cap.isOpened():
    # get vcap property
    video_width = cap.get(3)   # float
    video_height = cap.get(4) # float


fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(video_width),int(video_height)))

i=0
while(cap.isOpened()):
    # grab the frame from the threaded video
    # to have a maximum width of 400 pixels
    print("Processing Frame: " + str(i))
    i+=1
    ret, current_frame = cap.read() #cap.read returns a boolean, and an image [frame] currently displayed.
    if not ret:
        break
    
    original_im = Image.fromarray(current_frame[..., ::-1])
    seg_map = MODEL.run(original_im)

    #vis_segmentation(original_im, seg_map)
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    #plt.imshow(seg_image, alpha=0.7)

    Converted = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
    #cv2.imshow('Driving Scene Segmentation.', Converted)
    out.write(Converted)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print('DONE!')