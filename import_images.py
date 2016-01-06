import cv2
import scipy
import numpy as np
import Image
import pickle
import time
import linecache

class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        '''
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        '''
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data = DataSets()
    if fake_data:
        data.train = DataSet([], [], fake_data=True)
        data.validation = DataSet([], [], fake_data=True)
        data.test = DataSet([], [], fake_data=True)
        return data
    fmNum = 3     # just count N/A, pass, failure
    # fmNum = 9   # count N/A, pass, C, E, R, CE, ER, CR, CER (failure modes)

    # load sample image to get dimension information (grayscale)
    sample_img = cv2.imread('/home/mike/bagfiles/original_images/gates_patio/o_frame0346.jpg', 0)
    hPix, wPix = sample_img.shape[:2]

    # Note: will need a training and testing vector of images and corresponding labels

    datasets = ('winding_cloudy_up', 'smith_front', 'gates_bridge')
    '''
                'nsh_west_patio', 'health_building')

                'nshnorth_entrance_bright', 'nsheast_door_glass', 'main_walkway_tree', 'track_start_after5',
                'track_box_net', 'squash_court', 'garage_entrance', 'gates_into_darkroom', 'wean_darkroom_entrance',
                'wean_darkroom2', 'cmu_center_pavement', 'cmu_center_lawn', 'cfa_outside_west', 'kraus7',
                'business_building3', 'business_building4', 'nsheast_entrance', 'to_wean', 'smith_conf', 'white_wall',
                'nsh_west_darkness', 'nsh_west_entrance', 'acrossstreet_pall', 'winding_top', 'nsheast_winding_bright4',
                'wean_white_ceiling', 'wean_back_hallway2', 'schenley3', 'black_pavement', 'cmu_walkway',
                'cmu_center_lawn2', 'mini_garden2', 'kraus2', 'kraus4', 'kraus_grill', 'business_coffee',
                'nsheast_door_left', 'frc', 'elevator', 'frc_white_paper', 'track_turf2', 'track_bleachers3',
                'garage_exit_side', 'nsheast_winding_bright3', 'gates_nsh_bridge', 'wean_bridge_exit', 'nsheast_door_close',
                'nsheast_winding_bright6', 'nsheast_winding_bright8', 'schenley', 'porter_door2', 'wean_back_entrance',
                'nsheast_winding_bright9','nsheast_winding_bright11','mini_garden','tennis_courts3','cfa_outside_north',
                'cfa_outside_arch','cfa_outside_south','nat_reserve_run2','winding_sunny_up','nsheast_cloudy',
                'grass_extended','darkness','gates_gate','track_start','schenley2','track_pitch','cfa_inside_classroom',
                'nsh_fourth_entrance','nat_reserve','real_UC','track_turf','porter_door','nsh_wean_bars')
    '''
    startImgs = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0)

    endImgs = (1003, 1753, 1820, 440, 1207, 90, 228, 1678, 132, 197, 253, 220, 37, 256, 705, 42, 125, 376, 205, 96, 43,
               1531, 696, 67, 77, 102, 467, 8825, 891, 67, 227, 168, 891, 1015, 220, 98, 224, 81, 98, 56, 84, 314, 1617,
               247, 68, 111, 160, 1106, 65, 588, 309, 374, 440, 80, 1647, 54, 259, 650, 384, 542, 88, 400, 46, 64, 115,
               1018, 633, 1075, 509, 796, 951, 94, 217, 379, 264, 429, 1054, 179, 587, 706)

    dataLength = len(datasets)

    origS1 = '/home/mike/bagfiles/original_images/'
    origS3 = '/o_frame'
    S5 = '.jpg'

    fcS1 = '/home/mike/bagfiles/feature_counts/'
    fmS3 = '/FM.txt'


    # get total number of images in training set
    totalNumImgs = 0
    for datasetCounter in range (1, dataLength):
        totalNumImgs = totalNumImgs + endImgs[datasetCounter] - startImgs[datasetCounter] + 1

    imgMat = np.empty((totalNumImgs, hPix, wPix, 1), dtype=np.uint8)   # 4D uint8 array to comply with input_data.py
    labMat = np.empty((totalNumImgs, fmNum))


    imgCounterGlobal = 0

    # find and store all images for each dataset (range(a, b) is [a, b))
    for datasetCounter in range(0, 1):

        for imgCounterLocal in range(0, endImgs[datasetCounter] + 1):
            S2 = str(datasets[datasetCounter])
            S4 = '{0:04}'.format(imgCounterLocal)

            img_path = origS1 + S2 + origS3 + S4 + S5 # image file path
            img = Image.open(img_path)  # image type L (blk & white, values from 0 to 255), need to worry about file desc?
            imgMat[imgCounterGlobal, :, :, 0] = img

            fm_path = fcS1 + S2 + fmS3
            fm = linecache.getline(fm_path, imgCounterLocal + 1).rstrip()

            # failure mode one hot vector [N P C E R CE ER CR CER]
            # failure one hot vector [N P F]
            if fm == 'N':
                labMat[imgCounterGlobal] = [1, 0, 0]
            elif fm == 'X':
                labMat[imgCounterGlobal] = [0, 1, 0]
            else:
                labMat[imgCounterGlobal] = [0, 0, 1]

            imgCounterGlobal += 1
    '''
    with open('/home/mike/PycharmProjects/my_cnn_tf/objs.pickle', 'w') as f:
        pickle.dump([imgMat, labMat], f)
    '''

    train_test_split = int(totalNumImgs * 2/3)

    img_mat = imgMat[:train_test_split]
    lab_mat = labMat[:train_test_split]
    img_mat_T = imgMat[train_test_split:]
    lab_mat_T = labMat[train_test_split:]

    data.train = DataSet(img_mat, lab_mat)
    data.test = DataSet(img_mat_T, lab_mat_T)


    return data



# attempts to save workspace variables
'''
with open('/home/mike/PycharmProjects/my_cnn_tf/objs.pickle', 'w') as f:
    pickle.dump(imgMat, f)

filename = '/home/mike/PycharmProjects/my_cnn_tf/shelve.out'
my_shelf = shelve.open(filename,'n') # 'n' for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()
'''


