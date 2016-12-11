from skimage.morphology import skeletonize
import idx2numpy

class image_label:
    def __init__(self,skeleton,label):
        self.skeleton=skeleton
        self.label=label

def get_training_skeleton_images():

    print('Reading training images...')
    images = idx2numpy.convert_from_file('ai/train-images-idx3-ubyte')
    print('Reading training labels...')
    labels=idx2numpy.convert_from_file('ai/train-labels-idx1-ubyte')
    print('Pre Processing Training Images')
    dim=labels.shape
    arr=[]
    for i in range (dim[0]):
        label=labels[i]
        image=images[i]
        dim_img=image.shape
        for j in range(dim_img[0]):
            for k in range(dim_img[1]):
                if image[j][k]<120:
                    image[j][k]=0
                else:
                    image[j][k]=1
        skeleton = skeletonize(image)
        n= image_label(skeleton,label)
        arr.append(n)
    return arr

def get_testing_skeleton_images():

    print('Reading testing images...')
    images = idx2numpy.convert_from_file('ai/t10k-images.idx3-ubyte')
    print('Reading testing labels...')
    labels=idx2numpy.convert_from_file('ai/t10k-labels.idx1-ubyte')
    print('Pre Processing Testing Images')
    dim=labels.shape
    arr=[]
    for i in range (dim[0]):
        label=labels[i]
        image=images[i]
        dim_img=image.shape
        for j in range(dim_img[0]):
            for k in range(dim_img[1]):
                if image[j][k]<120:
                    image[j][k]=0
                else:
                    image[j][k]=1
        skeleton = skeletonize(image)
        n= image_label(skeleton,label)
        arr.append(n)
    return arr