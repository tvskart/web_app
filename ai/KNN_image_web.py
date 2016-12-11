from scipy.misc import imresize
from PIL import Image
from io import BytesIO
from skimage.morphology import skeletonize
import base64
import read_mnist_data as get_image_data
import numpy as np
from KNN_image_comparison import KNN
import sys
img_base64_before_split=sys.argv[1]
img_base64=img_base64_before_split.split('jpeg;base64,')[1]
img_grayscale = Image.open(BytesIO(base64.b64decode(img_base64))).convert('L')
img=np.array(img_grayscale)
img_resized=imresize(img,(28,28))
dim_img=img_resized.shape
for j in range(dim_img[0]):
    for n in range(dim_img[1]):
        if img_resized[j][n]<120:
            img_resized[j][n]=0
        else:
            img_resized[j][n]=1
k=27
skeleton = skeletonize(img_resized)
train_images = get_image_data.get_training_skeleton_images()
predicted_result=KNN(k, skeleton, train_images[:1])
print(predicted_result)