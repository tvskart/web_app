import base64
import sys

from PIL import Image

from io import BytesIO
import numpy as np
from scipy.misc import imresize
from skimage.morphology import skeletonize
from sklearn.externals import joblib
from sklearn import svm
svm_obj=joblib.load('ai/svm.sav')
#img_base64_before_split=sys.argv[1]
img_base64_before_split='data:image/jpeg;base64,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'
img_base64=img_base64_before_split.split('jpeg;base64,')[1]
img_grayscale = Image.open(BytesIO(base64.b64decode(img_base64))).convert('L')
img=np.array(img_grayscale)
img_resized=imresize(img,(8,8))
dim=img_resized.shape
for l in range(dim[0]):
    for m in range(dim[1]):
        if img_resized[l][m]!=0:
            img_resized[l][m]=np.random.randint(10,high=18)

print(img_resized)
#data=img_resized.flatten()
#data = img_resized.reshape((64, -1))
data = img_resized.reshape((1, 64))

# print(data.shape)
predicted = svm_obj.predict(data)
print(predicted[0])