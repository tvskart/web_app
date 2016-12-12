from net import NeuralNet;
from sklearn.externals import joblib;
import numpy as np;

from scipy.misc import imresize;
from PIL import Image;
from io import BytesIO;
from skimage.morphology import skeletonize;
import base64;

import sys;
img_base64_before_split=sys.argv[1];
#img_base64_before_split='jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAA4ADgDAREAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD/AD/6ACgAoAKACgAoAKACgAoAKAPp/wDYi+CnhX9pT9tD9kT9nPx1qHiDSfBPx+/af+AXwU8Y6p4TutOsfFWm+Ffip8VvCfgXxDqHhm+1jSte0iz8QWeka7eXGjXWqaHrOnW+ox2019pWo2yS2cwB+n//AAcLf8Ev/gF/wSZ/bQ+GP7Of7Ofi/wCMHjTwT40/Zg8F/GvVNU+Nev8AgvxH4qt/FXiP4rfGrwLfafp994F+H/w40iLw/FpHw40O4tbW40O61FNRutVmm1We2ns7OxAPwhoAKACgAoAKAPv/AP4JO/8AKU3/AIJp/wDZ/wD+xv8A+tFfDmgD9/v+D1b/AJSm/AP/ALMA+Fn/AK0V+1VQB/IFQAUAFABQAUAff/8AwSd/5Sm/8E0/+z//ANjf/wBaK+HNAH7/AH/B6t/ylN+Af/ZgHws/9aK/aqoA/kCoAKACgAoAKAPv/wD4JO/8pTf+Caf/AGf/APsb/wDrRXw5oA/f7/g9W/5Sm/AP/swD4Wf+tFftVUAfyBUAFABQAUAFAHuP7MXxw1P9mT9pT9nr9pLRdBsfFOsfs+fHH4TfHDSfDGqXdxYaZ4j1P4T+PdA8eWGg6jfWiS3VnY6vdaBFp93d20clxb29xJNCjyIqkA/t9/4PmNL0yLU/+CY2tRadYxaxf2P7ZGl3+rR2lump3umaRcfst3ek6dd36xi6ubHS7rW9audOtJpXt7K41fVJraOKS/u2lAP4EaACgAoAKACgAoA/v8/4PnP+cXX/AHez/wC+j0AfwB0AFAH/2Q==';
img_base64=img_base64_before_split.split('jpeg;base64,')[1];
img_grayscale = Image.open(BytesIO(base64.b64decode(img_base64))).convert('L');
# img_grayscale = Image.open('ai/image.png').convert('L');
img=np.array(img_grayscale);
img_resized=imresize(img,(28,28));

dim_img=img_resized.shape
for j in range(dim_img[0]):
   for n in range(dim_img[1]):
       if img_resized[j][n]==0:
           img_resized[j][n]=0
       else:
           img_resized[j][n]=1
inp = np.reshape(img_resized, (784,1));
#inp = img_resized.flatten();
#inp1 = np.transpose(inp);
# print(inp.shape);
net_obj = joblib.load('ai/NeuralNet.sav');

outi = net_obj.NNout(inp);
print(outi);