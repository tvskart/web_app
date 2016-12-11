import numpy as np

def KNN(k, new_data, train_images):
    vals = {}
    row_iter = 1
    for i in range(len(train_images)):
        train_object=train_images[i]
        train_img=train_object.skeleton
        train_img_dim=train_img.shape
        horizontal_pixels_count=np.zeros(train_img_dim[0])
        vertical_pixels_count =np.zeros(train_img_dim[1])
        for j in range(train_img_dim[0]):
            train_count=0
            test_count=0
            for n in range(train_img_dim[1]):
                if train_img[j][n]==True:
                    train_count+=1
                if new_data[j][n] == True:
                    test_count+=1
            horizontal_pixels_count[j]=abs(train_count-test_count)
        for j in range(train_img_dim[1]):
            train_count = 0
            test_count = 0
            for n in range(train_img_dim[0]):
                if train_img[n][j] == True:
                    train_count += 1
                if new_data[n][j] == True:
                    test_count += 1
            vertical_pixels_count[j] = abs(train_count - test_count)
        distance=np.dot(vertical_pixels_count,horizontal_pixels_count)
        distance_metric = {}
        distance_metric['distance'] = distance
        distance_metric['label'] = train_object.label
        vals[row_iter] = distance_metric
        row_iter += 1

    sorted_vals = sorted(vals.items(), key=lambda x: x[1]['distance'])
    #print(sorted_vals)
    iter_count = 0
    dict_result={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    for key, value in sorted_vals:
        iter_count += 1
        if iter_count > k:
            break
        dict_result[value['label']]+= 1
    sorted_counts = sorted(dict_result.items(), key=lambda x: x[1],reverse=True)
    #print(sorted_counts)
    return sorted_counts[0][0]