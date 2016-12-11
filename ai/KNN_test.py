import read_mnist_data as get_image_data
from KNN_image_comparison import KNN
print('Testing KNN model using Test-Data')
k = 27
train_images = get_image_data.get_training_skeleton_images()[:10000]

test_images= get_image_data.get_testing_skeleton_images()[:100]

print(len(test_images))
correct_predictions=0
counter=0
for image_object in test_images:
    print(counter)
    counter+=1
    skeleton=image_object.skeleton
    label=image_object.label
    predicted_result = KNN(k, skeleton, train_images)
    if predicted_result==label:
        correct_predictions+=1

print('Accuracy is ',(correct_predictions*100)/len(test_images))