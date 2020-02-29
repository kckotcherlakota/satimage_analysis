#The trained model is saved as the classifier.h5
#The predictor.py shows the way to use that seperately. 

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

#Make sure the version is correct
classifier = load_model('classifier.h5')


test_image = image.load_img('dataset/single_prediction/caa.jpg', target_size=(28,28))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
#print(np.argmax(result))
#print(result)
{'mgs': 1, 'acrim': 0}
if result[0][0] == 1:
    prediction = 'msg'
else:
    prediction = 'acrim'
    
print(prediction)
