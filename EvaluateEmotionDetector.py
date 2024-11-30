# import numpy as np
# from keras.models import model_from_json
# import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay


# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# # load json and create model
# json_file = open('D:/project/facial-emtion-recognition-model-main/model/emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)

# # load weights into new model
# emotion_model.load_weights("D:/project/facial-emtion-recognition-model-main/model/emotion_model.keras")
# print("Loaded model from disk")

# # Initialize image data generator with rescaling
# test_data_gen = ImageDataGenerator(rescale=1./255)

# # Preprocess all test images
# test_generator = test_data_gen.flow_from_directory(
#         r'D:\project\facial-emtion-recognition-model-main\model',
#         target_size=(48, 48),
#         batch_size=64,
#         color_mode="grayscale",
#         class_mode='categorical')

# # do prediction on test data
# predictions = emotion_model.predict_generator(test_generator)

# # see predictions
# # for result in predictions:
# #     max_index = int(np.argmax(result))
# #     print(emotion_dict[max_index])

# print("-----------------------------------------------------------------")
# # confusion matrix
# c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
# print(c_matrix)
# cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=emotion_dict)
# cm_display.plot(cmap=plt.cm.Blues)
# plt.show()

# # Classification report
# print("-----------------------------------------------------------------")
# print(classification_report(test_generator.classes, predictions.argmax(axis=1)))
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Mapping of emotions
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the model structure
with open(r'D:\project\facial-emtion-recognition-model-main\model\emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Create model from JSON
emotion_model = model_from_json(loaded_model_json)

# Load model weights
emotion_model.load_weights(r"D:\project\facial-emtion-recognition-model-main\model\emotion_model.keras")
print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1.0 / 255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
    r'D:\project\facial-emtion-recognition-model-main\resources\data\test',  # Update path for test data
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

# Perform predictions on test data
predictions = emotion_model.predict(test_generator)

# Confusion matrix
print("-----------------------------------------------------------------")
c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
print("Confusion Matrix:\n", c_matrix)

# Display confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=list(emotion_dict.values()))
cm_display.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("-----------------------------------------------------------------")
print("Classification Report:\n")
print(classification_report(test_generator.classes, predictions.argmax(axis=1), target_names=list(emotion_dict.values())))
