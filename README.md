# emotion_detection
Emotion Detection using Facial Expressions
This is a project that uses deep learning techniques to detect emotions based on facial expressions. The project uses a convolutional neural network (CNN) to classify facial expressions into one of seven emotions: anger, disgust, fear, happiness, neutral, sadness, and surprise.

Dataset
The dataset used in this project is the FER-2013 dataset, which contains 35,887 grayscale images of size 48x48 pixels. The dataset is split into a training set of 28,709 images and a test set of 3,589 images.

Model
The CNN model used in this project consists of five convolutional layers with max pooling, followed by two fully connected layers and a softmax output layer. The model was trained on the training set for 50 epochs using the Adam optimizer and achieved an accuracy of 64% on the test set.

Usage
To use the emotion detection model, run the app.py script and open the URL in your web browser. The application will allow you to upload an image and will display the predicted emotion based on the facial expression in the image.

Deployment
The application has been deployed on Streamlit Sharing for easy access. You can find the deployed application here.

Requirements
The project requires Python 3.7+ and the following packages:

tensorflow
opencv-python-headless
numpy
pandas
streamlit
You can install the required packages by running the following command:

Copy code
pip install -r requirements.txt
Credits
This project was inspired by the Facial Expression Recognition using Keras project on Kaggle.

License
This project is licensed under the MIT License.




