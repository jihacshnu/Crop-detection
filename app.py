import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# You can also use pretrained model from Keras
    

model =tf.keras.models.load_model('PlantDNet.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')






@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        # x = x.reshape([64, 64]);
        disease_class = ['Apple Scab Leaf->Treatment: Apply fungicides like captan or myclobutanil.Cultural Control: Prune trees for better airflow, remove and destroy fallen leaves.', 'Apple leaf->General Health Maintenance: Implement regular pruning, provide adequate nutrition, and water sufficiently.', 'Apple rust leaf->Treatment: Use fungicides such as myclobutanil or mancozeb.Cultural Control: Remove any juniper trees near the apple orchard, as they can host the rust.', 'Bell pepper leaf spot->Treatment: Apply copper-based fungicides.Cultural Control: Rotate crops, avoid overhead watering, and maintain adequate plant spacing.', 'Bell pepper leaf->General Health Maintenance: Provide balanced fertilization, proper irrigation, and crop rotation.', 'Blueberry leaf->General Health Maintenance: Use mulch to retain soil moisture and ensure well-draining soil.', 'Cherry leaf->General Health Maintenance: Prune infected branches and provide adequate nitrogen', 'Corn Gray leaf spot', 'Corn leaf blight->Treatment: Fungicides containing azoxystrobin or propiconazole.Cultural Control: Rotate crops and select resistant hybrids.', 'Corn rust leaf->Treatment: Fungicides like chlorothalonil or mancozeb.Cultural Control: Rotate crops and choose blight-resistant hybrids.', 'Peach leaf->General Health Maintenance: Apply dormant oil spray and maintain appropriate fertilization.', 'Potato leaf early blight->Treatment: Fungicides like chlorothalonil or mancozeb. Cultural Control: Practice crop rotation and use disease-free seeds.', 'Potato leaf late blight->Treatment: Apply fungicides like fluazinam or mancozeb. Cultural Control: Destroy infected plants and use certified disease-free seeds.', 'Potato leaf->General Health Maintenance: Ensure proper irrigation and remove diseased leaves. ', 'Raspberry leaf->General Health Maintenance: Thin canes to improve airflow and apply appropriate fertilizers.', 'Soyabean leaf->General Health Maintenance: Rotate crops and apply proper nitrogen fertilizers.', 'Soybean leaf->eneral Health Maintenance: Same as Soyabean Leaf.', 'Squash Powdery mildew leaf->Treatment: Sulfur-based fungicides or potassium bicarbonate sprays.', 'Strawberry leaf->General Health Maintenance: Mulch well and apply appropriate fertilizers', 'Tomato Early blight leaf->Treatment: Copper-based fungicides or chlorothalonil. Cultural Control: Remove affected leaves and maintain proper plant spacing.', 'Tomato Septoria leaf spot->Treatment: Copper fungicides or mancozeb. Cultural Control: Rotate crops and prune lower leaves.', 'Tomato leaf bacterial spot->reatment: Apply copper-based bactericides. Cultural Control: Rotate crops and remove infected plant material.', 'Tomato leaf late blight->Treatment: Chlorothalonil or mancozeb-based fungicides. Cultural Control: Remove and destroy infected plants immediately.', 'Tomato leaf mosaic virus->Control: Eliminate affected plants and use certified virus-free seeds', 'Tomato leaf yellow virus->Control: Use resistant varieties and apply insecticides to control whitefly vectors.', 'Tomato leaf->General Health Maintenance: Implement regular crop rotation and use balanced fertilizers.', 'Tomato mold leaf-> Treatment: Fungicides containing azoxystrobin or chlorothalonil. Cultural Control: Prune plants and avoid high humidity in greenhouses.', 'Tomato two spotted spider mites leaf->Treatment: Use miticides containing abamectin or sulfur. Cultural Control: Introduce natural predators like predatory mites.', 'grape leaf black rot->Treatment: Apply fungicides like myclobutanil or mancozeb.Cultural Control: Prune vines for airflow, remove mummified berries, and apply dormant sprays. ', 'grape leaf->General Health Maintenance: Ensure proper vine pruning and trellis training.']
        ar = preds[0]
        ind=np.argmax(ar)
        print('Prediction:', disease_class[ind])
        result=disease_class[ind]
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()
