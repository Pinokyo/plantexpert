from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import keras
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
from io import BytesIO

new_model=tf.keras.models.load_model('tomato_model.h5')
new_model.load_weights('tomato_model.h5')
new_model.compile(loss=keras.losses.categorical_crossentropy, optimizer= 'adam', metrics=['accuracy'])
new_model._make_predict_function()
dic = {0: 'Bacterial spot', 1: 'Early Blight', 2: 'Sağlıklı', 3: 'Late Blight', 4: 'Leaf Mold', 5: 'Septoria Leaf Spot', 6: 'Spider Mites Two Spotted Spider Mite' , 7: 'Target Spot', 8: 'Mocais Virus', 9: 'Yellow Leaf Curl Virus' } 

# API Tanımı ve CORS Ayarını yap
app = Flask(__name__)
api = Api(app)
CORS(app)

# API Classını tanımla
class PlantDiseasePredictor(Resource):
    def get(self):
        return {'message': 'Connection successfull'}
    # Post request tanımla
    
    def post(self):
        try:
            data = request.get_json()
            data = data['picture'].split(',')
            imgdata = base64.b64decode(data[1])
            im = Image.open(BytesIO(imgdata)).convert('RGB')
            im = im.resize((256,256), Image.LANCZOS)
            im = image.img_to_array(im)
            im = im / 255
            im = np.expand_dims(im, axis = 0)
            pred = new_model.predict(im)
            pred = pred.tolist()
            maxR = max(pred[0])
            R = pred[0].index(maxR)
            return {'result': dic[R]}
        except Exception as e:
            return {'error': str(e)}

api.add_resource(PlantDiseasePredictor, '/predictDisease')
# API' ı Çalıştır. Program çalışmaya başlayınca kapatılana kadar "/predictDisease" adlı route' tan bir post request dinliyor olacak.
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

