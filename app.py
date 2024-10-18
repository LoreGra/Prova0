from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Caricare il modello CNN
model = tf.keras.models.load_model('modello_cnn.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Prendi i dati inviati tramite POST (json)
    data = request.get_json(force=True)
    
    # Converti i dati nel formato corretto per il modello
    # Modifica questa parte in base al formato del tuo input
    input_data = np.array(data['input']).reshape((1, 28, 28, 1))  # per immagini 28x28
    
    # Effettua la previsione
    prediction = model.predict(input_data)
    
    # Restituisce la previsione come JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)