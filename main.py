# import joblib
# import numpy as np

# # Path to your .joblib file
# joblib_file = "disease_prediction_model.joblib"

# # Load the .joblib file
# try:
#     model_data = joblib.load(joblib_file)
#     model = model_data['model']  # Trained model
#     label_encoder = model_data['label_encoder']  # Label encoder
#     print("Model and label encoder loaded successfully.")
# except Exception as e:
#     print(f"Error loading .joblib file: {e}")
#     exit()

# # Test with a sample symptom description
# sample_symptom = "fever and cough"

# try:
#     # Make a prediction
#     prediction = model.predict([sample_symptom])  # Replace with appropriate preprocessing if needed
#     predicted_disease = label_encoder.inverse_transform(prediction)[0]

#     # Get prediction probabilities (if the model supports it)
#     probabilities = model.predict_proba([sample_symptom])  # Ensure the model supports `predict_proba`
#     confidence = float(np.max(probabilities))

#     print(f"Predicted Disease: {predicted_disease}")
#     print(f"Confidence: {confidence:.2f}")
# except Exception as e:
#     print(f"Error during prediction: {e}")


import http.server
import json
import joblib
import numpy as np
from urllib.parse import parse_qs

class SimpleDiseasePredictionHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Load the model at initialization
        self.predictor = self.load_model()
        super().__init__(*args, **kwargs)

    def load_model(self):
        """
        Load the trained model from the .joblib file
        """
        try:
            model_data = joblib.load('disease_prediction_model.joblib')
            return {
                'model': model_data['model'],
                'label_encoder': model_data['label_encoder']
            }
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict_disease(self, symptom_description):
        """
        Predict the disease based on symptom description
        """
        if not self.predictor:
            return None

        model = self.predictor['model']
        label_encoder = self.predictor['label_encoder']

        try:
            prediction = model.predict([symptom_description])
            predicted_disease = label_encoder.inverse_transform(prediction)[0]

            probabilities = model.predict_proba([symptom_description])
            confidence = float(np.max(probabilities))

            return {
                'predicted_disease': predicted_disease,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def do_OPTIONS(self):
        """
        Handle OPTIONS requests for CORS
        """
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """
        Handle POST requests to the /predict endpoint
        """
        if self.path != '/predict':
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'error', 'message': 'Endpoint not found'}).encode('utf-8'))
            return

        try:
            # Read the request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8'))

            # Validate input
            if 'symptom' not in body or not body['symptom']:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'error',
                    'message': 'Invalid input: symptom is required'
                }).encode('utf-8'))
                return

            # Predict the disease
            result = self.predict_disease(body['symptom'])
            if result is None:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'error',
                    'message': 'Prediction failed: Model not loaded or invalid input'
                }).encode('utf-8'))
                return

            # Send successful response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'success',
                'result': result
            }).encode('utf-8'))

        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'error',
                'message': 'Invalid JSON format'
            }).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'error',
                'message': f'Server error: {str(e)}'
            }).encode('utf-8'))

def run_server(port=5000):
    """
    Run the simple HTTP server
    """
    with http.server.HTTPServer(('', port), SimpleDiseasePredictionHandler) as httpd:
        print(f"Serving on port {port}")
        print("Endpoint: POST http://localhost:5000/predict")
        print("Request body format: {'symptom': 'your symptom description'}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
