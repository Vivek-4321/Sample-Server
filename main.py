import http.server
import json
import socketserver
import urllib.parse
import joblib
import numpy as np

class DiseasePredictionHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Load the model when the handler is initialized
        self.predictor = self.load_model()
        super().__init__(*args, **kwargs)

    def load_model(self):
        """
        Load the trained model
        """
        try:
            loaded_data = joblib.load('disease_prediction_model.joblib')
            return {
                'model': loaded_data['model'],
                'label_encoder': loaded_data['label_encoder']
            }
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict_disease(self, symptom_description):
        """
        Predict disease from symptom description
        """
        if not self.predictor:
            return None

        prediction = self.predictor['model'].predict([symptom_description])
        predicted_disease = self.predictor['label_encoder'].inverse_transform(prediction)[0]
        
        # Get prediction probabilities
        probabilities = self.predictor['model'].predict_proba([symptom_description])
        max_confidence = float(np.max(probabilities))
        
        return {
            'predicted_disease': predicted_disease,
            'confidence': max_confidence
        }

    def do_OPTIONS(self):
        """
        Handle CORS preflight requests
        """
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """
        Handle POST requests for disease prediction
        """
        # CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        
        # Check if the path is for prediction
        if self.path != '/predict':
            self.send_error(404, 'Not Found')
            return

        # Get the content length
        content_length = int(self.headers['Content-Length'])
        
        try:
            # Read the request body
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8'))
            
            # Validate input
            if 'symptom' not in body or not body['symptom']:
                self.send_error(400, 'Invalid input: symptom description is required')
                return

            # Predict disease
            result = self.predict_disease(body['symptom'])
            
            if result is None:
                self.send_error(500, 'Model not loaded')
                return

            # Prepare response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = json.dumps({
                'status': 'success',
                'result': result
            })
            
            self.wfile.write(response.encode('utf-8'))

        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
        except Exception as e:
            self.send_error(500, f'Server error: {str(e)}')

def run_server(port=5000):
    """
    Run the HTTP server
    """
    try:
        with socketserver.TCPServer(("", port), DiseasePredictionHandler) as httpd:
            print(f"Serving at port {port}")
            print("Endpoint: POST http://localhost:5000/predict")
            print("Send a POST request with JSON body: {'symptom': 'your symptom description'}")
            httpd.serve_forever()
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    run_server()

# Required dependencies:
# pip install scikit-learn joblib numpy