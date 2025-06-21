from flask import Flask, request, jsonify
from scan_service import ScanService, ValidationError
from load_model import load_model

app = Flask(__name__)
scan_service = ScanService()
model = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            raise ValidationError("Image file is required.")

        image_file = request.files["image"]
        image_bytes = image_file.read()

        result = scan_service.predict(image_bytes, model)
        return jsonify({
            "message": "Success scan",
            "data": result
        }), 200

    except ValidationError as ve:
        return jsonify({"message": str(ve)}), 400

    except Exception as e:
        print("Internal error:", e)
        return jsonify({"message": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
