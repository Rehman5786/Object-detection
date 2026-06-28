from flask import Flask, render_template, request
import os
import operation

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], "input.jpg")
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.jpg")

    file.save(input_path)

    # Run classical CV pipeline
    operation.process_image(input_path, output_path)

    return render_template("index.html", output=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)