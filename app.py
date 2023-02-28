import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, flash
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('ECG.h5')


@app.route('/')
@app.route('/home.html')
def about():
    return render_template("home.html")


@app.route('/info.html')
def info():
    return render_template('info.html')


@app.route("/upload.html")
def test():
    return render_template("upload.html")


@app.route("/predict.html/<result>")
def test1(result):
    return render_template("predict.html", result=result)


@app.route("/Left_Bundle_Branch_Block.html")
def p1():
    return render_template("Left_Bundle_Branch_Block.html")


@app.route("/Premature_Atrial_Contraction.html")
def p2():
    return render_template("Premature_Atrial_Contraction.html")


@app.route("/Premature_Ventricular_Contractions.html")
def p3():
    return render_template("Premature_Ventricular_Contractions.html")


@app.route("/Right_Bundle_Branch_Block.html")
def p4():
    return render_template("Right_Bundle_Branch_Block.html")


@app.route("/Ventricular_Fibrillation.html")
def p5():
    return render_template("Ventricular_Fibrillation.html")


@app.route('/upload.html', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            flash('No file selected')
        else:
            basepath = os.path.dirname('__file__')
            filepath = os.path.join(basepath, "static/uploads", f.filename)
            f.save(filepath)

            img = tf.keras.utils.load_img(filepath, target_size=(64, 64))
            x = tf.keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            pred = model.predict(x)
            classes_x = np.argmax(pred, axis=1)
            print(classes_x)
            index = ["Left Bundle Branch Block", "Normal", "Premature Atrial Contraction",
                     "Premature Ventricular Contractions", "Right Bundle Branch Block", "Ventricular Fibrillation"]
            result = str(index[classes_x[0]])
            return render_template('predict.html', result=result)
    return None


if __name__ == "__main__":
    app.run(debug=True)
