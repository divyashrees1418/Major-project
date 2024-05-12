
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd 
import os
from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
import librosa as lb
import librosa.display
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rdc_model


root_folder = os.path.abspath(os.path.dirname(__file__))
print(root_folder)
UPLOAD_FOLDER_temp = os.path.join(root_folder, "static")
UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER_temp,"uploads")
print(UPLOAD_FOLDER)
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route("/precautions")
def precautions():
  return render_template("precautions.html")

@app.route("/predict")
def index():
    dir = UPLOAD_FOLDER
   
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    return render_template("index.html",ospf = 1)

@app.route("/predict", methods = ['POST'])
def patient():
    if request.method == "POST":
        
        plt.figure().clear()
        
        print(request)
        name = request.form["name"] 
        lungSounds = request.files["lungSounds"]
        print("\n")
        filename = secure_filename(lungSounds.filename)
       
        lungSounds.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        url2 = os.path.join("static", "uploads")
        url = os.path.join(url2, filename)
      
        print(url)
        absolute_url =  os.path.abspath(url)
        
        
        res_list = rdc_model.classificationResults(absolute_url)

        
        audio1,sample_rate1 = lb.load(url,  mono=True)

        soundWave = librosa.display.waveshow(audio1,sr=sample_rate1, max_points=50000, x_axis='time', offset=0)
        
        plt.savefig("./static/uploads/outSoundWave.png")
        
        mfccs = lb.feature.mfcc(y=audio1, sr=sample_rate1, n_mfcc=40)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        plt.savefig("./static/uploads/outSoundMFCC.png")

        url3 = os.path.join(url2,"outSoundWave.png")
        print(url3)
        res_list.append(os.path.abspath(url3))

    return render_template("index.html",ospf = 0,n = name,  lungSounds = url, res = res_list)


if __name__ == '__main__':
    app.run(debug=True)
