from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle


pipe = pickle.load(open('pipe.pkl' ,'rb'))
rf_pipe = pickle.load(open('rf_pipe.pkl', 'rb'))
df= pd.read_csv('Clean_Laptop.csv')
app = Flask(__name__) 


@app.route('/')
def index():
    company = sorted(df['Company'].unique())
    type_name = sorted(df['TypeName'].unique())
    ram = sorted(df['Ram'].unique())
    # weight = sorted(df['Weight'].unique())
    touch_screen = sorted(df['TouchScreen'].unique())
    ips = sorted(df['Ips'].unique())
    # ppi = sorted(df['ppi'].unique())
    cpu_brand = sorted(df['Cpu Brand'].unique())
    hdd = sorted(df['HDD'].unique())
    ssd = sorted(df['SSD'].unique())
    gpu_brand = sorted(df['Gpu Brand'].unique())
    os = sorted(df['OS'].unique())
    return render_template('index.html', companies = company, type_names = type_name, rams = ram, touch_screens = touch_screen, ips_s = ips,cpu_brands = cpu_brand,hdd_s = hdd, ssd_s = ssd, gpu_brands = gpu_brand, os_s = os)


@app.route('/predict', methods = ['POST'])
def predict():
    company = request.form.get('brand')
    type_name = request.form.get('type')
    ram = int(request.form.get('ram'))
    weight = float(request.form.get('weight'))
    touch_screen = int(request.form.get('touch'))
    ips = int(request.form.get('ips'))
    ppi = float(request.form.get('s_size'))
    cpu_brand = request.form.get('cpu')
    hdd = int(request.form.get('hdd'))
    ssd = int(request.form.get('ssd'))
    gpu_brand = request.form.get('gpu')
    os = request.form.get('os')
    prediction = np.exp(rf_pipe.predict(pd.DataFrame([[company, type_name, ram, weight, touch_screen, ips, ppi,cpu_brand, hdd, ssd, gpu_brand, os]], columns=['Company','TypeName','Ram',	'Weight', 'TouchScreen','Ips','ppi','Cpu Brand','HDD','SSD','Gpu Brand','OS'])))
    return str (np.round(prediction[0],2))


if __name__ == '__main__':
    app.run(debug=True)
