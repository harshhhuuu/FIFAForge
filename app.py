from flask import Flask, request, render_template
import joblib 
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from training_model import Model as m
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from database import Database as D
from k_means import Clustering as C
from pca import PCAModel

model = joblib.load('linear_regression_model.pkl')  # Load the trained model
# Save with protocol=4 for compatibility with Python 3.4+
joblib.dump(model, 'pipeline_model.pkl', protocol=4)
mdl = m()
df = mdl.df
dtbs = D()
 
 
cls = C('fifa_players.csv')  # Initialize clustering with the CSV file
app= Flask(__name__)

@app.route('/')
def index():
    return render_template("dashboard.html")


@app.route('/predict', methods=['GET','POST'])
def predict():
    features= []
    prediction= ""
    if request.method == 'POST':
        #Retrieve form values and convert to floats
        for x in request.form.values():
            features.append(float(x))
        x_new= np.array([features])
        
        #Prediction
        prediction= model.predict(x_new)
        return render_template('simple_form.html',prediction=int(prediction[0]))
    else:
        return render_template('simple_form.html',prediction=prediction)
    
@app.route('/eda', methods=['GET','POST'])
def eda():
    image_file = None
    selected_x = None
    selected_y = None
    selected_plot = None
    title = ''
    df = pd.read_csv('fifa_players.csv')
    columns= df.columns.tolist()
    if request.method == 'POST':
        selected_x = request.form.get('xColumn')
        selected_y = request.form.get('yColumn')
        selected_plot = request.form.get('plotType')
        if selected_plot =='bar':
            plt.bar(df[selected_x], df[selected_y], color='teal')
            title = f'Bar Chart {selected_x} VS {selected_y}'
            plt.title(title)
            plt.xlabel(selected_x)
            plt.ylabel(selected_y)
            plt.legend()
            plt.savefig("static/images/bar.png")
            plt.close()
            image_file = 'images/bar.png'
        elif selected_plot == 'line':
            df.sort_values(by=selected_x,inplace=True)
            plt.plot(df[selected_x], df[selected_y],marker='o', color='blue')
            title = f'Line Chart {selected_x} VS {selected_y}'
            plt.title(title)
            plt.xlabel(selected_x)
            plt.ylabel(selected_y)
            plt.grid(True)
            plt.savefig("static/images/line.png")
            plt.close()
            image_file = 'images/line.png'
        elif selected_plot == 'scatter':
            plt.scatter(df[selected_x], df[selected_y], color='red')
            title = f'Scatter Chart {selected_x} VS {selected_y}'
            plt.title(title)
            plt.xlabel(selected_x)
            plt.ylabel(selected_y)
            plt.savefig("static/images/scatter.png")
            plt.close()
            image_file = 'images/scatter.png'
        elif selected_plot == 'pie':
            plt.pie( df[selected_x], autopct=' %.3f%%',  startangle = 90)
            title= f'Pie Chart {selected_x} VS {selected_y}'
            plt.title(title)
            plt.savefig("static/images/pie.png")
            plt.close()
            image_file = 'images/pie.png'
    return render_template('EDA.html',columns=columns,image_file=image_file,selected_x=selected_x,selected_y=selected_y,selected_plot=selected_plot,title=title)       
               
@app.route('/train-model', methods=['GET', 'POST'])
def train_model():
    s = None
    if request.method == 'POST':
        s = mdl.train()
    return render_template('train.html', score=s) 

@app.route('/add-data', methods=['GET', 'POST'])
def add_data():
    columns = list(zip(df.columns, df.dtypes))
    input_data = {}
    result = None
    message = None

    if request.method == 'POST':
        input_data = request.form.to_dict()
        for name, dtype in columns:
            try:
                if 'int' in str(dtype):
                    input_data[name] = int(input_data[name])
                elif 'float' in str(dtype):
                    input_data[name] = float(input_data[name])
                # else keep as string
            except ValueError:
                message = f"Invalid input for {name}"
                break

        if not message:
            result = dtbs.add_single_document(input_data)
            if result:
                message = 'Insertion Successful'
            else:
                message = 'Insertion Failed'

    return render_template('ADD_data.html', columns=columns, message=message)

 
@app.route('/k-means', methods = ['GET','POST'])
def cluster():
    
    image_file = None
    column1 = None
    column2 = None
    x = None
    scaler = None
    df=pd.read_csv('fifa_players.csv')
    columns = df.select_dtypes(include=['number']).columns.tolist()
    if request.method == 'POST':
        
        column1 = request.form.get('column1')
        column2 = request.form.get('column2')
        x = df[[column1,column2]].values
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        wcss = []
        k_range = range(1,11)
        for k in k_range:
            km = KMeans(n_clusters=k,random_state=42,n_init=12)
            km.fit(x_scaled)
            wcss.append(km.inertia_)
        plt.figure(figsize=(6,4))
        plt.plot(k_range,wcss,marker='o')
        plt.xlabel('NUMBER OF CLUSTER(K)')
        plt.ylabel('WCSS')
        plt.title("number of clusters vs wcss")
        plt.xticks(k_range)
        plt.savefig("static/images/k.png")
        plt.close()
        image_file = 'images/k.png'
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42,n_init=12)
        labels = kmeans.fit_predict(x_scaled)
        df['cluster'] = labels
        print(df['cluster'].value_counts())
        centroids_scaled = kmeans.cluster_centers_
        centroids = scaler.inverse_transform(centroids_scaled)
        plt.figure(figsize=(10,8))
        for i in range(optimal_k):
            plt.scatter( x[labels == i,0], x[labels == i,1], label = f'cluster{i}')
        plt.scatter(centroids[:,0], centroids[:,1], color='black',marker='X', label = 'centroids')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.title(f"cluster between:{column1} VS {column2}")
        plt.legend()
        plt.savefig("static/images/cluster.png")
        plt.close()
        image_file = 'images/cluster.png'
    return render_template('k-means.html',columns=columns, image_file=image_file)

@app.route('/pca', methods=['GET', 'POST'])
def pca_analysis():
    model = PCAModel('fifa_players.csv')
    max_components = model.load_and_prepare_data()  # get max number of PCA components

  
    pca_result = None
    image_file = None
    r2_score_val = None

    if request.method == 'POST':
        try:
            n_components = int(request.form['n_components'])
            model.train_model(n_components=n_components)
            model.get_variance_plot()  # saves image
            pca_df = model.get_variance_plot() 
            r2 = model.evaluate_model()
            r2_score_val = f"{r2 * 100:.2f}" # returns dataframe
            pca_result = pca_df.to_string(index=False)
            image_file = 'images/pca_variance.png'
        except Exception as e:
            pca_result = f"Error: {str(e)}"

    return render_template('pca.html', max_components=max_components, pca_result=pca_result, r2_score=r2_score_val,image_file=image_file)

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    
    
