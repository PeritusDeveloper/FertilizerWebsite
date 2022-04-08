#import libraries
# from crypt import methods
import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/',methods=['POST','GET'])
def home():
    
    return render_template('home.html')


#To use the predict button in our web-app
@app.route('/Predict',methods=['POST','GET'])
def predict():
    #For rendering results on HTML GUI
    if request.method=='POST':
        pass
        return render_template('final.html')
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2) 
    return render_template('/home.html', prediction_text='Use :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True) 