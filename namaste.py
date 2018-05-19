import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import svm
from sklearn.model_selection import train_test_split

from flask import Flask, send_from_directory, request


iris = pd.read_csv('iris.data.txt', names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

# Asigmanos un numero a cada tipo de flor para que podamos aplicar la SVM
iris['class'] = pd.Categorical(iris['class'])

# Plots
plt.interactive(True)


plt.show(block=True)

# I will be using a ML classifiaction

X = iris.drop(columns=['class'])
y = iris.ix[:, 'class']

X_train, X_test, y_train, y_test = train_test_split(X, y)
# Train model
classififier = svm.SVC()
classififier.fit(X_train, y_train)
print("Score: " + str(classififier.score(X_test, y_test)))

# Server
app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory('', 'index.html')

@app.route('/guess')
def guess():
    # do the magic
    #prediction = classififier.predict([request.args.values])
    print(request.args)
    keys = request.args.keys()
    params = []
    for a in keys:
        try:
            params.append(float(request.args.get(a)))
        except ValueError:
            params.append(0.0)
    return "You've got an Iris " + classififier.predict([params])[0].split('-')[1] + " flower"

app.run(debug=True, host='0.0.0.0')
