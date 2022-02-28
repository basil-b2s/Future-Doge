from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn
from datetime import date



min_date = date(2022,1,18)

model = pickle.load(open("model.pkl", 'rb'))

app = Flask(__name__)
@app.route('/')

def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    doge_price_list = np.array(
    [[0.190657],[0.19002 ],[0.187705],[0.174117],[0.167765],[0.171313],
    [0.170496],[0.173035],[0.174403],[0.170088],[0.168803],[0.15942 ],
    [0.160213],[0.155023],[0.151954],[0.151065],[0.143359],[0.153399],[0.161652],
    [0.172043],[0.183549],[0.185103],[0.177176],[0.171145],[0.166144]])

    req_date = request.form['date']
    req_date = req_date.split("-")
    user_date = date(int(req_date[0]), int(req_date[1]), int(req_date[2]))
    tot_days = (user_date - min_date).days
    # tot_days +=1
    # print(tot_days)
    (doge_price_list)
    for _ in range(tot_days+1):
        predicted_value = model.predict(doge_price_list.reshape(1,-1)).reshape(1,-1)
        doge_price_list = np.concatenate((doge_price_list[1:], predicted_value))
    
    # print(doge_price_list)
    # print(predicted_value)
    return render_template("home.html", pred="{}".format(np.round(predicted_value[0][0], decimals=4)))
if __name__ == "__main__":
    app.run(debug=True)