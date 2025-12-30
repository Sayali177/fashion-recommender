from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model, features, data = pickle.load(open("model.pkl","rb"))

from sklearn.neighbors import NearestNeighbors

def recommend(colour, price):
    colour = str(colour).lower()
    
    user = pd.DataFrame([[colour, price]], columns=['colour','price'])

    user_enc = pd.get_dummies(user[['colour']])
    user_enc = user_enc.reindex(columns=features.columns, fill_value=0)
    user_enc['price'] = price

    distances, indices = model.kneighbors(user_enc)

    return data.iloc[indices[0]].to_dict(orient="records")

@app.route("/", methods=["GET","POST"])
def home():
    results = None
    if request.method == "POST":
        colour = request.form.get("colour")
        price = float(request.form.get("price"))
        results = recommend(colour, price)
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run()
