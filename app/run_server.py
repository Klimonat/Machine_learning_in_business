import dill
import pandas as pd
import os

dill._dill._reverse_typemap['ClassType'] = type
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(model_path):
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    print(model)


modelpath = "./gradient_boosting_pipeline.dill"
load_model(modelpath)


@app.route("/", methods=["GET"])
def general():
    return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")
    if flask.request.method == "POST":
        duration, nr_employed, cons_conf_idx, euribor3m, pdays = "", "", "", "", ""
        request_json = flask.request.get_json()
        if request_json["duration"]:
            duration = request_json['duration']
        if request_json["nr_employed"]:
            nr_employed = request_json['nr_employed']
        if request_json["cons_conf_idx"]:
            cons_conf_idx = request_json['cons_conf_idx']
        if request_json["euribor3m"]:
            euribor3m = request_json['euribor3m']
        if request_json["pdays"]:
            pdays = request_json['pdays']
        logger.info(f'{dt} Data: duration={duration}, nr_employed={nr_employed}, cons_conf_idx={cons_conf_idx}, euribor3m={euribor3m}, pdays={pdays}')
        try:
            preds = model.predict_proba(pd.DataFrame({"duration": [duration],
                                                      "nr_employed": [nr_employed],
                                                      "cons_conf_idx": [cons_conf_idx],
                                                      "euribor3m": [euribor3m],
                                                      "pdays": [pdays]}))
        except AttributeError as e:
            logger.warning(f'{dt} Exception: {str(e)}')
            data['predictions'] = str(e)
            data['success'] = False
            return flask.jsonify(data)

        data["predictions"] = preds[:, 1][0]
        if preds[:, 1][0] >= 0.5:
            data["success"] = True
        else:
            data["success"] =False

    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
