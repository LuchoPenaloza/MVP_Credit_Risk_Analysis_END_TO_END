import os
import settings
from middleware import model_predict

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    url_for,
    jsonify
)

router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/", methods=["GET"])
def index():
    """
    Index endpoint, renders our HTML code.
    """
    return render_template("index.html")


@router.route("/application", methods=["POST"])
def application():
    """
    Function used in our frontend so we can upload a form and show the result.
    When it receives a "form" from the UI, it also calls our ML or DL model to
    get and display the predictions.
    """
    
    form_req = request.form.to_dict(flat=False)
    form_dict = {}
    for element in form_req:
        form_dict[element] = form_req[element][0]

    list_info = list(form_dict.values())
    full_name = list_info[0]
    dni = list_info[1]
    result_pred, result_pred_proba = model_predict(form_dict)
    
    if result_pred_proba < 0.5:
        target = "GOOD CLIENT - PRODUCT APPROVED"
    else:
        target = "BAD CLIENT - PRODUCT DISAPPROVED"

    context = {
        "full_name": full_name.upper(),
        "dni": dni,
        "prediction": result_pred,
        "prediction_proba": round(result_pred_proba, 4),
        "target": target
    }
    
    return render_template("response.html", context=context)


@router.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint used to get predictions without need to access the UI.

    Parameters
    ----------
    file : json
        Input json we want to get predictions from.

    Returns
    -------
    flask.Response
        JSON response from our API having the following format:
            {
                "Success": bool,
                "Full_name": str,
                "DNI": str,
                "Prediction": int,
                "Prediction_proba": float,
                "Message": str,
            }
    """

    rpse = {"Success": False, "Full_name": None, "DNI": None, "Prediction": None, "Prediction_score": None, "Message": None}

    try:
        form_dict = request.get_json()
        list_info = list(form_dict.values())
        full_name = list_info[0]
        dni = list_info[1]
        result_pred, result_pred_proba = model_predict(form_dict)

        if result_pred_proba < 0.5:
            mssg = "GOOD CLIENT - PRODUCT APPROVED"
        else:
            mssg = "BAD CLIENT - PRODUCT DISAPPROVED"

        rpse = {
            "Success": True,
            "Full_name": full_name.upper(),
            "DNI": dni,
            "Prediction": result_pred,
            "Prediction_score": round(result_pred_proba, 4),
            "Message": mssg
        }
        return jsonify(rpse)  
    
    # If user sends an invalid request (e.g. no file provided) this endpoint
    # should return `rpse` dict with default values HTTP 400 Bad Request code
    except:
        return jsonify(rpse), 400
