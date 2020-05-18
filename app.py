import torch
import transformers
import numpy as np
from flask import Flask, render_template, request
from model import DISTILBERTBaseUncased

MAX_LEN = 320
TOKENIZER = transformers.DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased", do_lower_case=True
)
DEVICE = "cpu"

app = Flask(__name__)


def sentence_prediction(sentence):
    tokenizer = TOKENIZER
    max_len = MAX_LEN
    comment = str(sentence)
    comment = " ".join(comment.split())

    inputs = tokenizer.encode_plus(
        comment,
        None,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/model")
def models():
    return render_template("model.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        sentence = request.form.get("text")
        Toxic_prediction = sentence_prediction(sentence)
        return render_template(
            "index.html", prediction_text=np.round((Toxic_prediction * 100), 2)
        )
    return render_template("index.html", prediction_text="")


if __name__ == "__main__":
    MODEL = DISTILBERTBaseUncased()
    MODEL.load_state_dict(torch.load("weight.bin"))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(debug=True)
