import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn import model_selection
from transformers import AdamW, get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import DISTILBERTBaseUncased


# running script funtion
def run():
    # Reading the data file
    dfx = pd.read_csv(config.TRAINING_FILE, usecols=["comment_text", "toxic"]).fillna(
        "none"
    )

    # Spliting data into training 90% and validation 10%
    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.toxic.values
    )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # pass the sentence and target from training dataset into class
    train_dataset = dataset.DISTILBERTDataset(
        comment_text=df_train.comment_text.values, target=df_train.toxic.values
    )

    # Combine the training inputs into a TensorDataset.
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    # pass the sentence and target from validation dataset into class
    valid_dataset = dataset.DISTILBERTDataset(
        comment_text=df_valid.comment_text.values, target=df_valid.toxic.values
    )

    # Combine the validation inputs into a TensorDataset.
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cuda")  # define the device
    model = DISTILBERTBaseUncased()  # define the model
    model.to(device)  # copy the model to the gpu

    # Prepare optimizer and schedule (linear warmup and decay)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # Create the numer of training steps, optimizer and scheduler
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # running the loop for every epochs
    best_f1_score = 0
    for epoch in range(config.EPOCHS):
        # passing training and validation funtion
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        # evalution metrics
        f1_score = metrics.f1_score(targets, outputs)
        print(f"F1 Score = {f1_score}")
        # saving the model
        if f1_score > best_f1_score:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_f1_score = f1_score


if __name__ == "__main__":
    run()
