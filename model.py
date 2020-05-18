# import torch
import torch
import torch.nn as nn
import transformers


class DISTILBERTBaseUncased(nn.Module):
    """
    we'll be using distilbert model. This class we feed the bert input data, the entire pre-trained
    distilbert model and the additional untrained classification layer is trained on our specific task.
    """

    def __init__(self):
        super(DISTILBERTBaseUncased, self).__init__()
        self.bert = transformers.DistilBertModel.from_pretrained(
            "input/distilbert-base-uncased",
        )
        self.bert_drop = nn.Dropout(0.3)  # define the dropout
        self.out = nn.Linear(768, 1)  # fully connected linear layer

    def forward(self, ids, mask):
        # Perform a forward pass. Feeding the inputs in the distilbert model and return last hidden layer
        last_hidden_state = self.bert(ids, attention_mask=mask)
        # pool the first hidden state layer
        last_hidden_state = last_hidden_state[0]
        # Pool the output vectors into a single mean vector
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        # performing dropout on  output vector
        bo = self.bert_drop(mean_last_hidden_state)
        output = self.out(bo)
        return output
