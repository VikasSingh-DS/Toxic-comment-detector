import torch
import torch.nn as nn
from tqdm import tqdm

# define the binary_cross_entropy loss function
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


# define the trainging function
def train_fn(data_loader, model, optimizer, device, scheduler):
    # Set the model to training mode
    model.train()
    # trange is a tqdm wrapper around the normal python range
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Unpack training batch from our dataloader.
        ids = d["ids"]
        mask = d["mask"]
        targets = d["targets"]

        # copy each tensor to the GPU
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        # clear any previously calculated gradients
        optimizer.zero_grad()
        # outputs prior to activation.
        outputs = model(ids=ids, mask=mask,)

        loss = loss_fn(outputs, targets)  # Perform a loss funtion
        loss.backward()  # Perform a backward pass to calculate the gradients
        optimizer.step()  # Update parameters
        scheduler.step()  # Update the learning rate


# define the validation function
def eval_fn(data_loader, model, device):
    model.eval()  # Set the model to training mode
    fin_targets = []  # target variable
    fin_outputs = []  # ouput variable

    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():
        # trange is a tqdm wrapper around the normal python range
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Unpack validation batch from our dataloader.
            ids = d["ids"]
            mask = d["mask"]
            targets = d["targets"]

            # copy each tensor to the GPU
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            # outputs prior to activation.
            outputs = model(ids=ids, mask=mask,)

            # Move target and output to CPU
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
