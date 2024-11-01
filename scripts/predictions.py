# Predictions

import torch

import gc
torch.cuda.empty_cache()
gc.collect()



def make_predictions(model, dataloader):
    ce_losses = []
    preds_all = []
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(batch['image'].to(model.device))
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=-1).to(model.device)

            preds_all.append(softmax_outputs)

            ce_loss = model.loss(softmax_outputs, batch['label'].to(model.device))
            ce_losses.append(ce_loss)

    ce_losses = torch.cat(ce_losses) # concatenating all batches to form one tensor
    preds_all = torch.cat(preds_all)
    return ce_losses, preds_all