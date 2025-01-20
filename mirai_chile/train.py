def train_model(model, dataset, device, dataloader, optimizer, epoch, dry_run=False):
    assert dataset in ["logit", "transformer_hidden", "encoder_hidden"]

    train_loss = 0

    model.train()

    for batch_idx, data in enumerate(dataloader):

        data[dataset] = data[dataset].to(device)

        if "batch" in data:
            for key, val in data["batch"].items():
                data["batch"][key] = val.to(device)

        optimizer.zero_grad()
        logit, _, _ = model(data[dataset])
        pmf, s = model.head.logit_to_cancer_prob(logit)
        t = data["time_to_event"].to(device)
        d = data["cancer"].to(device)

        loss = model.loss_function(logit, pmf, s, t, d)
        loss.backward()

        train_loss += loss.item()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * data[dataset].size(0), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))
            if dry_run:
                break

    print('\nTrain set: Total loss: {:.6f}\tAverage loss: {:.6f}\tAverage batch loss: {:.6f}'.format(
        train_loss, train_loss / len(dataloader.dataset), train_loss / len(dataloader)))
