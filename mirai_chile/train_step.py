import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def get_discriminator_loss(hidden, logit, batch, loss_function, discriminator_model):
    data = logit
    device_logit = discriminator_model(data)
    batch = {'device_logit': device_logit, 'device': batch['device']}
    gen_loss, adv_loss = loss_function(batch)
    return gen_loss, adv_loss


def discriminator_step(hidden, logit, batch, loss_function, discriminator_model, discriminator_optimizer):
    hidden_with_no_hist, logit_no_hist = hidden, logit.detach()
    _, loss = get_discriminator_loss(hidden_with_no_hist, logit_no_hist, batch, loss_function, discriminator_model)

    loss.backward()
    discriminator_optimizer.step()
    discriminator_optimizer.zero_grad()
    return loss


def mirai_step(data, models, optimizers, device, loss_functions, dry_run=False):
    if "batch" in data:
        for key, val in data["batch"].items():
            data["batch"][key] = val.to(device)
    else:
        data["batch"] = None

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            data[key] = val.to(device)

    mirai_optimizer = optimizers['mirai']
    mirai_model = models['mirai']
    mirai_loss_function = loss_functions['mirai']
    args = mirai_model.args

    discriminator_optimizer = optimizers['discriminator']
    discriminator_model = models['discriminator']
    discriminator_loss_function = loss_functions['discriminator']

    mirai_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    for _ in range(args.num_adv_steps):
        with torch.no_grad():
            logit, hidden, _ = mirai_model(data['data'], data["batch"])

        device_logit = discriminator_model(torch.cat([hidden.detach(), logit], dim=1))
        data['device_logit'] = device_logit
        adv_loss, disc_loss = discriminator_loss_function(data)
        if disc_loss > args.manufacturer_entropy:
            discriminator_optimizer.zero_grad()
            disc_loss.backward()
            discriminator_optimizer.step()

    logit, hidden, _ = mirai_model(data['data'], data["batch"])
    if isinstance(mirai_model, DDP):
        pmf, sf = mirai_model.module.head.logit_to_cancer_prob(logit)
    else:
        pmf, sf = mirai_model.head.logit_to_cancer_prob(logit)

    data['logit'] = logit
    data['pmf'] = pmf
    data['sf'] = sf

    model_loss = mirai_loss_function(data)

    device_logit = discriminator_model(torch.cat([hidden, logit], dim=1))
    data['device_logit'] = device_logit
    adv_loss = - discriminator_loss_function(data)[1]

    # print(model_loss, adv_loss)

    total_loss = model_loss + adv_loss

    mirai_optimizer.zero_grad()
    total_loss.backward()
    mirai_optimizer.step()

    return total_loss, adv_loss

    # logit, hidden, _ = mirai_model(data['data'], None)
    # print(hidden)
    #
    # num_discriminator_steps = 0
    # with torch.autograd.set_detect_anomaly(True):
    #     discriminator_loss, adv_loss = get_discriminator_loss(hidden, logit, data, discriminator_loss_function, discriminator_model)
    #     adv_loss.backward(retain_graph=True)
    #     discriminator_optimizer.step()
    #     discriminator_optimizer.zero_grad()
    #     discriminator_step(
    #         hidden,
    #         logit,
    #         data,
    #         discriminator_loss_function,
    #         discriminator_model,
    #         discriminator_optimizer
    #     )
    #     while adv_loss.cpu().item() > args.manufacturer_entropy and num_discriminator_steps < args.num_adv_steps:
    #         num_discriminator_steps += 1
    #         adv_loss = discriminator_step(
    #             hidden,
    #             logit,
    #             data,
    #             discriminator_loss_function,
    #             discriminator_model,
    #             discriminator_optimizer
    #         )
    #
    #     if isinstance(mirai_model, DDP):
    #         pmf, sf = mirai_model.module.head.logit_to_cancer_prob(logit)
    #     else:
    #         pmf, sf = mirai_model.head.logit_to_cancer_prob(logit)
    #
    #     data['logit'] = logit
    #     data['pmf'] = pmf
    #     data['sf'] = sf
    #
    #     model_loss = mirai_loss_function(data)
    #
    #     loss = model_loss + discriminator_loss
    #     loss.backward()
    #     mirai_optimizer.step()
    #
    #     return model_loss, discriminator_loss
