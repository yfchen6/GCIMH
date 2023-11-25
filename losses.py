import torch


def negative_log_likelihood_similarity_loss0(u, v, s):
    u = u.double()
    v = v.double()
    omega = torch.mm(u, v.T) / 2
    loss = -((s > 0).float() * omega - torch.log(1 + torch.exp(omega)))
    loss = torch.mean(loss)
    return loss


def negative_log_likelihood_similarity_loss1(u, v, s, bit):
    u = u.double()
    v = v.double()
    omega = torch.mm(u, v.T) / (bit / 18)
    loss = -((s > 0).float() * omega - torch.log(1 + torch.exp(omega)))
    loss = torch.mean(loss)
    return loss


def similarity_loss(outputs1, outputs2, similarity):
    loss = (2 * similarity - 1) - torch.mm(outputs1, outputs2.T) / outputs1.shape[1]
    loss = torch.mean(loss ** 2)
    return loss


def quantization_loss(outputs):
    loss = outputs - torch.sign(outputs)
    loss = torch.mean(loss ** 2)
    return loss


def quantization_loss1(outputs):
    BCELoss = torch.nn.BCELoss()
    loss = BCELoss((outputs + 1) / 2, (torch.sign(outputs) + 1) / 2)
    return loss


def correspondence_loss(outputs_x, outputs_y):
    loss = outputs_x - outputs_y
    loss = torch.mean(loss ** 2)
    return loss
