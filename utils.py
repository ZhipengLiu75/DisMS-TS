import torch
from torch import nn
import torch.nn.functional as F


class Sim(nn.Module):
    def __init__(self):
        super(Sim, self).__init__()
        self.similarity_function = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, X):
        X = F.normalize(X, dim=-1)
        batch_size, scale, dimension = X.size()
        loss = 0
        for b in range(batch_size):
            similarity_matrix = self.similarity_function(X[b].unsqueeze(1), X[b].unsqueeze(0))
            loss += F.mse_loss(similarity_matrix, torch.ones(similarity_matrix.shape).to(similarity_matrix.device))
        return loss / batch_size


class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.similarity_function = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, X):
        X = F.normalize(X, dim=-1)
        batch_size, scale, dimension = X.size()
        loss = 0
        for b in range(batch_size):
            similarity_matrix = self.similarity_function(X[b].unsqueeze(1), X[b].unsqueeze(0))
            loss += F.mse_loss(similarity_matrix, torch.eye(scale).to(similarity_matrix.device))
        return loss / batch_size