import math 
import torch
import torch.nn.functional as F
import torch.nn as nn

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        print(inputs.shape, targets.shape)
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (N, num_classes)
            targets: ground truth labels with shape (N)
        """
        log_probs = self.logsoftmax(inputs)
        print(log_probs.shape)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        loss = (-targets * log_probs).mean(0).sum()
        return loss


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, embedding_dim, num_classes, scale=30.0, margin=0.4, use_label_smoothing=True):
        '''
        Additive Margin Softmax Loss


        Attributes
        ----------
        embedding_dim : int 
            Dimension of the embedding vector
        num_classes : int
            Number of classes to be embedded
        scale : float
            Global scale factor
        margin : float
            Size of additive margin        
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, embedding_dim, max_norm=1)
        if use_label_smoothing:
            self.loss = CrossEntropyLabelSmooth(num_classes)
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        '''
        x : (N, embedding_dim), labels : (N)
        '''
        n, m = x.shape    
        assert n == len(labels)
        assert m == self.embedding_dim
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.num_classes

        x = F.normalize(x, dim=1) # (N, embedding_dim)
        w = self.embedding.weight # (num_classes, embedding_dim)   
        cos_theta = torch.matmul(w, x.T).T # (N, num_classes)
        psi = cos_theta - self.margin # (N, num_classes)
        
        onehot = F.one_hot(labels, self.num_classes) # (N, num_classes)
        logits = self.scale * torch.where(onehot == 1, psi, cos_theta) # (N, num_classes)
        err = self.loss(logits, labels)
        
        return err


if __name__ == "__main__":
    data = torch.ones((16, 10))
    batch_size = data.shape[0] // 4
    label = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).view(-1)
    loss = AdMSoftmaxLoss(10, 2)
    print(loss(data, label))