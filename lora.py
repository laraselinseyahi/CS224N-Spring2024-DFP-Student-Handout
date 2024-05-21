import torch
import torch.nn as nn

class LORAExtension(nn.Module):
    def __init__(self, model, rank, d):
        super(LORAExtension, self).__init__() # constructor

        self.model = model # to pass in miniBERT as the model
        self.A = nn.Parameter(torch.randn(d, rank))
        self.B = nn.Parameter(torch.randn(rank, d))
        self.C = nn.Parameter(torch.randn(d, rank))
        self.D = nn.Parameter(torch.randn(rank, d))

    def adapt_attention(self, Q, K, V):
        # adding low rank transformations to each of the query, key, and value matrices 
        Q = Q + torch.matmul(Q, torch.matmul(self.A, self.B))
        K = K + torch.matmul(K, torch.matmul(self.A, self.B))
        V = V + torch.matmul(V, torch.matmul(self.A, self.B))
        return Q, K, V
    
    def adapt_ffn(self, W1):
        # modifies W1, which is the weight layer of the first linear layer in the FFN
        return W1 + torch.matmul(W1, torch.matmul(self.C, self.D))

    def forward(self, *input):
        # Modify the model's forward function to use the adapted attention and FFN
        # add in finetuning
        pass
