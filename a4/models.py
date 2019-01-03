import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()
        ######
        # 4.1 YOUR CODE HERE
        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.embed_dim = embedding_dim
        self.fc1 = nn.Linear(embedding_dim,1)
        ######

    def forward(self, x, lengths=None):
        #pass

        ######
        x = self.embed(x)
        print('Shape after embedding',x.shape)
        x = x.mean(dim = 1)
        x = self.fc1(x)
        x = F.sigmoid(x)
        print('Final shape',x.shape)
        return x
        # 4.1 YOUR CODE HERE

        ######

class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()


        ######
        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gru1 = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim,num_layers=self.num_layers)
        self.linear = nn.Linear(embedding_dim,1)
        # 4.2 YOUR CODE HERE

        ######

    def forward(self, x, lengths):
        pass
        x = self.embed(x)
        #print('')
        #print(lengths, "before embedding")
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        #print(x.shape, "after embedding")
        output, h_n = self.gru1(x) #have to transpose if not using padded
        #print(h_n.shape)
        #print('')
        x = h_n[self.num_layers-1]
        x = self.linear(x)
        x = F.sigmoid(x)
        return x

        # 4.2 YOUR CODE HERE

        ######

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.embed_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=n_filters, kernel_size= (filter_sizes[0], embedding_dim))
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels=n_filters, kernel_size= (filter_sizes[1], embedding_dim))
        self.linear = nn.Linear(embedding_dim,1)
        ######
        # 4.3 YOUR CODE HERE

        ######


    def forward(self, x, lengths=None):
        pass
        ######

        x = self.embed(x)
        y = F.relu(self.conv1(x.reshape(-1,1,x.shape[1],x.shape[2]))).squeeze(3)  #batch, channel, len of text - 1
        y, pos = torch.max(y, 2)

        z = F.relu(self.conv2(x.reshape(-1,1,x.shape[1],x.shape[2]))).squeeze(3)
        z, pos = torch.max(z, 2)
        x = torch.cat((y , z), 1)
        return F.sigmoid(self.linear(x))
        ######
