import torch
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)
    
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)    
       
    
    def forward(self, features, captions):
        emd = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), emd),1)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.fc_out(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        result = []
        
        embeddings = inputs.clone()
        
        for i in range(max_len):
            op, st = self.lstm(embeddings, states)
            op = self.fc_out(op.squeeze(1))
            ix = op.max(1)[1]
            result.append(ix.item())
            embeddings = self.embed(ix).unsqueeze(1)
        return result
        