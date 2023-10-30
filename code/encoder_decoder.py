import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
import tsp
from torch.utils.data import DataLoader 
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from scipy.spatial import distance_matrix
from tsp import TSPDataset


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.1):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.input_dim, self.hidden_dim)
        self.LSTM = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, state):
        embedded = self.embedding(state)
        embedded = self.dropout(embedded)
        context_vector, hidden = self.LSTM(embedded)
        return context_vector, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_dim, n_action):
        super(Decoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_action = n_action #number of city 

        self.embedding = nn.Linear(1, self.hidden_dim)
        self.LSTM = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.out = nn.Linear(self.hidden_dim, self.n_action)

    def apply_mask_to_logits(self, logits, mask, indexes):
        batch_size = logits.size(0)
        clone_mask = mask.clone()
        if indexes is not None:
            clone_mask[[i for i in range(batch_size)], indexes.data.squeeze(1).long()] = 1
            logits[clone_mask.unsqueeze(1)] = -np.inf
        else:
            logits[:, :] = -np.inf
            # we want to start from depot, ie the first node
            logits[:, :, 0] = 1
        return logits, clone_mask

    def forward(self, encoder_output, encoder_hidden):
        batch_size = encoder_output.size(0)
        seq_len = encoder_output.size(1)

        decoder_input = torch.ones(batch_size, 1)
        decoder_hidden = encoder_hidden
        decoder_outputs = [] 

        paths = [] 
        path_logp = []
        mask = torch.zeros(batch_size, seq_len).byte()

        chosen_indexes = None 

        for i in range(seq_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            # use decoder output as the next input 
            _, topi  = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1).detach().float()
            masked_logits, mask = self.apply_mask_to_logits(decoder_output, mask, chosen_indexes)

            # transform decoder output to actual result 
            chosen_indexes = torch.argmax(masked_logits, dim=2).float() 
            log_probs = F.log_softmax(decoder_output, dim=2)
            logp = torch.gather(log_probs, 2, chosen_indexes.unsqueeze(2).long()).squeeze(2)

            path_logp.append(logp.unsqueeze(1))
            paths.append(chosen_indexes.unsqueeze(1))

        paths = torch.cat(paths, 2)
        path_logp = torch.cat(path_logp, dim=1)

        return paths, path_logp

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.LSTM(output.unsqueeze(1), hidden)
        output = self.out(output)
        return output, hidden 
    
def reward_fn(static, tour_indices):
    """
    static: [batch_size, 2, sequence_length]
    tour_indices: [batch_size, tour_length]
    Euclidean distance between all cities / nodes given by tour_indices
    """
    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)
    tour_len = torch.sqrt(torch.sum(torch.pow(tour[:, :-1] - tour[:, 1:], 2), dim=2))

    return tour_len.sum(1)
    
class Seq2SeqTSPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length):
        super().__init__()
        # self.hidden_dim=hidden_dim
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, sequence_length)

    def forward(self, inputs):
        encoder_outputs, encoder_hidden = self.encoder(inputs)

        paths, path_logp = self.decoder(encoder_outputs, encoder_hidden)

        return paths, path_logp
    
def trainSeq2SeqTSPWithRL(train_dataset, test_dataset, epochs, experiment_details, batch_size=10, num_nodes=13, lr=1e-4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = Seq2SeqTSPModel(input_dim=num_nodes, hidden_dim=128, sequence_length=num_nodes)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    val_loss = []
    train_loss = []
    val_loss_per_epoch = []
    losses_per_epoch = []

    path_lengths_per_epoch = []

    for ep in range(epochs):
        model.train()

        loss_at_epoch = 0.0
        val_loss_at_epoch = 0.0
        iterator = tqdm(train_loader, unit='Batch')

        for batch_id, sample_batch in enumerate(iterator):
            optimizer.zero_grad()
            train_batch = Variable(sample_batch['Points'])
            output_paths, path_logp = model(train_batch)

            reward = reward_fn(train_batch.transpose(1,2), output_paths.squeeze(1).to(torch.int64))
            loss = torch.mean(reward.detach() * path_logp.sum(dim=1))

            loss.backward()
            optimizer.step()
            loss_at_epoch += loss.detach().sum().item()
            average_path_length = 0

            # for path in range(batch_size):
            #     points = sample_batch['Points'][path]
            #     distance_matrix_array = distance_matrix(points, points)

        model.eval()

        for val_batch in validation_loader:
            train_batch = Variable(val_batch['Points'])
            tours, tour_logp = model(train_batch)

            reward = reward_fn(train_batch.transpose(1, 2), tours.squeeze(1).to(torch.int64))

            loss = torch.mean(reward.detach() * tour_logp.sum(dim=1))

            val_loss.append(loss.data.item())
            val_loss_at_epoch += loss.detach().item()

    return model, tours

if __name__ == '__main__':
    num_nodes = 10
    train_size = 1
    hidden = 128 
    embedding_dim = 6
    db = TSPDataset(train_size, num_nodes)
    # train_loader = DataLoader(db, batch_size=10, shuffle=True, num_workers=1)
    # print(torch.tensor(db.data['Points_List'], dtype=torch.float32))
    encoder = Encoder(num_nodes, hidden)
    context, hidden = encoder(torch.tensor(db.data['Points_List'], dtype=torch.int64))
    print(context)

    # epochs = 100
    # test_size = 1
    # batch_size = 10
    # lr = 1e-4
    # train_dataset = TSPDataset(train_size, num_nodes)
    # test_dataset = TSPDataset(test_size, num_nodes)

    # experiment_details = f'seq2seq_epochs{epochs}_train{train_size}_seqLen{num_nodes}_batch{batch_size}_lr{lr}'

    # trainSeq2SeqTSPWithRL(train_dataset,
    #                                                 test_dataset,
    #                                                 epochs,
    #                                                 experiment_details,
    #                                                 batch_size,
    #                                                 num_nodes,
    #                                                 lr)
    
