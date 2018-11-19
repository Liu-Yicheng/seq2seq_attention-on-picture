import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_size,n_layers=1):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_size, n_layers,
                           bidirectional=True)
        self.hidden_size = hidden_size

    def forward(self, src, hidden=None):
        #outputs:[T, B, H*direction]->[T, B, H*2],
        #hidden:[num_layers*direction, B, H]->[2, B, H]
        outputs, hidden = self.gru(src, hidden)

        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        hidden = hidden.repeat(timestep, 1, 1).transpose(0, 1)  # [B, T, H]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B, T, H]
        energy = F.tanh(self.attn(
                    torch.cat([hidden, encoder_outputs], dim=2)))#[B, T, H]
        energy = energy.transpose(1, 2)  # [B, H, T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B, 1, H]
        attn_energies = torch.bmm(v, energy).squeeze(1)  # [B, T]
        return F.softmax(attn_energies,dim=1).unsqueeze(1)


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        #input:[B]
        #last_hidden:[1, B, H]
        #encoder_outputs[T, B, H]
        embedded = self.embed(input).unsqueeze(0)  #[1, B, E]
        embedded = self.dropout(embedded)
        attn_weights = self.attention(last_hidden[-1], encoder_outputs) #[B, 1, T]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) #[B, 1, H]
        context = context.transpose(0, 1)  #[1, B, H]
        rnn_input = torch.cat([embedded, context], 2) #[1, B, E+H]

        #output:[1, B, H]  hidden:[1, B, H]
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  #[1, B, H] -> [B, H]
        context = context.squeeze(0) #[1, B, H] -> [B, H]
        output = self.out(torch.cat([output, context], 1)) #[B, out_dim]
        output = F.log_softmax(output, dim=1) #[B, out_dim]
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1) #src:[seq_num, batch_size, seq_len]
        max_len = trg.size(0) #trg:[seq_num, bacth_size]
        vocab_size = self.decoder.output_size
        #outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size))
        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            #output = Variable(trg.data[t] if is_teacher else top1).cuda()
            output = Variable(trg.data[t] if is_teacher else top1)
        return outputs
