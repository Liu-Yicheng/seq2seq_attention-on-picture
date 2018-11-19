import os
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from data import Dataset
import config as cfg
from collections import Counter

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=1000,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='initial learning rate')
    return p.parse_args()


def evaluate(model, data, vocab_size):
    model.eval()
    correct = 0
    val_step = 0
    predict = {}
    total_loss = 0
    data.val_index = 0
    data.val_finish = False
    while(not data.val_finish):
        val_step += 1
        file_indexs, src, trg = data.get_val_batch()
        src = torch.Tensor(src)
        trg = torch.LongTensor(trg)
        #src = Variable(src.data.cuda())
        #trg = Variable(trg.data.cuda())
        src = Variable(src)
        trg = Variable(trg)
        output = model(src, trg, teacher_forcing_ratio=0.0)
        pred = output[1:].view(-1, vocab_size).data.max(1, keepdim=True)[1]
        correct += pred.eq(trg[1:].data.view_as(pred)).cpu().sum()
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                          trg[1:].contiguous().view(-1),)
        total_loss += loss.data.item()
        for index_1, file_index in enumerate(file_indexs):
            for index_2, name in enumerate(file_index):
                if name not in predict.keys():
                    predict[name] = []
                predict[name].append(
                    int(pred[index_1 * cfg.Batch_size+index_2][0].cpu().numpy()))

    with open('./result.txt', 'w') as f:
        for key,item in predict.items():
            label = data.labels[key]
            answer = Counter(item).most_common(1)[0][0]
            f.write(key+'\t'+str(label)+'\t'+str(answer)+'\n')

    return total_loss/(val_step)


def train(e, model, optimizer, data, vocab_size, grad_clip):
    model.train()
    total_loss = 0
    train_step = 50
    for _, _ in enumerate(range(train_step)):
        src,trg = data.get_train_batch()
        src = torch.Tensor(src)
        trg = torch.LongTensor(trg)
        #src, trg = src.cuda(), trg.cuda()

        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data.item()
    return total_loss/train_step


def main():
    args = parse_arguments()
    hidden_size = 128
    embed_size = 128
    en_size = cfg.Class_num
    data = Dataset()

    print("[!] Instantiating models...")
    encoder = Encoder(cfg.Input_dim, hidden_size, n_layers=2)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      n_layers=1, dropout=0.5)
    #seq2seq = Seq2Seq(encoder, decoder).cuda()
    seq2seq = Seq2Seq(encoder, decoder)
    print(seq2seq)
    best_val_loss = None
    lr = args.lr
    bad_times = 0
    for e in range(1, args.epochs+1):
        optimizer = optim.Adam(seq2seq.parameters(), lr=lr)
        train_loss = train(e, seq2seq, optimizer, data,
                           en_size, args.grad_clip)
        val_loss = evaluate(seq2seq, data, en_size)
        print("[Epoch:{}] | train_loss:{:.3f} | val_loss:{:.3f}"
              .format(str(e).zfill(3), train_loss, val_loss))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            bad_times = 0
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
        else:
            bad_times += 1
            if bad_times == 3:
                lr = lr/2 if lr >1e-7 else 1e-7
                bad_times = 0
                print('model did not improve for {} times! Be a man! lr:{}'.format(bad_times,lr))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
