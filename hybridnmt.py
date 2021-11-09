#The encoder uses 2 Transformer layers
#The decoder uses 2 LSTM layers with attention to encoder’s corresponding layer.

#Grading/HW1
#How do I do correct tranlation from Chinese to English
#LSTM (takes input from previous forward) from target, that goes into MHA with encoder out, added, and goes into another LSTM
#hidden state from previous forward is fed into next with target? 
#For loop is encoder to get encoded, then this is fed over and over with decoder one trg at a time or :trg fed in?

%env CUDA_LAUNCH_BLOCKING=1

import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
import sys
import math
import random
from torch.autograd import Variable

class Config():
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    # WEIGHT_DECAY = 0
    BATCH_SIZE = 10
    VOCAB_SIZE = 10735
    EMBEDDING_SIZE = 256
    D_MODEL = 256
    NUM_HEADS = 8
    FEED_FORWARD_DIM = 1024
    NUM_ENCODER_LAYERS = 2
    DROPOUT = 0.1
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PADDING_IDX = 10732
    PATH = "/content/drive/MyDrive/CS291K_HW2/"
    GENERATE_DATASETS = False
    LOAD_MODEL = True
    
class EnglishHausaTrainingSet():
    
    def __init__(self):
        super().__init__()
        #format file
        #go line by line, split by tabs, tokenize sentences, get x and y
        dataset = open(f'{Config.PATH}opus.ha-en.tsv', 'r')
        full_text = ""
        english_text = ""
        hausa_text = ""
        f_name = 'en_ha_file'
        #Adds english and hausa into full text
        for line in dataset:
            sentences = line.split('\t')
            full_text += sentences[0] + '\t' + sentences[1] + '\n'
            
            english_text += sentences[0] + '\n'
            hausa_text += sentences[1] + '\n'
        
        #get tokenizer from formatted file
        f_name = 'en_ha_file'
        en = 'train_file.L1'
        ha = 'train_file.L2'
        num_operations = '10000' #vocab size
        train_file = 'train_file'
        codes_file = 'codes_file'
        vocab_file = 'vocab_file'
        validation_file = 'validation_file'
        
        with open(f_name, 'w') as output:
            output.write(full_text)
        
        with open(en, 'w') as output:
            output.write(english_text)

        with open(ha, 'w') as output:
            output.write(hausa_text)

        #generate test data if true
        ##################### TRAIN DATA #####################
        if Config.GENERATE_DATASETS:
          os.system(f'subword-nmt learn-joint-bpe-and-vocab --input {Config.PATH}{train_file}.L1 {Config.PATH}{train_file}.L2 -s {num_operations} -o {Config.PATH}{codes_file} --write-vocabulary {Config.PATH}{vocab_file}.L')
          os.system(f'subword-nmt apply-bpe -c {Config.PATH}{codes_file} --vocabulary {Config.PATH}{vocab_file}.L1 --vocabulary-threshold 50 < {Config.PATH}{train_file}.L1 > {Config.PATH}{train_file}.BPE.L1')
          os.system(f'subword-nmt apply-bpe -c {Config.PATH}{codes_file} --vocabulary {Config.PATH}{vocab_file}.L2 --vocabulary-threshold 50 < {Config.PATH}{train_file}.L2 > {Config.PATH}{train_file}.BPE.L2')
          os.system(f"python3 {Config.PATH}build_dictionary.py {Config.PATH}{train_file}.BPE.L1 {Config.PATH}{train_file}.BPE.L2")
        ###########################################################
        
        #training and validation file and length
        engl = open(f'{Config.PATH}{train_file}.BPE.L1', 'r')
        engl_num_lines = sum(1 for line in open(f'{Config.PATH}{train_file}.BPE.L1', 'r'))
        
        #vocab
        eng_bpe = open(f"{Config.PATH}{train_file}.BPE.L1.json", 'r')
        eng_bpe_json = json.load(eng_bpe)
        
        #training and validation file and length
        ha = open(f'{Config.PATH}{train_file}.BPE.L2', 'r')
        ha_num_lines = sum(1 for line in open(f'{Config.PATH}{train_file}.BPE.L2', 'r'))
        
        #vocab
        ha_bpe = open(f"{Config.PATH}{train_file}.BPE.L2.json", 'r')
        ha_bpe_json = json.load(ha_bpe)
        
        #get features and labels
        self.x = np.empty(engl_num_lines + 1, dtype=list)
        self.y = np.empty(ha_num_lines + 1, dtype=list)
        
        print("[Dataset] English training started")
        max_sentence_len_en = 0
        count = 0
        unk = 0
        for line in engl:
            tokens = line.split('\n')
            tokens = tokens[0].split(' ')

            if len(tokens) > max_sentence_len_en:
                max_sentence_len_en = len(tokens)+1
            
            english_indexes = np.empty(len(tokens)+1, dtype=int)#change to known length
            for token in range(len(tokens)):
                try:
                  english_indexes[token] = eng_bpe_json[tokens[token]]
                except:
                  english_indexes[token] = eng_bpe_json['<UNK>']
                  unk += 1
            english_indexes[len(tokens)] = 0
            
            self.x[count] = english_indexes
            
            count += 1
            # print(self.x)
        print("[Dataset] English training done", unk) 
        print("[Dataset] Hausa training started") 
        max_sentence_len_ha = 0
        count = 0
        unk = 0
        for line in ha:
            tokens = line.split('\n')
            tokens = tokens[0].split(' ')
            # print(tokens)
            
            if len(tokens) > max_sentence_len_ha:
                max_sentence_len_ha = len(tokens)+1
            
            hausa_indexes = np.empty(len(tokens)+1, dtype=int)#change to known length
            for token in range(len(tokens)):
                try:
                  hausa_indexes[token] = ha_bpe_json[tokens[token]]
                except:
                  hausa_indexes[token] = ha_bpe_json['<UNK>']
                  unk += 1
            hausa_indexes[len(tokens)] = 0
            
            self.y[count] = hausa_indexes

            count += 1
            # print(self.x)
        print("[Dataset] Hausa training done", unk)
            
        # self.x = torch.from_numpy(self.x)
        # np.pad(self.x, (0, max_sentence_len))
        # print(tuple(self.x))
        if max_sentence_len_ha < max_sentence_len_en:
            longest = np.zeros(max_sentence_len_en)
            longest.fill(Config.PADDING_IDX)
            self.x[engl_num_lines] = torch.from_numpy(longest)
            self.y[ha_num_lines] = torch.from_numpy(longest)
            self.max_len = max_sentence_len_en
        else:
            longest = np.zeros(max_sentence_len_ha)
            longest.fill(Config.PADDING_IDX)
            self.x[engl_num_lines] = torch.from_numpy(longest)
            self.y[ha_num_lines] = torch.from_numpy(longest)
            self.max_len = max_sentence_len_ha
            
        # self.x = pad_sequence(tuple(self.x), padding_value=Config.PADDING_IDX, batch_first=True)
        # self.y = pad_sequence(tuple(self.y), padding_value=Config.PADDING_IDX, batch_first=True)

        self.x = self.x[0:-1]
        self.y = self.y[0:-1]
        self.num_train_samples = len(self.x)
   
    def getX(self):
        return self.x
    def getY(self):
        return self.y

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.num_train_samples
    
class EnglishHausaValidationSet(Dataset):
    
    def __init__(self):
        super().__init__()
        # num_operations = '10000' #vocab size
        train_file = 'train_file'
        codes_file = 'codes_file'
        vocab_file = 'vocab_file'
        validation_file = 'validation_file'
        
        #uncomment to get validation data 
        ##################### VALIDATION DATA #####################
        if Config.GENERATE_DATASETS:
          os.system(f'subword-nmt apply-bpe -c {Config.PATH}{codes_file} --vocabulary {Config.PATH}{vocab_file}.L1 --vocabulary-threshold 50 < {Config.PATH}{validation_file}.L1 > {Config.PATH}{validation_file}.BPE.L1')
          os.system(f'subword-nmt apply-bpe -c {Config.PATH}{codes_file} --vocabulary {Config.PATH}{vocab_file}.L2 --vocabulary-threshold 50 < {Config.PATH}{validation_file}.L2 > {Config.PATH}{validation_file}.BPE.L2')
        ###########################################################
        
        #training and validation file and length
        engl_validation = open(f'{Config.PATH}{validation_file}.BPE.L1', 'r')
        engl_validation_num_lines = sum(1 for line in open(f'{Config.PATH}{validation_file}.BPE.L1', 'r'))
        
        #vocab
        eng_bpe = open(f"{Config.PATH}{train_file}.BPE.L1.json", 'r')
        eng_bpe_json = json.load(eng_bpe)
        
        #training and validation file and length
        ha_validation = open(f'{Config.PATH}{validation_file}.BPE.L2', 'r')
        ha_validation_num_lines = sum(1 for line in open(f'{Config.PATH}{validation_file}.BPE.L1', 'r'))
        
        #vocab
        ha_bpe = open(f"{Config.PATH}{train_file}.BPE.L2.json", 'r')
        ha_bpe_json = json.load(ha_bpe)
        
        #get features and labels
        self.val_x = np.empty(engl_validation_num_lines + 1, dtype=list)
        self.val_y = np.empty(ha_validation_num_lines + 1, dtype=list)
        
        print("[Dataset] English validation started") 
        count = 0
        max_sentence_len_en = 0
        unk = 0
        for line in engl_validation:
            tokens = line.split('\n')
            tokens = tokens[0].split(' ')

            if len(tokens) > max_sentence_len_en:
                max_sentence_len_en = len(tokens)+1
            
            english_indexes = np.empty(len(tokens)+1, dtype=int)#change to known length
            for token in range(len(tokens)):
                try:
                  english_indexes[token] = eng_bpe_json[tokens[token]]
                except:
                  english_indexes[token] = eng_bpe_json['<UNK>']
                  unk += 1
            english_indexes[len(tokens)] = 0
            
            self.val_x[count] = torch.from_numpy(english_indexes)
            
            count += 1
        print("[Dataset] English validation done", unk) 
        print("[Dataset] Hausa validation started") 
        count = 0
        max_sentence_len_ha = 0
        unk = 0
        for line in ha_validation:
            tokens = line.split('\n')
            tokens = tokens[0].split(' ')
            # print(tokens)
            
            if len(tokens) > max_sentence_len_ha:
                max_sentence_len_ha = len(tokens)+1
            
            hausa_indexes = np.empty(len(tokens)+1, dtype=int)#change to known length
            for token in range(len(tokens)):
                try:
                  hausa_indexes[token] = ha_bpe_json[tokens[token]]
                except:
                  hausa_indexes[token] = ha_bpe_json['<UNK>']
                  unk += 1
            hausa_indexes[len(tokens)] = 0
            
            self.val_y[count] = torch.from_numpy(hausa_indexes)            
            count += 1

        print("[Dataset] Hausa validation done", unk) 
            
        #pad sequence to max length
        if max_sentence_len_ha < max_sentence_len_en:
            longest = np.zeros(max_sentence_len_en)
            longest.fill(Config.PADDING_IDX)
            self.val_x[engl_validation_num_lines] = torch.from_numpy(longest)
            self.val_y[ha_validation_num_lines] = torch.from_numpy(longest)
            self.max_len = max_sentence_len_en
        else:
            longest = np.zeros(max_sentence_len_ha)
            longest.fill(Config.PADDING_IDX)
            self.val_x[engl_validation_num_lines] = torch.from_numpy(longest)
            self.val_y[ha_validation_num_lines] = torch.from_numpy(longest)
            self.max_len = max_sentence_len_ha
            
        # self.val_x = pad_sequence(tuple(self.val_x), padding_value=Config.PADDING_IDX, batch_first=True)
        # self.val_y = pad_sequence(tuple(self.val_y), padding_value=Config.PADDING_IDX, batch_first=True)

        self.val_x = self.val_x[0:-1]
        self.val_y = self.val_y[0:-1]

        self.num_val_samples = len(self.val_x)
    def getX(self):
        return self.val_x
    def getY(self):
        return self.val_y
    def __getitem__(self, index):
        return self.val_x[index], self.val_y[index]
    
    def __len__(self):
        return self.num_val_samples

class Decoder(nn.Module):
    def __init__(
        self,
        embedding_size, 
        vocab_size,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        max_len,
        device
    ):
        super(Decoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=nhead, batch_first=True)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=d_model) #LSTM decoder layer
        self.lstm2 = nn.LSTM(input_size=embedding_size, hidden_size=d_model)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    #encoder output, decoder hidden state, 
    def forward(
        self, 
        tgt, 
        memory, 
        tgt_mask, 
        memory_mask, 
        tgt_key_padding_mask, 
        memory_key_padding_mask, 
        hidden_state
    ):
        #custom decoder with LSTM
        if hidden_state == None:
          out_lstm, hidden = self.lstm(tgt)
        else:
          out_lstm, hidden = self.lstm(tgt, hidden_state)
        out_lstm = self.layer_norm(out_lstm)

        attn_output, attn_weights = self.attention(out_lstm, memory, memory, need_weights=True)
        attn_output = self.layer_norm(attn_output)

        out = out_lstm + attn_output

        out, hidden = self.lstm2(out)
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = out + out_lstm

        return out, hidden

class PositionalEncoding(nn.Module):
    pass

class TransformerLSTM(nn.Module):
    def __init__(
        self,
        embedding_size, 
        vocab_size,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        max_len,
        device
    ):
        #embedding
        super(TransformerLSTM, self).__init__()
        self.d_model = d_model
        self.device = device
        self.src_word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=Config.PADDING_IDX)
        self.trg_word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=Config.PADDING_IDX)
   
        #encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True) #hidden size 256?
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        #decoder layer
        self.custom_decoder = Decoder(embedding_size, vocab_size, d_model, nhead, dim_feedforward, dropout, max_len, device)

        #transformer with custom decoder
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead,
            num_encoder_layers=2, 
            num_decoder_layers=2, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            custom_decoder=self.custom_decoder, 
            batch_first=True, 
            device=Config.DEVICE
        )
            
        self.linear = nn.Linear(in_features=embedding_size, out_features=vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def src_mask(self, src):
        src_mask = src == Config.PADDING_IDX
        return src_mask

    def make_trg_mask(self, trg):
        trg_mask = trg != Config.PADDING_IDX
        return trg_mask

    def generate_square_subsequent_mask(self, size): # Generate mask covering the top right triangle of a matrix
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg):
        batch_size, src_seq_length  = src.shape
        batch_size, trg_seq_length = trg.shape
        
        src_pe = torch.zeros(batch_size, src_seq_length, self.d_model)
        src_position = torch.arange(0, src_seq_length).unsqueeze(1)
        src_div_term = torch.exp(torch.arange(0, self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model))
        src_pe[:, :, 0::2] = torch.sin(src_position * src_div_term)
        src_pe[:, :, 1::2] = torch.cos(src_position * src_div_term)

        trg_pe = torch.zeros(batch_size, src_seq_length, self.d_model)
        trg_position = torch.arange(0, src_seq_length).unsqueeze(1)
        trg_div_term = torch.exp(torch.arange(0, self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model))
        trg_pe[:, :, 0::2] = torch.sin(trg_position * trg_div_term)
        trg_pe[:, :, 1::2] = torch.cos(trg_position * trg_div_term)
        
        #source and target position    
        embed_src = self.dropout(
            (self.src_word_embedding(src))
        )
        embed_src = embed_src.to(self.device) + src_pe.to(self.device)

        embed_trg = self.dropout(
            (self.trg_word_embedding(trg))
        )
        embed_trg = embed_trg.to(self.device) + trg_pe.to(self.device)
        
        #source and target mask
        # src_key_padding_mask = self.src_mask(src)
        # trg_key_padding_mask = self.src_mask(trg)
        # trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(Config.DEVICE)
        
        out = self.transformer(
            src=embed_src, 
            tgt=embed_trg,
            src_mask=None,
            tgt_mask=trg_mask, 
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=None 
        )

        out = self.linear(out)
        
        return out
    
    def encode(self, src, batch_size, src_seq_length):

        src_pe = torch.zeros(batch_size, src_seq_length, self.d_model)
        src_position = torch.arange(0, src_seq_length).unsqueeze(1)
        src_div_term = torch.exp(torch.arange(0, self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model))
        src_pe[:, :, 0::2] = torch.sin(src_position * src_div_term)
        src_pe[:, :, 1::2] = torch.cos(src_position * src_div_term)
        
        # print('src shape', src.shape)
        embed_src = self.dropout(
            (self.src_word_embedding(src)).to(self.device)
        )

        # print('embed', embed_src.shape)
        embed_src = embed_src.to(self.device) + src_pe.to(self.device)
        
        #source and target mask
        src_key_padding_mask = self.src_mask(src)

        return self.transformer.encoder(src=embed_src, src_key_padding_mask=src_key_padding_mask)

    def decode(self, trg, encoded_src, batch_size, trg_seq_length, hidden_state):
        trg_pe = torch.zeros(batch_size, trg_seq_length, self.d_model)
        trg_position = torch.arange(0, trg_seq_length).unsqueeze(1)
        trg_div_term = torch.exp(torch.arange(0, self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model))
        trg_pe[:, :, 0::2] = torch.sin(trg_position * trg_div_term)
        trg_pe[:, :, 1::2] = torch.cos(trg_position * trg_div_term)

        embed_trg = self.dropout(
            (self.trg_word_embedding(trg))
        )
        embed_trg = embed_trg.to(self.device) + trg_pe.to(self.device)      

        decoded, hidden = self.transformer.decoder(tgt=embed_trg, memory=encoded_src, tgt_mask=None, tgt_key_padding_mask=None, memory_mask=None, memory_key_padding_mask=None, hidden_state=hidden_state)
        decoded = self.linear(decoded)
        # softmax = self.softmax(decoded)
        return decoded, hidden

def get_batch(batch_num, dataloader):
  data_batch = (dataloader[0][batch_num: (batch_num+Config.BATCH_SIZE)], dataloader[1][batch_num: (batch_num+Config.BATCH_SIZE)])
  
  max_length = 0
  for sequence in range(len(data_batch[0])):
    if len(data_batch[0][sequence]) > max_length:
      max_length = len(data_batch[0][sequence])
    if len(data_batch[1][sequence]) > max_length:
      max_length = len(data_batch[1][sequence])
  
  new_x = []
  for sequence in range(len(data_batch[0])):
    new_seq = []  
    for word in data_batch[0][sequence]:
      new_seq.append(word)
    while len(new_seq) < max_length:
      new_seq.append(Config.PADDING_IDX)
    new_x.append(new_seq)

  new_y = []
  for sequence in range(len(data_batch[1])):
    new_seq = []  
    for word in data_batch[1][sequence]:
      new_seq.append(word)
    while len(new_seq) < max_length:
      new_seq.append(Config.PADDING_IDX)
    new_y.append(new_seq)
    np.array(new_y).shape

  new_x = torch.as_tensor(new_x, dtype=int)
  new_y = torch.as_tensor(new_y, dtype=int)

  return (new_x, new_y)

def shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def one_epoch(model, dataloader, running_loss, writer, loss_function, epoch, start_batch, optimizer, train):
    if train == True:
      model.train()
    else:
      model.eval()
    
    update = 0
    number_exeptions = 0
    for index in range(start_batch, len(dataloader[0]), Config.BATCH_SIZE):

        if train:
          print(f"[Training Epoch] {epoch}/{Config.NUM_EPOCHS - 1}, Batch Number: {index}/{len(dataloader[0])}")
        else:
          print(f"[Validation Epoch] {epoch}/{Config.NUM_EPOCHS - 1}, Validating: {index}/{len(dataloader[0])}")
        try:
          input, target = get_batch(index, dataloader)

          extra_input_padding = torch.full((Config.BATCH_SIZE, 1), Config.PADDING_IDX, dtype=int)
          bos_target_padding = torch.full((Config.BATCH_SIZE, 1), 1, dtype=int)

          model_input = torch.cat((input, extra_input_padding), dim=1)
          model_target = torch.cat((bos_target_padding, target), dim=1)

          encoder_output = model.encode(model_input.to(Config.DEVICE), model_input.shape[0], model_input.shape[1]).to(Config.DEVICE)
          decoder_hidden = None
        
          
          loss = 0
          for word in range(input.shape[1]): #loop through inputs, get one output at a time and do loss

            current_sequence = torch.reshape(model_target[:, word], (Config.BATCH_SIZE, 1))

            decoder_output, decoder_hidden = model.decode(current_sequence.to(Config.DEVICE), encoder_output, Config.BATCH_SIZE, current_sequence.shape[1], hidden_state=decoder_hidden)

            loss_output = decoder_output.reshape(-1, decoder_output.shape[2])
            loss_target = target[:, word].reshape(-1)

            loss += (loss_function(loss_output.to(Config.DEVICE), loss_target.to(Config.DEVICE)))

          if train:
            loss /= (input.shape[1])
            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()

          

        except Exception as e:
          number_exeptions += 1
          print('[EXCEPTION]', e)
          print('Memory', torch.cuda.memory_allocated(Config.DEVICE))
          print('Number Exceptions', number_exeptions)
          torch.cuda.empty_cache()
          continue

        update += 1
        running_loss += loss.item()
        
        #update tensorboard and save model
        if update == 10:    # every 10 mini-batches
            running_avg = running_loss / 10
            graph = ''
            if train:
              checkpoint = {
                  "epoch":epoch,
                  "batch":index,
                  "model_state":model.state_dict(),
                  "optim_state":optimizer.state_dict()
              }
              torch.save(checkpoint, f'checkpoint_t.pth')
              graph = 'training loss'
            else:
              graph = 'validation loss'
            writer.add_scalar(graph,
                            running_avg,
                            epoch * len(dataloader) + index)
            print(f"[Loss] {running_avg}")
            running_loss = 0.0

            update = 0
            
if __name__ == '__main__':
    print('Device:', Config.DEVICE)

    training_data = EnglishHausaTrainingSet()
    validation_data = EnglishHausaValidationSet()

    training_dataloader = (training_data.getX(), training_data.getY()) 
    validation_dataloader = (validation_data.getX(), validation_data.getY())

    model = TransformerLSTM(
        Config.EMBEDDING_SIZE,
        Config.VOCAB_SIZE,
        Config.D_MODEL,
        Config.NUM_HEADS,
        Config.FEED_FORWARD_DIM,
        Config.DROPOUT,
        training_data.max_len,
        Config.DEVICE,
    ).to(Config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=Config.PADDING_IDX)
    start_epoch = 0
    start_batch = 0

    if Config.LOAD_MODEL:
        checkpoint = torch.load(f'checkpoint.pth', map_location=Config.DEVICE)

    start_batch = checkpoint["batch"]
    start_epoch = checkpoint["epoch"]  
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number Parameters:", pytorch_total_params)
    #tensorboard
    writer = SummaryWriter("runs")

    running_loss = 0.0
    running_val_loss = 0.0

    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        print(f"[Epoch] {epoch}/{Config.NUM_EPOCHS - 1}")
        
        training = training_dataloader#shuffled_copies(training_dataloader[0], training_dataloader[1])
        validation = validation_dataloader#shuffled_copies(validation_dataloader[0], validation_dataloader[1])
        
        one_epoch(model, training, running_loss, writer, loss_function, epoch, start_batch, optimizer, train=True)
        one_epoch(model, validation, running_val_loss, writer, loss_function, epoch, start_batch, optimizer, train=False)
        
        start_batch = 0
        