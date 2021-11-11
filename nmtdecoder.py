import argparse
from sacrebleu import metrics
import torch
import torch.nn as nn
import json
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from hybridnmt import TransformerLSTM, Config, Decoder
import math
import sys
import sacrebleu.metrics as metrics
import sacrebleu
from subword_nmt.apply_bpe import BPE
from subword_nmt.get_vocab import get_vocab

sys.path.append(Config.PATH)
sys.path.append('/Users/alexrasla/Documents/cs291k/hw2')

parser = argparse.ArgumentParser(description='BPE tokenization.')
parser.add_argument('-i')
parser.add_argument('-en_dict')
parser.add_argument('-ha_dict')
parser.add_argument('-ref')
parser.add_argument('-codes')
parser.add_argument('-eval', default='text')
parser.add_argument('-model')
parser.add_argument('-k')
args = parser.parse_args()

class EnglishTestSet(Dataset):
    def __init__(self, test_file, vocab_file, codes_file):
        
        #apply bpe tokenization given code file
        if test_file[-3:] == 'xml':
            os.system(f"python3 extract.py {test_file}")
            test_file = test_file[:-4]
            bpe_tokenized = open(f'{test_file}.en.BPE', "w")
            codes_file = open(codes_file, "r")
            test_file = open(f'{test_file}.en', 'r')
            bpe = BPE(codes_file)
            for line in test_file:
                bpe_tokenized.writelines(bpe.process_line(line))
            engl_test = open(f'{test_file.name}.BPE', 'r')
            engl_test_num_lines = sum(1 for line in open(f'{test_file.name}.BPE', 'r'))
            
        else:
            bpe_tokenized = open(f'{test_file}.BPE', "w")
            codes_file = open(codes_file, "r")
            test_file = open(f'{test_file}', 'r')
            bpe = BPE(codes_file)
            for line in test_file:
                bpe_tokenized.writelines(bpe.process_line(line))
            engl_test = open(f'{test_file.name}.BPE', 'r')
            engl_test_num_lines = sum(1 for line in open(f'{test_file.name}.BPE', 'r'))

        # vocab
        eng_bpe = open(vocab_file, 'r')
        eng_bpe_json = json.load(eng_bpe)

        self.test_x = np.empty(engl_test_num_lines + 1, dtype=list)
        # self.test_y = np.empty(engl_test_num_lines, dtype=list)

        print("[Dataset] English test started")
        count = 0
        unk = 0
        max_sentence_len_en = 0
        for line in engl_test:
            tokens = line.split('\n')
            tokens = tokens[0].split(' ')

            if len(tokens) > max_sentence_len_en:
                max_sentence_len_en = len(tokens)+1

            # change to known length
            english_indexes = np.empty(len(tokens)+1, dtype=int)
            for token in range(len(tokens)):
                try:
                  english_indexes[token] = eng_bpe_json[tokens[token]]
                except:
                  english_indexes[token] = eng_bpe_json['<UNK>']
                  unk += 1
            english_indexes[len(tokens)] = 0

            self.test_x[count] = torch.from_numpy(english_indexes)
            count += 1
        print("[Dataset] English test done", unk)

        longest = np.zeros(max_sentence_len_en)
        longest.fill(Config.PADDING_IDX)

        self.test_x[engl_test_num_lines] = torch.from_numpy(longest)
        self.test_x = pad_sequence(
            tuple(self.test_x), padding_value=Config.PADDING_IDX, batch_first=True)
        self.test_x = self.test_x[0:-1]
        self.num_val_samples = len(self.test_x)
        self.max_len = max_sentence_len_en
        
    def getX(self):
        return self.test_x

    def __getitem__(self, index):
        return self.test_x[index]

    def __len__(self):
        return self.num_val_samples


# beam search algorithm
def beamSearch(sentence, k, model):
    """
    Beam search algorithm, sentence is input sentence, k is amount of top k, model is model
    """
    max_sequence_length = sentence.shape[0]

    best_scores = []
    best_scores.append((0, np.ones(1)))

    # encode to get encoder output (and hidden states?)
    encoder_input = torch.reshape(sentence, (1, -1))
    encoded = model.encode(encoder_input.to(
        Config.DEVICE), 1, max_sequence_length)
    decoder_hidden = None
    for i in range(1, max_sequence_length):
        new_seqs = PriorityQueue(k)
        for score, candidate in best_scores:
            
            if candidate[-1] == 0: #if EOS token reached, add to priority queue
                if not new_seqs.full():
                    new_seqs.put((score, list(candidate)))
                else:
                    if new_seqs.queue[0][0] < score:
                        new_seqs.get()  # pop the one with lowest score
                        new_seqs.put((score, list(candidate)))

            else: #otherwise decoder next token
                candidates = torch.from_numpy(
                    np.array([[candidate[-1]]], dtype=int))
                
                #decoder output
                output, decoder_hidden = model.decode(candidates.to(
                    Config.DEVICE), encoded, 1, candidates.shape[1], hidden_state=decoder_hidden)  # just decoder?
                # print(output)
                predicted_id = torch.nn.functional.log_softmax(output, dim=-1)
                softmaxes = predicted_id[-1].to('cpu')[-1]
                indicies = np.argpartition(softmaxes.to('cpu'), (k*-1))[(k*-1):]

                #add potential new candidates to priority queue
                for index in indicies:
                    sm_score = softmaxes[index]
                    new_candidate = np.append(candidate, index)
                    new_score = np.add(score, sm_score)
                    
                    if not new_seqs.full():
                        new_seqs.put((new_score, list(new_candidate)))
                    else:
                        if new_seqs.queue[0][0] < new_score:
                            new_seqs.get()  # pop the one with lowest score
                            new_seqs.put((new_score, list(new_candidate)))

            #append to new best_scores
            best_scores = []
            while not new_seqs.empty():
                best_scores.append(new_seqs.get())

    #get overall best score
    best_score = -1 * math.inf
    cand = []
    for score, candidate in best_scores:
        if score > best_score:
            best_score = score
            cand = candidate

    return cand


def convertToText(sequence, dictionary):
    '''
    Convert indexes to text
    '''
    ha_bpe = open(dictionary, 'r')
    ha_bpe_json = json.load(ha_bpe)

    # print(ha_bpe_json)
    translation = []
    key_list = list(ha_bpe_json.keys())
    val_list = list(ha_bpe_json.values())
    
    for byte in sequence:
        if byte == Config.PADDING_IDX:
            return translation
        position = val_list.index(byte)
        translation.append(key_list[position])
                
    return translation


if __name__ == "__main__":
    from queue import PriorityQueue
    
    test_file = args.i
    en_dictionary = args.en_dict
    ha_dictionary = args.ha_dict
    evaluation = args.eval
    model_path = args.model
    ref = args.ref
    codes = args.codes
    k_val = int(args.k)

    test_data = EnglishTestSet(test_file, en_dictionary, codes)
    
    test_dataloader = test_data.getX()[:10]
    
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model = TransformerLSTM(
        Config.EMBEDDING_SIZE,
        Config.VOCAB_SIZE,
        Config.D_MODEL,
        Config.NUM_HEADS,
        Config.FEED_FORWARD_DIM,
        Config.DROPOUT,
        test_data.max_len,
        Config.DEVICE,
    ).to(Config.DEVICE)             


    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    
    
    #do beam search each each input
    translations = []
    
    bleu_score = 0
    
    if evaluation != 'text':
        ref_data = EnglishTestSet(ref, ha_dictionary, codes)
        references = []
        for reference in ref_data.getX():
            reference = convertToText(reference, ha_dictionary)
            references.append(" ".join(reference))

    with torch.no_grad():
        for index in range(len(test_dataloader)):
            best_score = beamSearch(test_dataloader[index], k_val, model)
            translation = convertToText(best_score, ha_dictionary)            
            translations.append(" ".join(translation[1:]))

            if index % 10 == 0:
                print(f'[Testing] at {index}')

    # different evaluations
    if evaluation == 'BLEU':
        bleu = metrics.BLEU()
        res = bleu.corpus_score(translations, [references])
        print(f"K value: {k_val}", res, '\n')
    elif evaluation == 'CHRF':
        with open('ref.txt', 'w') as ref:
            with open('cand.txt', 'w') as cand:
                for item in references:
                    ref.write(item)
                for item in translations:
                    cand.write(item)
        os.system(f'sacrebleu ref.txt -i cand.txt -m chrf')
    elif evaluation == 'TER':
        with open('ref.txt', 'w') as ref:
            with open('cand.txt', 'w') as cand:
                for item in references:
                    ref.write(item)
                for item in translations:
                    cand.write(item)
        os.system(f'sacrebleu ref.txt -i cand.txt -m ter')
    elif evaluation == 'text':
        with open('model_output.txt', 'w') as output:
            for item in translations:
                output.write(item + '\n')
        
