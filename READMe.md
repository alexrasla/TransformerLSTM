# Example arguments: 

python3 nmtdecoder.py -i ../hw3/test/xml/newsdev2021.en-ha.xml -ha_dict ../hw2/train_file.BPE.L2.json  -en_dict ../hw2/train_file.BPE.L1.json -k 3 -model checkpoint.pth -eval BLEU -codes ../hw2/codes_file 

python3 nmtdecoder.py -i ../hw3/test/xml/newsdev2021.en-ha.en -ha_dict ../hw2/train_file.BPE.L2.json  -en_dict ../hw2/train_file.BPE.L1.json -k 3 -model checkpoint.pth -eval BLEU -codes ../hw2/codes_file 

# Trained model at:
https://ucsb.box.com/s/y9m1rwdm0ew1u36ml5dpmkun11z2qs6e
