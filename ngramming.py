# python 3

import sys
import os
import glob
import json
import re
from nltk import ngrams

corpus = sys.argv[1]

# Find files
joined_files = os.path.join("*txt")
joined_list = glob.glob(joined_files)

# Bring to corpus
import codecs
files = [codecs.open(file, "r", "utf-8").read() for file in joined_list]
words = [re.sub(r'\W+', ' ', sent) for sent in files]

with open('words_{}'.format(corpus), 'w', encoding ='utf8') as json_file:
    json.dump(words, json_file, ensure_ascii = False)

lines = [re.sub(r' ', '_ _', sent) for sent in words]
lines = ["_" + file for file in lines]
lines = [file + "_" for file in lines]
		
# Split to bigrams
bigram = [str(["".join(k1) for k1 in list(ngrams(file, 2)) if not any(c.isdigit() for c in k1)]) for file in lines]
bigram = [re.sub(r'(\w*\s\w*)', '', file) for file in bigram]
bigram = [re.sub(r'(\b_+\b)', '', file) for file in bigram]
bigram = [file for file in bigram if file]

# Split to trigrams
trigram = [str(["".join(k1) for k1 in list(ngrams(file, 3)) if not any(c.isdigit() for c in k1)]) for file in lines]
trigram = [re.sub(r'(\w*\s\w*)', '', file) for file in trigram]
trigram = [re.sub(r'(\b_+\b)', '', file) for file in trigram]
trigram = [file for file in trigram if file]

# Split to fourgrams
fourgram = [str(["".join(k1) for k1 in list(ngrams(file, 4)) if not any(c.isdigit() for c in k1)]) for file in lines]
fourgram = [re.sub(r'(\w*\s\w*)', '', file) for file in fourgram]
fourgram = [re.sub(r'(\b_+\b)', '', file) for file in fourgram]
fourgram = [file for file in fourgram if file]

with open('bigram_{}'.format(corpus), 'w', encoding ='utf8') as json_file:
    json.dump(bigram, json_file, ensure_ascii = False)


with open('trigram_{}'.format(corpus), 'w', encoding ='utf8') as json_file:
    json.dump(trigram, json_file, ensure_ascii = False)

    
with open('fourgram_{}'.format(corpus), 'w', encoding ='utf8') as json_file:
    json.dump(fourgram, json_file, ensure_ascii = False)
