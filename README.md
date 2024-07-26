# The Framework DUVK

## Environment

Computational platform: PyTorch 1.4.0, NVIDIA Geforce RTX 3090 (GPU), Inter i9-10900X (CPU), CUDA Toolkit 10.0

Development language: Python 3.6
       
Liabraries are listed as follow, which can be installed via the command `pip install -r requirements.txt`.

Please download the crawl-300d-2M.vec.zip from https://fasttext.cc/docs/en/english-vectors.html.

## Datasets

### ReVerb45K (obtained from [1])  
triples: 45k; noun phrases: 15.5k   
used by the previous work: [CESI](https://dl.acm.org/doi/abs/10.1145/3178876.3186030) [SIST](https://ieeexplore.ieee.org/abstract/document/8731346) [JOCL](https://dl.acm.org/doi/abs/10.1145/3448016.3452776)   

### OPIEC59K (obtained from [2])   
triples: 59k; noun phrases: 22.8k      
used by the previous work: [CMVC](https://dl.acm.org/doi/abs/10.1145/3534678.3539449)   

## Reproduce
### Run DUVK on the ReVerb45K data set:
    python DUVK_main_reverb.py
### Run DUVK on the OPIEC59K data set:
    python DUVK_main_opiec.py

