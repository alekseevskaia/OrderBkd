# Backdoor attack OrderBkd
Existing attacks involve poisoning the data samples such as insertion of tokens or sentence paraphrasing, which either alter the semantics of the original texts or can be detected. Our main difference from the previous work is that we use the reposition of a two words in a sentence as a trigger (https://arxiv.org/abs/2402.07689).


### Install packages
- `pdm venv create`


### Available
1. Datasets: `sst-2`, `ag`, `imbd`
2. Models: `bert`, `albert`, `roberta`, `lstm`, `distilbert`


### Run poisoning dataset and model
```
python3 attack_orderbkd.py --dataset sst-2 --model bert
```


### Citation
If the code or method help you in the research, please cite the following paper:
```
@article{Alekseevskaia2024OrderBkdTB,
  title={OrderBkd: Textual backdoor attack through repositioning},
  author={Irina Alekseevskaia and Konstantin Arkhipenko},
  journal={ArXiv},
  year={2024},
  volume={abs/2402.07689},
  url={https://api.semanticscholar.org/CorpusID:267627289}
}
```
