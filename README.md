# ZSL_music_tagging

Zero-shot learning for music auto-tagging and classification.
This code works for both MSD and FMA dataset.
All pre-filtering of tracks and tags were performed beforehand. 
We provide the list of tracks/tags and their necessary matrices.

[Zero-shot Learning for Audio-based Music Classification and Tagging](https://arxiv.org/abs/1907.02670), _Jeong Choi\*, Jongpil Lee\*, Jiyoung Park, Juhan Nam_
_(\* : equally contributed authors)_ Accepted at [ISMIR 2019](https://ismir2019.ewi.tudelft.nl/?q=accepted-papers)

### Data preparation
 First, prepare tag splits (train / test tags)

```console  
python data_split_tag.py --dataset msd --tag_split_name TGS01 
```

 Using the tag split, prepare track splits (train / valid for AB, A, B)

```console  
python data_split_track.py --dataset msd --tag_split_name TGS01  --track_split_name TRS01 
```



### Model training / inference / evaluation
 
Training 

```console  
python train.py --dataset msd --track_split_name TRS01 --tag_split_name TGS01 --tag_vector_type glove --epochs 20 --track_split A
```

Inference 

```console  
python extract_embeddings_multi.py --load_weights PATH_TO_WEIGHT_FILE
```

Evaluation

```console  
python eval.py --load_weights PATH_TO_WEIGHT_FILE
```
