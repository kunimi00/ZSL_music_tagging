# ZSL_music_tagging

Zero-shot learning for music auto-tagging and classification.
This code works for both MSD and FMA dataset.
All pre-filtering of tracks and tags were performed beforehand. 
We provide the list of tracks/tags and their necessary matrices.


[Zero-shot Learning for Audio-based Music Classification and Tagging](https://arxiv.org/abs/1907.02670), _Jeong Choi\*, Jongpil Lee\*, Jiyoung Park, Juhan Nam_
_(\* : equally contributed authors)_ Accepted at [ISMIR 2019](https://ismir2019.ewi.tudelft.nl/?q=accepted-papers)


### Separately provided data : _[Link](https://drive.google.com/open?id=10Hd4SQ8CY1Du87t2vICltFsUfjDBdbw8)_
```
├─ data_common
  ├─ msd
    ├─ tag_key_split_TGSPP.p 
        : tag split used in paper
    ├─ track_keys_AB_TRSPP_TGSPP.p 
    ├─ track_keys_A_TRSPP_TGSPP.p
    ├─ track_keys_B_TRSPP_TGSPP.p
    ├─ track_keys_C_TRSPP_TGSPP.p
        : track splits used in paper
          
    ├─ all_tag_to_track_bin_matrix.p
    ├─ tag_ids_in_key_order.p
    ├─ track_ids_in_key_order.p
    ├─ track_id_to_file_path_dict.p
    ├─ tag_key_to_id_dict.p 
       
  ├─ fma
    ├─ ... : same as above 

       
├─ data_tag_vector
  ├─ msd
    ├─ ttr_ont_tag_1126_to_glove_dict.p
        : GloVe vector data 
          (filtered using Tagtraum genre ontology) 
  ├─ fma
    ├─ genre_id_to_inst_posneg40_cnt_norm_dict.p
    ├─ genre_id_to_inst_posneg40_conf_norm_dict.p    
        : Instrument vector data
```


### Audio (mel-spectogram) preparation (in 'scripts' folder)

```console  
python preprocess_audio_msd.py --dir_wav PATH_TO_MSD_AUDIO_WAV --dir_mel PATH_FOR_SAVING_MEL_FILES
```


### Tag/track split data preparation (in 'scripts' folder)

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
