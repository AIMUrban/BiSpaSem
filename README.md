# BiSpaSem: Bidirectional Spatial-Semantic
Interaction for POI Category Annotation

These are the source codes for the **BiSpaSem** model and its corresponding data.

- Data
  
  The files in the datasets/ folder are the data used to train the **BiSpaSem** model.
  
  haidian/haidianF_data.csv is the data of HD
  
  lixia/lixiaF_data.csv is the data of LX
  
  NY/poi_2024_with_cate.csv is the data of BK
  
  TYO/poi_2024_with_cate.csv is the data of TKY
  
  The data in the file is random. We take the first 80% (sorted by their subscripts in ascending order) as the training set and the last 20% as the test set. When considering the neighbors of a POI, we ignore any POIs that belong to the test set. Note that the format of longitude and latitude coordinates in HD and LX is GCJ-02.
  
- Code

  1. src/models/GeoSemPA.py is the code of the main structure of the **BiSpaSem** model.
  
  2. src/models/TCSE_model is the code of **Text-centric Contextual Semantic Encoder**.
  
  3. src/main.py is executed to train the model.
  
  4. Base semantic model see: 
  
     https://huggingface.co/google-bert/bert-base-chinese
  
     https://huggingface.co/google-bert/bert-base-uncased
  
     https://huggingface.co/tohoku-nlp/bert-base-japanese-v3
  
  5. The model weights of the pre-trained Text-centric Contextual Semantic Encoder can be accessed at https://drive.google.com/drive/folders/10z05U8lMkvB41JtLKf-sRTVlm3yhPL2q?usp=drive_link

- Preparing the inputs

  Paths in the preprocessing scripts are resolved relative to this repository, independently of the working directory. Set `id_dataset` consistently in the scripts (`haidian`, `lixia`, `NY`, or `TYO`).

  1. Run `src/datapreprocess.py`. HD/LX CSV fields are read by header (`name`, `location`, `type`), so the extra index column in LX is ignored. For HD/LX, the first run exports latitude/longitude halves to `datasets/<city>/type_tree/poi_original_location_first_half.csv` and `poi_original_location_second_half.csv`. Convert their GCJ-02 coordinates externally, retain row order and the latitude,longitude column order, and save the converted files in the same directory as `poi_processed_original_location_first_half.csv` and `poi_processed_original_location_second_half.csv`. Rerun the script after these files are available. NY/TYO use their existing coordinates directly.
  2. The preprocessing script produces both `datasets/<city>/v1_poi_data_neighbor_k_15.json` and `v1_poi_data with_class_neighbor_k_15.json`.
  3. Run `src/get_neighborNum_from_multi_scale.py` to generate `neighbor_counts_50_500_small_circle.npy`. Set `num_cate` for the selected city: HD 248, LX 140, NY 899, TYO 804.
  4. Put the corresponding language model in `model_weight/bert-base-chinese`, `bert-base-uncased`, or `bert-base-japanese-v3`, then run `src/getPoiNameEmbedding.py` to generate `datasets/<city>/name_embedding.pt`.

  These output names match the paths read by `src/main.py`. The separately trained contextual encoder checkpoint is still required at `model_weight/<city>/v1_class_geobert.pth`.
  

