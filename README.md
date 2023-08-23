# MANIQA_reproducing
Reproducing results and examining the MANIQA, method of Blind Image Quiality Assessment.
## Current Results
The training has been conducted with 2 train/test splits (80:20 correspondingly) with different seeds, 5fold grouped CV inside the train split. The resulting SROCC in the split is the average of the results of 5 CV models on the corresponding test split.
| Backbone | SROCC, split 1 | SROCC, split 2 |
| --- | --- | --- |
|vit_base_patch8_224	| 0.9509	| 0.7422|
|vit_large_patch14_224_clip_laion2b |	0.9362 |	0.6803|
|vit_huge_patch14_224_clip_laion2b |	0.9466 |	0.6765|
## Current Conclusions
Although the number of experiments is small, some intermediate conclusions can be drawn.
The model's performance is highly susceptible to the split. 
Also the averaging method during inference impacts the quality with five_point_crop being superior to the random crops.

The impact of the patch_size on the performance seems to prevail over the model size and the pre-training task.
## TODO TID2013 training 
- [x] 1) dataset for TID2013 and dataloader
 Rewrite Dataset to take dist_img and scores as pandas DataFrame to stratify train_test_split and make val and test Datasets
- [x] 2) validation on val set and test set 
- [x] 3) 5 train__test_splits  10 fold run through the training split (10  times train+predict test) = 50 times 
  - [x] 3.5) RUN training on full 80% and test on 20% - do this 5 times
- [x] 4) how to train new models so that we won't have memory problems ? 
- [ ] 5) Exepriments with different backbones - 2 splits - 5 folds
  - [x] 5.1) DEFAULT backbone - vit_base_patch8_224
  - [x] 5.2) vit_large_patch14_224_clip_laion2b  
  - [ ] 5.3) vit_base_patch8_224_dino 
  - [ ] 5.4) CLIP ViT - OpenCLIP
  - [ ] 5.5) ResNet
- [x] *) Write a logger that logs to a file AND prints to the console
 
