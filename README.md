# MANIQA_reproducing
Trying to reproduce results of MANIQA paper

## TO DO TID2013 training 
- [x] 1) dataset for TID2013 and dataloader
 Rewrite Dataset to take dist_img and scores as pandas DataFrame to stratify train_test_split and make val and test Datasets
- [x] 2) validation on val set and test set 
- [ ] 3) list of seeds 
5 train__test_splits  10 fold run through the training split (10  times train+predict test) = 50 times 
  - [ ] 3.5) RUN training on full 80% and test on 20% - do this 5 times
- [ ] 4) how to train new models so that we won't have memory problems ? 
- [ ] 5) Connect new Github token but write it this time somewhere !
- [ ] 6) Write a logger that logs to a file and prints to the console
 
