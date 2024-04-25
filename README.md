# CVSITP
- **structure**:
  - Discover\
    - the codes for discovering insights
  - NN\
    - main codes for NN defination, training and tuning
  - Process\
    - the codes for processing data
- NN\
  - DataSet: *load the data*
  - DataLoader: *prepare the data for training*
  - NnStructure: *define the network*
  - NnTrain: *train the network*
  - HyperTuning: *tuning the network(using optuna)*
  - storageLinkTemplate: *the database URL template for optuna*
  - vit-model: *ViT*
  - BestBatchLr: *to find the best lr for each batch size*

# IMPORTANT
remember to set your **own** database URL in `storageLinkTemplate.txt` and then **change the name** of it to `storageLink.txt`, this file will be ignored in .gitignore to protect you database from leak