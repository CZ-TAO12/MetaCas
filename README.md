# MetaCas

A PyTorch implementation of our MetaCas.

## Dependencies
Install the dependencies via [Anaconda](https://www.anaconda.com/):
+ Python (>=3.8)
+ PyTorch (>=1.8.1)
+ NumPy (>=1.17.4)
+ Scipy (>=1.7.3)
+ torch-geometric(>=2.0.4)
+ tqdm(>=4.62.2)


create virtual environment:
```
conda create --name MetaCas python=3.8
```

activate environment:
```
conda activate MetaCas
```

install pytorch from [pytorch](https://pytorch.org/get-started/previous-versions/):
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
```

To install all dependencies:
```
pip install -r requirements.txt
```

## Usage
Here we provide the implementation of MetaCas along with twitter dataset.

+ To generate cascade attributes:
```
python cas_attribute.py
```

+ To train and evaluate on Twitter:
```
python run.py -data_name=twitter
```
More running options are described in the codes, e.g., `-data_name= twitter`

## Folder Structure

MetaCas
```
└── data: # The file includes datasets
    ├── twitter
       ├── cascades.txt       # original data
       ├── cascadetrain.txt   # training set
       ├── cascadevalid.txt   # validation set
       ├── cascadetest.txt    # testing data
       ├── edges.txt          # social network
       ├── idx2u.pickle       # idx to user_id
       ├── u2idx.pickle       # user_id to idx
       
└── models: # The file includes each part of the modules in MetaCas.
    ├── Meta_GNN.py # The core source code of Meta_GNN.
    ├── MetaLSTM.py # The core source code of MetaLSTM.
    ├── TransformerBlock.py # The core source code of time-aware attention.

└── utils: # The file includes each part of basic modules (e.g., metrics, earlystopping).
    ├── EarlyStopping.py  # The core code of the early stopping operation.
    ├── Metrics.py        # The core source code of metrics.
    ├── graphConstruct.py # The core source code of building social network.
    ├── parsers.py        # The core source code of parameter settings. 
└── Constants.py:     
└── cas_attribute.py:  # The file includes the core source code of constructing cascade attributes.
└── dataLoader.py:     # Data loading.
└── run.py:            # Run the model.
└── Optim.py:          # Optimization.

```
