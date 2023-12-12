# Deep Modularity Networks (DMoN)
This project implements a deep learning model for graph clustering called Deep Modularity Networks (DMoN). This model is used to perform community detection in graphs, a problem of significant importance in network science.

# Dependencies 

You need to have the following python packages installed to run the project:

- scipy
- tensorflow
- matplotlib
- numpy
You can install these packages using pip:

`pip install -r requirements.txt`

# Usage 

## Data 

The input data should be a `.npz` file containing the graph adjacency matrix, node feature matrix, and node labels. The `load_npz` function in `train.py` is used to load this data.

## Training 


You can train the DMoN model using the `train.py` script. This script loads the input data, builds the DMoN model, and trains it using the Adam optimizer. The loss history is plotted and saved as `corafull3.png`. <br>
You can edit some configurations by changing these variables in `train.py`:
```
ARG_graph_path = "data/npz/cora_full.npz" # Path to the training graph
ARG_architecture = [64]; 
ARG_collapse_regularization = 1; 
ARG_dropout_rate = 0; 
ARG_n_clusters = 16; 
ARG_n_epochs = 1000; 
ARG_learning_rate = 0.01; 
```
You can also change where the loss history file will be saved by changing this line: 
```
plt.savefig('corafull3.png')
```
To start training: 
```
python train.py
```

The training will last 1000 epochs. 

# Evaluation 

The `train.py` script also evaluates the trained model using several metrics, including conductance, modularity, normalized mutual information (NMI), and F1 score. The results are printed to the console.

