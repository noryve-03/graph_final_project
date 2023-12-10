import scipy.sparse
from typing import Tuple
import tensorflow as tf
from absl import app
import sklearn.metrics
import numpy as np
from scipy.sparse import base
import dmon
import gcn
import metrics
import utils
import matplotlib.pyplot as plt 

ARG_graph_path = "data/npz/ms_academic_phy.npz"
ARG_architecture = [64]; 
ARG_collapse_regularization = 1; 
ARG_dropout_rate = 0; 
ARG_n_clusters = 16; 
ARG_n_epochs = 1000; 
ARG_learning_rate = 0.01; 

def load_npz(
    filename
):
  """Loads an attributed graph with sparse features from a specified Numpy file.

  Args:
    filename: A valid file name of a numpy file containing the input data.

  Returns:
    A tuple (graph, features, labels, label_indices) with the sparse adjacency
    matrix of a graph, sparse feature matrix, dense label array, and dense label
    index array (indices of nodes that have the labels in the label array).
  """
  with np.load(open(filename, 'rb'), allow_pickle=True) as loader:
    loader = dict(loader)
    adjacency = scipy.sparse.csr_matrix(
        (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
        shape=loader['adj_shape'])

    features = scipy.sparse.csr_matrix(
        (loader['attr_data'], loader['attr_indices'],
         loader['attr_indptr']),
        shape=loader['attr_shape'])

    
    labels = loader['labels']
    label_indices =  [i for i in range(len(labels))] 
    label_indices = np.asarray(label_indices)
  assert adjacency.shape[0] == features.shape[
      0], 'Adjacency and feature size must be equal!'
  assert labels.shape[0] == label_indices.shape[
      0], 'Labels and label_indices size must be equal!'
  return adjacency, features, labels, label_indices


def convert_scipy_sparse_to_sparse_tensor(
    matrix):
  """Converts a sparse matrix and converts it to Tensorflow SparseTensor.

  Args:
    matrix: A scipy sparse matrix.

  Returns:
    A ternsorflow sparse matrix (rank-2 tensor).
  """
  matrix = matrix.tocoo()
  return tf.sparse.SparseTensor(
      np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
      matrix.shape)


def build_dmon(input_features,
               input_graph,
               input_adjacency):
  """Builds a Deep Modularity Network (DMoN) model from the Keras inputs.

  Args:
    input_features: A dense [n, d] Keras input for the node features.
    input_graph: A sparse [n, n] Keras input for the normalized graph.
    input_adjacency: A sparse [n, n] Keras input for the graph adjacency.

  Returns:
    Built Keras DMoN model.
  """
  output = input_features
  for n_channels in ARG_architecture:
    output = gcn.GCN(n_channels)([output, input_graph])
  pool, pool_assignment = dmon.DMoN(
      ARG_n_clusters,
      collapse_regularization=ARG_collapse_regularization,
      dropout_rate=ARG_dropout_rate)([output, input_adjacency])
  return tf.keras.Model(
      inputs=[input_features, input_graph, input_adjacency],
      outputs=[pool, pool_assignment])


def main(argv):
  # Load and process the data (convert node features to dense, normalize the
  # graph, convert it to Tensorflow sparse tensor.
  adjacency, features, labels, label_indices = load_npz(ARG_graph_path)
  features = features.todense()
  n_nodes = adjacency.shape[0]
  feature_size = features.shape[1]
  graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
  graph_normalized = convert_scipy_sparse_to_sparse_tensor(
      utils.normalize_graph(adjacency.copy()))

  # Create model input placeholders of appropriate size
  input_features = tf.keras.layers.Input(shape=(feature_size,))
  input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
  input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

  model = build_dmon(input_features, input_graph, input_adjacency)

  # Computes the gradients wrt. the sum of losses, returns a list of them.
  def grad(model, inputs):
    with tf.GradientTape() as tape:
      _ = model(inputs, training=True)
      loss_value = sum(model.losses)
    return model.losses, tape.gradient(loss_value, model.trainable_variables)

  optimizer = tf.keras.optimizers.Adam(ARG_learning_rate)
  model.compile(optimizer, None)
  loss_history = []
  for epoch in range(ARG_n_epochs):
    loss_values, grads = grad(model, [features, graph_normalized, graph])
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f'epoch {epoch}, losses: ' +
          ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values]))
    loss_history.append(loss_values[0])
  # Obtain the cluster assignments.
  _, assignments = model([features, graph_normalized, graph], training=False)
  assignments = assignments.numpy()
  clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
  plt.figure(figsize=(10, 7))
  plt.plot(loss_history, color='green', label='Loss')  # Change color here
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.title('Training Loss Over Epochs')
  plt.grid(False)  # Add grid lines
  plt.legend()
  plt.savefig('training_loss_plot_4.png')
  plt.show()  # Prints some metrics used in the paper.
  print('Conductance:', metrics.conductance(adjacency, clusters))
  print('Modularity:', metrics.modularity(adjacency, clusters))
  print(
      'NMI:',
      sklearn.metrics.normalized_mutual_info_score(
          labels, clusters[label_indices], average_method='arithmetic'))
  precision = metrics.pairwise_precision(labels, clusters[label_indices])
  recall = metrics.pairwise_recall(labels, clusters[label_indices])
  print('F1:', 2 * precision * recall / (precision + recall))


if __name__ == '__main__':
  app.run(main)
