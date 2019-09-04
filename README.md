# Fast Robustness Prediction for Deep Neural Network
## Prerequisite
The code shoule be run using Python 2.7.12, Tensorflow 1.12.0, Keras 2.2.4, 
Numpy 1.16.4, Sklearn 0.20.3.

The code use data computed from the [C&W](https://github.com/carlini/nn_robust_attacks)
and [SA](https://github.com/coinse/sadl).

## File Description
`data/` contains the `.csv` files and the `.npy` file for the experiments.

`index_lenet1.npy` and `index_lenet1_test.npy` contains the 10000 
training examples index and 1000 test examples index in the MNIST dataset
that LeNet-1 model can classify correctly.

`confidence_test.csv` and `confidence_train.csv` are the confidence values 
of 1000 test examples and 10000 training examples respectively.

`dsa_lenet1.csv` contains the [Distance-based Surprise Adequacy (DSA)](https://arxiv.org/pdf/1808.08444.pdf) 
values of 1000 test examples for the DSA regression model testing. 
`lsa_lenet1.csv` is similar for  [Likelihood-based Surprise Adequacy (LSA)](https://arxiv.org/pdf/1808.08444.pdf)
regression model.

`dsa_lenet1_sa.csv` contains the 5000 DSA values for DSA regression model
training. `robust_lenet1_sa.csv` and `lsa_lenet1_sa.csv` are both similar.

`.model/` contains the original model to be tested and the trained model
to predict the robustness values.
## To run
Get the penultimate layer output:
```angular2
python layer.py
```

Train the regression model for DNN robustness prediction:
```angular2
python train_layer_regress.py
```

See the results of the trained model:
```angular2
python layer_regress.py
```

Train and see the results of confidence, LSA and DSA regression:
```angular2
python train_confidence_regress.py
python train_lsa_regress.py
python train_dsa_regress.py
```

## Note
The code shows the process for LeNet-1 model and robustness prediction for
other models is the same as it.