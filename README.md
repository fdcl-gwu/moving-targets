# Regression with constraints through affine extension function

Submitted to NIPS

## Datasets

Datasets are in the folder
`resources/{dataset_name}`

## Example

Our results can be obtained by using the following type of code

```
python run.py --dataset crime --mtype fairness --implement cplex --loss mse --algo affine --ltype gb --alpha 0.5 --beta 0.1 --iterations 30
```

## Parameters for regression with fairness constraints

* dataset = {'crime', 'student', 'blackfriday'}
* mtype = {'fairness'}
* implement = {'cplex', 'cvxpy'}
* loss = {'mse', 'mae', 'mhl'}
* algo = {'movtar', 'affine'}
* ltype = {'lr', 'gb', 'nn'}
* alpha
* beta
* iterations
* nfolds = number of folds (default=5)
