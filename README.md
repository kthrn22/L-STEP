# L-STEP

## Data Preprocessing

#### Datasets

13 datasets: Wikipedia, Reddit, MOOC, LastFM, Enron, Social Evo., UCI, Flights, Can. Parl., US Legis., UN Trade, UN Vote, and Contact. 

These datasets are adopted from [Towards Better Evaluation for Dynamic Link Prediction](https://openreview.net/forum?id=1GVpwr2Tfdg), which can be downloaded from [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o).

After downloading the datasets, please place them in the ```DG_data``` folder.

#### Preprocessing

For a dataset ```dataset_name```, run the following code to preprocess the dataset:

```python3
cd preprocess_data/
python preprocess_data.py --dataset_name [dataset_name]
```

For example, we preprocess the Enron dataset by running:

```python3
cd preprocess_data/
python preprocess_data.py --dataset_name enron
```

## Executing Scripts for running Temporal Link Prediction

#### Train L-STEP

In order to train L-STEP on dataset ```dataset_name```, run

```python3
python train_STEP_link_prediction.py --dataset_name [dataset_name] --model_name LSTEP --num_runs 5 --gpu [cuda index] --[other configs]
```

Here is an example of training L-STEP on *Enron* dataset:

```python3
python train_STEP_link_prediction.py --dataset_name enron --model_name LSTEP --num_runs 5 --gpu 0 --[other configs]
```

If you want to load the best configurations for *Enron* , run

```python3
python train_STEP_link_prediction.py --dataset_name enron --model_name LSTEP --num_runs 5 --gpu 0 --load_best_configs
```

#### Evaluate L-STEP with different negative sampling strategy (NSS)

We evaluate L-STEP on 3 NSS: random, historical, and inductive.

Here is an example of evaluating L-STEP on Enron with random NSS

```python3
python evaluate_LSTEP_link_prediction.py --dataset_name enron --model_name LSTEP --num_runs 5 --gpu 0 --negative_sample_strategy random --[other configs]
```

If you want to load the best configurations during the evaluation, run:

```python3
python evaluate_LSTEP_link_prediction.py --dataset_name enron --model_name LSTEP --num_runs 5 --gpu 0 --load_best_configs --negative_sample_strategy random --[other configs]
```

For historical NSS, set ```--negative_sample_strategy``` to ```historical```, and for inductive NSS, set ```--negative_sample_strategy``` to ```inductive```.

# Acknowledgments

We are grateful to the authors of [DyGFormer](https://github.com/yule-BUAA/DyGLib) for making their project codes publicly available.
