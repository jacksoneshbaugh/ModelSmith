# ModelSmith
Create TensorFlow models from YAML config files.

## About
Created as a tool for a research project out of Lafayette College, ModelSmith is a useful tool to build networks quickly and efficiently.

## Setting Up a Model Config YAML File
There are five distinct sections in any model config YAML file:
- `data`
- `model`
- `layers`
- `training`
- `random_seed`

### 1. `data`
In `data`, configure the format of your data. Currently, only `csv` files are supported by ModelSmith.

To configure a csv dataset, set `type: csv` and provide the path to your dataset in `path`. Then, state the input and output columns in the CSV using the `inputs` and `outputs` fields. **Note**: the number of inputs and outputs should correspond with the geometry of your network—if there are two input columns and one output column, then there should be two input neurons and one output neuron in the model. Below, there is an example of this setup. 
```yaml
# Data configuration (currently, only csv is a valid input file type)
data:
  type: csv
  path: 'xor.csv'
  inputs:
    - Input1
    - Input2
  outputs:
    - Output
```

### 2. `model`
The `model` field is currently not very useful as only `Sequential` models are supported. Ensure this field is set to `sequential`. See an example below.

```yaml
# Model type (currently not very useful, would be used if support for FunctionalAPI
# is needed)
model: sequential
```

### 3. `layers`
In `layers`, configure the actual geometry of your model. All of these parameters are analogous to `keras` layer objects and their constructor parameters. The first layer should always be `type: Input`, and there should be a `shape` field determining the shape of the input layer. See an example below.

```yaml
# Network layer configuration
layers:
  - type: Input
    shape: [2]
  - type: Dense
    units: 8
    activation: 'relu'
    kernel_initializer: 'he_normal'
  - type: Dense
    units: 1
    activation: 'sigmoid'
```

### 4. `training`
In `training`, set training parameters. Again, these are word-for-word from the TensorFlow training workflow.

```yaml
# Configuration of the training algorithm to use
training:
  optimizer: 'adam'
  loss: 'binary_crossentropy'
  learning_rate: 0.001
  metrics: ['accuracy']
  batch_size: 8
  epochs: 1000
  validation_split: 0.2
```

### 5. `random_seed`
`random_seed` is used to set `numpy`'s pseudorandom number generator seed.

```yaml
# Configuration of the random number generator seed (for reproducibility)
random_seed: 38
```

## Training a Model
It's _super_ simple: just run the Python script and input the name of the config file when prompted.

```zsh
python3 modelsmith.py
```

## Future Features
- I hope to add a GUI that makes creating these YAML files even easier.