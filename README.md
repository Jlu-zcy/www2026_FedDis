FedDis: A Causal Disentanglement Framework for Federated Traffic Prediction

## Dependencies

Our framework is implemented in Python 3.12.3 and PyTorch 2.7.0.

You can use requirements.txt directly.
```bash
pip install -r requirements.txt
```

## Data
You can download data from [BasicTS]([https://github.com/GestaltCogTeam/BasicTS] and unzip it into the data directory.
```bash
python data/generate_training_data.py
```

## Usage
```bash
cd FedDis
```

Train the model.
```bash
python main.py --config configs/dataset.yaml
```

Evaluate the model.

The best model of each client is stored in experiments/dataset/runtime.

```bash
python evaluate.py --config configs/dataset.yaml --log_dir experiments/dataset/runtime
```
