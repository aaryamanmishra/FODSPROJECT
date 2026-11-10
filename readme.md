# Sorting & Searching Algorithm Recommendation System

An interactive C++ project that benchmarks sorting and searching algorithms on generated integer datasets and recommends suitable approaches using dataset characteristics. The repository also contains an experimental Python notebook for an ML-based sorting recommendation path.

## Repository contents

| File | Purpose |
|---|---|
| `fodscp.cpp` | Main C++ interactive benchmark and recommendation program |
| `predict2.ipynb` | Experimental Python feature extraction and ML prediction notebook |
| `dataset.txt` | Small sample integer dataset used by the C++ ML mode |
| `balanced_sorting_dataset_16k.csv` | Dataset for sorting-algorithm experiments |

## Run the C++ program

### Requirements

- A C++17-compatible compiler (for example, GCC or Clang).

### Build

```bash
g++ -std=c++17 -O2 fodscp.cpp -o algo_recommender
```

### Run

```bash
./algo_recommender
```

The interactive program asks you to choose a dataset pattern (random, sorted, reverse-sorted, or many duplicates), enter the dataset size and value range, and provide a search key. It then runs the available benchmarks and prints the comparison results and recommendations.

On Windows, compile with a C++17-capable compiler and run the resulting executable from the repository directory.

## Experimental Python / ML path

The notebook `predict2.ipynb` contains feature calculations and a sorting-algorithm prediction function. It loads a trained model from `model.pkl`.

**The model-training script and the `model.pkl` artifact are not currently included in this repository.** The C++ `--train` / `--ml` extension also refers to Python helper scripts that are not present in the checked-in file list. Treat these ML paths as incomplete until the missing training and prediction files are added.

The standalone C++ interactive benchmark does not require the Python model.

## Features explored

- Dataset characteristics such as size, sortedness, uniqueness, and value range.
- Sorting and searching algorithm comparison.
- Rule-based search selection that considers whether the data is sorted and the expected number of queries.
- Experimental ML-based sorting recommendation.

## Current limitations

- The Python notebook expects a pre-trained `model.pkl` that is not committed.
- The C++ ML extension references `train_model.py` and `predict.py`, which are not present in the repository.
- Benchmark timings depend on the machine, compiler, optimization flags, and generated data. Re-run benchmarks in your own environment before drawing performance conclusions.

## Contributing

Issues and pull requests are welcome. Please include compiler details and sample input when reporting C++ benchmark problems.

## License

No `LICENSE` file is currently included in this repository. Until a license is added, the code does not have an explicit open-source reuse license.
