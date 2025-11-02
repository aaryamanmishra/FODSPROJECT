# Sorting and Search Algorithm Selector with ML and Rule-Based Logic

This project provides an intelligent system to **predict the best sorting algorithm using Machine Learning (ML)** based on array characteristics and then **selects the optimal search algorithm using a rule-based approach** considering the array and number of queries.

---

## Features

- **ML-based Sorting Algorithm Predictor**  
  Predicts the most efficient sorting algorithm for a given array based on its features: size, sortedness, uniqueness, and value range.  
  The ML model is trained using these features to classify among popular sorting algorithms.

- **Rule-Based Search Algorithm Selector**  
  Selects the optimal search algorithm given the array, its distribution, sortedness (wider definition for searching), and the number of queries to be performed.

- **Interactive CLI**  
  Users can input an array and number of queries to receive both sorting and search algorithm recommendations.

---

## Supported Algorithms

### Sorting Algorithms (ML classes)
- **Radix Sort**
- **Counting Sort**
- **Merge Sort**
- **Quick Sort**
- **Heap Sort**
- **Insertion Sort**
- **Selection Sort**
- **Bubble Sort**

### Searching Algorithms (Rule-based selection)
- **Linear Search** (unsorted arrays or fallback)
- **Sentinel Search** (optimized linear variant)
- **Binary Search** (sorted arrays)
- **Jump Search** (sorted arrays, good for medium size)
- **Interpolation Search** (sorted uniform data)
- **Exponential Search** (sorted arrays with unknown size)
- **Fibonacci Search** (sorted arrays, variation of binary search)
- **Ternary Search** (sorted arrays, splits into thirds)

---

## Machine Learning Overview

### Input Features
- **Array size**: Number of elements in the array
- **Sortedness**: Strictly non-decreasing measure
- **Uniqueness**: Ratio of unique to total elements
- **Range**: Maximum value minus minimum value

### Model
A supervised classification model (e.g., Random Forest or similar) trained to predict which sorting algorithm is most efficient given the input features.

---

## Time Complexity Comparison

### Sorting Algorithms

| Algorithm          | Average Case Time Complexity | Worst Case Time Complexity | Notes                                      |
|--------------------|------------------------------|----------------------------|--------------------------------------------|
| **Radix Sort**     | O(d(n + k))                  | O(d(n + k))                | d = digits, k = base                       |
| **Counting Sort** | O(n + k)                     | O(n + k)                   | Works on integers in range k               |
| **Merge Sort**     | O(n log n)                   | O(n log n)                 | Stable, divide-and-conquer                 |
| **Quick Sort**     | O(n log n)                   | O(n²)                      | Fast in practice, unstable                 |
| **Heap Sort**      | O(n log n)                   | O(n log n)                 | In-place, not stable                       |
| **Insertion Sort** | O(n²)                        | O(n²)                      | Efficient for small/mostly sorted arrays   |
| **Selection Sort** | O(n²)                        | O(n²)                      | Simple but inefficient                     |
| **Bubble Sort**    | O(n²)                        | O(n²)                      | Generally slow                             |

### Searching Algorithms

| Searching Algorithm      | Average Case Time Complexity | Worst Case Time Complexity | Requires Sorted Data? |
|-------------------------|------------------------------|----------------------------|-----------------------|
| **Linear Search**       | O(n)                         | O(n)                       | No                    |
| **Sentinel Search**     | O(n)                         | O(n)                       | Typically yes*        |
| **Binary Search**       | O(log n)                     | O(log n)                   | Yes                   |
| **Jump Search**         | O(√n)                        | O(√n)                      | Yes                   |
| **Interpolation Search**| O(log log n) (uniform) / O(n) (worst) | O(n)       | Yes                   |
| **Exponential Search**  | O(log n)                     | O(log n)                   | Yes                   |
| **Fibonacci Search**    | O(log n)                     | O(log n)                   | Yes                   |
| **Ternary Search**      | O(log₃ n)                    | O(log₃ n)                  | Yes                   |

\* Sentinel search optimizes linear search, often implemented on sorted or partially sorted data.

---

## Usage

### Prerequisites

1. **Ensure Python 3 environment with dependencies:**
   ```bash
   pip install pandas scikit-learn numpy
   ```

2. **Place the trained ML model file (`model.pkl`) in the working directory.**

### Running the Application

3. **Run the provided interactive script:**
   ```bash
   python predict.py
   ```
   or
   ```bash
   python script.py
   ```

4. **Input your array and number of queries when prompted.**

5. **See both sorting and search algorithm recommendations.**

---

## Future Work

- Adding support for approximate or probabilistic distributions
- Extending ML model to predict search algorithms or hybrid strategies
- Benchmarking on real-world datasets to refine heuristic thresholds
- Adding visualization of algorithm performance predictions

---

## Contact

For questions or contributions, please open an issue or pull request on the repository.

---

## Summary

This project integrates classical algorithm theory with machine learning techniques to build a practical hybrid selection system for sorting and searching tasks.

