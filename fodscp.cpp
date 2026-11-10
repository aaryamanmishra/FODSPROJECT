// fodscp.cpp
// Compile: g++ -std=c++17 -O2 fodscp.cpp -o algo_recommender
// Run: ./algo_recommender
//
// Project: Sorting & Searching Algorithm Recommendation System
// Single-file implementation: implements 8 sorting and 8 searching algorithms,
// benchmarks them on user-selected dataset, and recommends the best one(s).

#include <bits/stdc++.h>
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif
using namespace std;
using ll = long long;
using Clock = chrono::high_resolution_clock;
using us = chrono::microseconds;

struct SortStats {
    string name;
    long long time_us;
    unsigned long long comparisons;
    unsigned long long swaps;
    bool success; // for algorithms that might fail due to constraints
};

struct SearchStats {
    string name;
    long long time_us;
    unsigned long long comparisons;
    bool found;
    bool success;
};

// ---------- ML feature extraction & utilities ----------
// Forward declarations for algorithms/utilities defined later in file
void bubbleSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp);
void insertionSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp);
void mergeSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp);
void quickSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp);
void heapSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp);
void print_sep(int w);
vector<int> gen_random(int n, int lo, int hi);
vector<int> gen_sorted(int n, int lo, int hi);
vector<int> gen_reversed(int n, int lo, int hi);
vector<int> gen_many_duplicates(int n, int unique_count, int lo, int hi);
struct Features {
    long long size;
    double sortednessRatio;   // fraction of adjacent pairs in non-decreasing order
    double uniquenessRatio;   // unique_count / size
    long long valueRange;     // max - min
};

vector<int> read_dataset_txt(const string &relativePath) {
    ifstream fin(relativePath);
    vector<int> data;
    if (!fin) return data;
    string content;
    getline(fin, content, '\0');
    fin.close();
    int sign = 1;
    long long val = 0;
    bool inNum = false;
    for (char c : content) {
        if (c == '-') { sign = -1; }
        else if (isdigit((unsigned char)c)) { val = val * 10 + (c - '0'); inNum = true; }
        else {
            if (inNum) { data.push_back((int)(val * sign)); val = 0; sign = 1; inNum = false; }
        }
    }
    if (inNum) { data.push_back((int)(val * sign)); }
    return data;
}

Features compute_features(const vector<int>& a) {
    Features f{0, 0.0, 0.0, 0};
    f.size = (long long)a.size();
    if (a.empty()) return f;
    // sortedness: adjacent non-decreasing
    long long inOrder = 0;
    for (size_t i = 1; i < a.size(); ++i) if (a[i-1] <= a[i]) ++inOrder;
    f.sortednessRatio = (double)inOrder / (double)max<size_t>(1, a.size()-1);
    // uniqueness
    unordered_set<int> s(a.begin(), a.end());
    f.uniquenessRatio = (double)s.size() / (double)a.size();
    // range
    int mn = *min_element(a.begin(), a.end());
    int mx = *max_element(a.begin(), a.end());
    f.valueRange = (long long)mx - (long long)mn;
    return f;
}

void ensure_results_dir() {
#ifdef _WIN32
    _mkdir("results");
#else
    mkdir("results", 0755);
#endif
}

bool write_csv_rows(const string& csvPath, const vector<array<string,5>>& rows, bool writeHeader) {
    ofstream fout(csvPath, ios::out | ios::trunc);
    if (!fout) return false;
    if (writeHeader) fout << "size,sortedness,uniqueness,range,best_algorithm\n";
    for (const auto& r : rows) {
        fout << r[0] << ',' << r[1] << ',' << r[2] << ',' << r[3] << ',' << r[4] << "\n";
    }
    return true;
}

// subset benchmark for exactly 5 algorithms required
vector<SortStats> run_core_sorts(const vector<int>& data) {
    vector<SortStats> results;
    vector<pair<string, function<void(vector<int>&, unsigned long long&, unsigned long long&)>>> sorts;
    sorts.push_back({"Bubble",    [](vector<int>& a, unsigned long long &c, unsigned long long &s){ bubbleSort(a, c, s); }});
    sorts.push_back({"Insertion", [](vector<int>& a, unsigned long long &c, unsigned long long &s){ insertionSort(a, c, s); }});
    sorts.push_back({"Merge",     [](vector<int>& a, unsigned long long &c, unsigned long long &s){ mergeSort(a, c, s); }});
    sorts.push_back({"Quick",     [](vector<int>& a, unsigned long long &c, unsigned long long &s){ quickSort(a, c, s); }});
    sorts.push_back({"Heap",      [](vector<int>& a, unsigned long long &c, unsigned long long &s){ heapSort(a, c, s); }});
    for (auto &p : sorts) {
        vector<int> arr = data;
        unsigned long long cmp = 0, swp = 0;
        auto start = Clock::now();
        p.second(arr, cmp, swp);
        auto stop = Clock::now();
        long long t = chrono::duration_cast<us>(stop - start).count();
        bool sorted = true; for (size_t i=1;i<arr.size();++i) if (arr[i-1]>arr[i]) { sorted=false; break; }
        results.push_back({p.first, t, cmp, swp, sorted});
    }
    return results;
}

SortStats best_of_core(const vector<SortStats>& stats) {
    SortStats best = {"<none>", LLONG_MAX, ULLONG_MAX, ULLONG_MAX, false};
    for (auto &s : stats) {
        if (!s.success) continue;
        if (!best.success || s.time_us < best.time_us ||
            (s.time_us == best.time_us && s.comparisons < best.comparisons) ||
            (s.time_us == best.time_us && s.comparisons == best.comparisons && s.swaps < s.swaps)) {
            best = s;
        }
    }
    return best;
}

void print_core_sort_table(const vector<SortStats>& stats) {
    cout << left << setw(12) << "Algorithm" << setw(12) << "Time(us)" << setw(15) << "Comparisons" << setw(12) << "Swaps" << "Status\n";
    print_sep(60);
    for (auto &s : stats) {
        cout << left << setw(12) << s.name
             << setw(12) << s.time_us
             << setw(15) << s.comparisons
             << setw(12) << s.swaps
             << (s.success ? "OK" : "FAILED") << "\n";
    }
}

string run_python_predict(const Features& f) {
    // Use system() to write prediction into a temp file; add retry for Windows file locks
    ensure_results_dir();
    // unique temp file
    unsigned int r = (unsigned int)rand();
    string outPath = string("results") + string(
#ifdef _WIN32
        "\\pred_"
#else
        "/pred_"
#endif
    ) + to_string((unsigned long long)time(nullptr)) + "_" + to_string(r) + ".txt";
    stringstream ss;
    ss << "python predict.py"
       << " --size " << f.size
       << " --sortedness " << fixed << setprecision(6) << f.sortednessRatio
       << " --uniqueness " << fixed << setprecision(6) << f.uniquenessRatio
       << " --range " << f.valueRange
       << " > \"" << outPath << "\" 2>&1";
    string cmd = ss.str();
    int rc = system(cmd.c_str());
    (void)rc;
    // retry open
    string result;
    for (int attempt = 0; attempt < 20; ++attempt) {
        ifstream fin(outPath);
        if (fin) {
            getline(fin, result);
            fin.close();
            break;
        }
#ifdef _WIN32
        Sleep(50);
#else
        this_thread::sleep_for(chrono::milliseconds(50));
#endif
    }
    // trim
    while (!result.empty() && (result.back()=='\n' || result.back()=='\r' || isspace((unsigned char)result.back()))) result.pop_back();
    return result;
}

string run_python_predict_search(const Features& f, int num_queries, int distribution = 0) {
    // distribution: 0=random, 1=uniform, 2=clustered
    int is_sorted = (f.sortednessRatio > 0.95) ? 1 : 0;
    
    ensure_results_dir();
    unsigned int r = (unsigned int)rand();
    string outPath = string("results") + string(
#ifdef _WIN32
        "\\pred_search_"
#else
        "/pred_search_"
#endif
    ) + to_string((unsigned long long)time(nullptr)) + "_" + to_string(r) + ".txt";
    stringstream ss;
    ss << "python predict_search.py"
       << " --size " << f.size
       << " --data_range " << f.valueRange
       << " --is_sorted " << is_sorted
       << " --distribution " << distribution
       << " --num_queries " << num_queries
       << " > \"" << outPath << "\" 2>&1";
    string cmd = ss.str();
    int rc = system(cmd.c_str());
    (void)rc;
    // retry open
    string result;
    for (int attempt = 0; attempt < 20; ++attempt) {
        ifstream fin(outPath);
        if (fin) {
            getline(fin, result);
            fin.close();
            break;
        }
#ifdef _WIN32
        Sleep(50);
#else
        this_thread::sleep_for(chrono::milliseconds(50));
#endif
    }
    // trim
    while (!result.empty() && (result.back()=='\n' || result.back()=='\r' || isspace((unsigned char)result.back()))) result.pop_back();
    return result;
}

void generate_training_csv(const string& outCsv, int numDatasets = 50) {
    ensure_results_dir();
    vector<array<string,5>> rows;
    rows.reserve(numDatasets);
    random_device rd; mt19937 rng(rd());
    uniform_int_distribution<int> sizeDist(200, 5000);
    uniform_int_distribution<int> typeDist(1, 4); // 1=random,2=sorted,3=reverse,4=dups
    uniform_int_distribution<int> loDist(-100000, 0);
    uniform_int_distribution<int> hiDist(1, 100000);
    for (int i = 0; i < numDatasets; ++i) {
        int n = sizeDist(rng);
        int t = typeDist(rng);
        int lo = loDist(rng);
        int hi = max(lo + 10, lo + hiDist(rng));
        vector<int> data;
        if (t == 1) data = gen_random(n, lo, hi);
        else if (t == 2) data = gen_sorted(n, lo, hi);
        else if (t == 3) data = gen_reversed(n, lo, hi);
        else data = gen_many_duplicates(n, 5 + (i%10), lo, lo + 100);
        Features f = compute_features(data);
        auto stats = run_core_sorts(data);
        auto best = best_of_core(stats);
        array<string,5> row = {
            to_string(f.size),
            (static_cast<ostringstream&>(ostringstream() << fixed << setprecision(6) << f.sortednessRatio)).str(),
            (static_cast<ostringstream&>(ostringstream() << fixed << setprecision(6) << f.uniquenessRatio)).str(),
            to_string(f.valueRange),
            best.name
        };
        rows.push_back(row);
    }
    write_csv_rows(outCsv, rows, true);
}

// Utility: print a separator line
void print_sep(int w = 80) {
    for (int i = 0; i < w; ++i) cout << '-';
    cout << '\n';
}

// ---------- Dataset generation ----------
vector<int> gen_random(int n, int lo=0, int hi=1000000) {
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<int> dist(lo, hi);
    vector<int> a(n);
    for (int i = 0; i < n; ++i) a[i] = dist(rng);
    return a;
}
vector<int> gen_sorted(int n, int lo=0, int hi=1000000) {
    auto a = gen_random(n, lo, hi);
    sort(a.begin(), a.end());
    return a;
}
vector<int> gen_reversed(int n, int lo=0, int hi=1000000) {
    auto a = gen_sorted(n, lo, hi);
    reverse(a.begin(), a.end());
    return a;
}
vector<int> gen_many_duplicates(int n, int unique_count=5, int lo=0, int hi=100) {
    // create many duplicates by sampling from a small range
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<int> dist(lo, min(hi, lo + unique_count - 1));
    vector<int> a(n);
    for (int i = 0; i < n; ++i) a[i] = dist(rng);
    return a;
}

// ---------- Helpers for cloning and timing ----------
template<typename F, typename... Args>
long long time_fn(F f, Args&&... args) {
    auto start = Clock::now();
    f(forward<Args>(args)...);
    auto stop = Clock::now();
    return chrono::duration_cast<us>(stop - start).count();
}

// ---------- Sorting algorithms (all operate on vector<int>&, update comparisons/swaps) ----------

// Bubble Sort
void bubbleSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp) {
    cmp = swp = 0;
    int n = (int)a.size();
    for (int i = 0; i < n-1; ++i)
        for (int j = 0; j < n-i-1; ++j) {
            ++cmp;
            if (a[j] > a[j+1]) {
                swap(a[j], a[j+1]);
                ++swp;
            }
        }
}

// Selection Sort
void selectionSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp) {
    cmp = swp = 0;
    int n = (int)a.size();
    for (int i = 0; i < n-1; ++i) {
        int minidx = i;
        for (int j = i+1; j < n; ++j) {
            ++cmp;
            if (a[j] < a[minidx]) minidx = j;
        }
        if (minidx != i) { swap(a[i], a[minidx]); ++swp; }
    }
}

// Insertion Sort
void insertionSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp) {
    cmp = swp = 0;
    int n = (int)a.size();
    for (int i = 1; i < n; ++i) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0) {
            ++cmp;
            if (a[j] > key) {
                a[j+1] = a[j];
                ++swp;
                --j;
            } else break;
        }
        a[j+1] = key;
    }
}

// Merge Sort (count comparisons; swaps = 0 because merge uses copies)
void mergeSortRec(vector<int>& a, int l, int r, unsigned long long &cmp) {
    if (l >= r) return;
    int m = l + (r-l)/2;
    mergeSortRec(a, l, m, cmp);
    mergeSortRec(a, m+1, r, cmp);
    vector<int> tmp;
    tmp.reserve(r-l+1);
    int i = l, j = m+1;
    while (i <= m && j <= r) {
        ++cmp;
        if (a[i] <= a[j]) tmp.push_back(a[i++]);
        else tmp.push_back(a[j++]);
    }
    while (i <= m) tmp.push_back(a[i++]);
    while (j <= r) tmp.push_back(a[j++]);
    for (int k = l; k <= r; ++k) a[k] = tmp[k-l];
}
void mergeSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp) {
    cmp = swp = 0;
    mergeSortRec(a, 0, (int)a.size()-1, cmp);
}

// Quick Sort (Lomuto) with random pivot to reduce worst-case
int quickPartition(vector<int>& a, int lo, int hi, unsigned long long &cmp, unsigned long long &swp) {
    int pivot = a[hi];
    int i = lo - 1;
    for (int j = lo; j <= hi-1; ++j) {
        ++cmp;
        if (a[j] <= pivot) {
            ++i;
            swap(a[i], a[j]);
            ++swp;
        }
    }
    swap(a[i+1], a[hi]);
    ++swp;
    return i+1;
}
void quickSortRec(vector<int>& a, int lo, int hi, unsigned long long &cmp, unsigned long long &swp) {
    if (lo < hi) {
        // randomize pivot by swapping a random index with hi
        int r = lo + rand() % (hi-lo+1);
        swap(a[r], a[hi]); ++swp;
        int p = quickPartition(a, lo, hi, cmp, swp);
        quickSortRec(a, lo, p-1, cmp, swp);
        quickSortRec(a, p+1, hi, cmp, swp);
    }
}
void quickSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp) {
    cmp = swp = 0;
    quickSortRec(a, 0, (int)a.size()-1, cmp, swp);
}

// Heap Sort
void heapify(vector<int>& a, int n, int i, unsigned long long &cmp, unsigned long long &swp) {
    int largest = i;
    int l = 2*i + 1, r = 2*i + 2;
    if (l < n) { ++cmp; if (a[l] > a[largest]) largest = l; }
    if (r < n) { ++cmp; if (a[r] > a[largest]) largest = r; }
    if (largest != i) {
        swap(a[i], a[largest]); ++swp;
        heapify(a, n, largest, cmp, swp);
    }
}
void heapSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp) {
    cmp = swp = 0;
    int n = (int)a.size();
    for (int i = n/2 - 1; i >= 0; --i) heapify(a, n, i, cmp, swp);
    for (int i = n-1; i > 0; --i) {
        swap(a[0], a[i]); ++swp;
        heapify(a, i, 0, cmp, swp);
    }
}

// Counting Sort (works well when range is not huge)
bool countingSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp) {
    cmp = swp = 0;
    if (a.empty()) return true;
    int mn = *min_element(a.begin(), a.end());
    int mx = *max_element(a.begin(), a.end());
    long long range = (long long)mx - (long long)mn + 1;
    if (range > 20000000) { // too large to allocate safely
        return false;
    }
    vector<long long> cnt((size_t)range);
    for (int v : a) cnt[v - mn]++;
    int idx = 0;
    for (long long i = 0; i < range; ++i) {
        while (cnt[i]-- > 0) {
            a[idx++] = (int)(i + mn);
            ++swp; // consider each placed element a "swap"/write
        }
    }
    return true;
}

// Radix Sort (LSD) for integers (handles negatives via offset)
bool radixSort(vector<int>& a, unsigned long long &cmp, unsigned long long &swp) {
    cmp = swp = 0;
    if (a.empty()) return true;
    int mn = *min_element(a.begin(), a.end());
    int mx = *max_element(a.begin(), a.end());
    // shift to non-negative
    long long offset = (mn < 0) ? - (long long)mn : 0;
    vector<unsigned int> b(a.size());
    for (size_t i = 0; i < a.size(); ++i) b[i] = (unsigned int)((long long)a[i] + offset);

    unsigned int maxv = *max_element(b.begin(), b.end());
    if (maxv > 0x7FFFFFFFu && b.size() > 0) {
        // very large numbers — still can sort but will be slower; allow it
    }
    // do radix base 256 (bytes)
    vector<unsigned int> tmp(b.size());
    const int base = 256;
    for (int shift = 0; shift < 32; shift += 8) {
        vector<int> cnt(base, 0);
        for (size_t i = 0; i < b.size(); ++i) {
            cnt[(b[i] >> shift) & 0xFF]++;
        }
        for (int i = 1; i < base; ++i) cnt[i] += cnt[i-1];
        for (int i = (int)b.size()-1; i >= 0; --i) {
            tmp[--cnt[(b[i] >> shift) & 0xFF]] = b[i];
        }
        b.swap(tmp);
    }
    // write back
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = (int)((long long)b[i] - offset);
        ++swp;
    }
    return true;
}

// ---------- Searching algorithms ----------

// Linear Search
SearchStats linearSearch(const vector<int>& a, int key) {
    SearchStats s{"Linear Search", 0, 0, false, true};
    auto start = Clock::now();
    for (size_t i = 0; i < a.size(); ++i) {
        ++s.comparisons;
        if (a[i] == key) { s.found = true; break; }
    }
    s.time_us = chrono::duration_cast<us>(Clock::now() - start).count();
    return s;
}

// Binary Search (iterative) — requires sorted
SearchStats binarySearch(const vector<int>& a, int key) {
    SearchStats s{"Binary Search", 0, 0, false, true};
    auto start = Clock::now();
    int l = 0, r = (int)a.size() - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        ++s.comparisons;
        if (a[m] == key) { s.found = true; break; }
        ++s.comparisons;
        if (a[m] < key) l = m + 1;
        else r = m - 1;
    }
    s.time_us = chrono::duration_cast<us>(Clock::now() - start).count();
    return s;
}

// Jump Search (works on sorted)
SearchStats jumpSearch(const vector<int>& a, int key) {
    SearchStats s{"Jump Search", 0, 0, false, true};
    auto start = Clock::now();
    int n = (int)a.size();
    if (n == 0) { s.time_us = chrono::duration_cast<us>(Clock::now() - start).count(); return s; }
    int step = max(1, (int)sqrt(n));
    int prev = 0;
    while (prev < n) {
        ++s.comparisons;
        if (a[min(n-1, prev)] >= key) break;
        prev += step;
    }
    int l = max(0, prev - step + 1);
    int r = min(n-1, prev);
    for (int i = l; i <= r; ++i) {
        ++s.comparisons;
        if (a[i] == key) { s.found = true; break; }
    }
    s.time_us = chrono::duration_cast<us>(Clock::now() - start).count();
    return s;
}

// Interpolation Search (works on sorted with roughly uniform distribution)
SearchStats interpolationSearch(const vector<int>& a, int key) {
    SearchStats s{"Interpolation Search", 0, 0, false, true};
    auto start = Clock::now();
    int lo = 0, hi = (int)a.size() - 1;
    while (lo <= hi && key >= a[lo] && key <= a[hi]) {
        if (a[lo] == a[hi]) { ++s.comparisons; if (a[lo] == key) s.found = true; break; }
        long long pos = lo + ( (long long)(hi - lo) * (key - a[lo]) ) / (a[hi] - a[lo]);
        if (pos < lo || pos > hi) break;
        ++s.comparisons;
        if (a[pos] == key) { s.found = true; break; }
        ++s.comparisons;
        if (a[pos] < key) lo = (int)pos + 1;
        else hi = (int)pos - 1;
    }
    s.time_us = chrono::duration_cast<us>(Clock::now() - start).count();
    return s;
}

// Exponential Search (requires sorted) — find range then binary search
SearchStats exponentialSearch(const vector<int>& a, int key) {
    SearchStats s{"Exponential Search", 0, 0, false, true};
    auto start = Clock::now();
    int n = (int)a.size();
    if (n == 0) { s.time_us = chrono::duration_cast<us>(Clock::now() - start).count(); return s; }
    ++s.comparisons;
    if (a[0] == key) { s.found = true; s.time_us = chrono::duration_cast<us>(Clock::now() - start).count(); return s; }
    int bound = 1;
    while (bound < n) {
        ++s.comparisons;
        if (a[bound] >= key) break;
        bound <<= 1;
    }
    int l = bound/2;
    int r = min(bound, n-1);
    // binary on [l,r]
    while (l <= r) {
        int m = l + (r-l)/2;
        ++s.comparisons;
        if (a[m] == key) { s.found = true; break; }
        ++s.comparisons;
        if (a[m] < key) l = m+1;
        else r = m-1;
    }
    s.time_us = chrono::duration_cast<us>(Clock::now() - start).count();
    return s;
}

// Ternary Search (on sorted for unimodal or just to find exact element — here used to locate equality)
SearchStats ternarySearch(const vector<int>& a, int key) {
    SearchStats s{"Ternary Search", 0, 0, false, true};
    auto start = Clock::now();
    int l = 0, r = (int)a.size() - 1;
    while (l <= r) {
        int mid1 = l + (r-l)/3;
        int mid2 = r - (r-l)/3;
        ++s.comparisons;
        if (a[mid1] == key) { s.found = true; break; }
        ++s.comparisons;
        if (a[mid2] == key) { s.found = true; break; }
        ++s.comparisons;
        if (key < a[mid1]) r = mid1 - 1;
        else if (key > a[mid2]) l = mid2 + 1;
        else { l = mid1 + 1; r = mid2 - 1; }
    }
    s.time_us = chrono::duration_cast<us>(Clock::now() - start).count();
    return s;
}

// Fibonacci Search (sorted)
SearchStats fibonacciSearch(const vector<int>& a, int key) {
    SearchStats s{"Fibonacci Search", 0, 0, false, true};
    auto start = Clock::now();
    int n = (int)a.size();
    int fibMm2 = 0;
    int fibMm1 = 1;
    int fibM = fibMm1 + fibMm2;
    while (fibM < n) {
        fibMm2 = fibMm1;
        fibMm1 = fibM;
        fibM = fibMm1 + fibMm2;
    }
    int offset = -1;
    while (fibM > 1) {
        int i = min(offset + fibMm2, n - 1);
        ++s.comparisons;
        if (a[i] < key) {
            fibM = fibMm1;
            fibMm1 = fibMm2;
            fibMm2 = fibM - fibMm1;
            offset = i;
        } else {
            ++s.comparisons;
            if (a[i] > key) {
                fibM = fibMm2;
                fibMm1 -= fibMm2;
                fibMm2 = fibM - fibMm1;
            } else {
                s.found = true;
                break;
            }
        }
    }
    if (!s.found && fibMm1 && offset + 1 < n) {
        ++s.comparisons;
        if (a[offset+1] == key) s.found = true;
    }
    s.time_us = chrono::duration_cast<us>(Clock::now() - start).count();
    return s;
}

// Hash-based Search: build unordered_set and lookup
SearchStats hashSearch(const vector<int>& a, int key) {
    SearchStats s{"Hash Search (unordered_set)", 0, 0, false, true};
    auto start = Clock::now();
    unordered_set<int> st;
    st.reserve(a.size()*2 + 10);
    for (int v : a) st.insert(v);
    s.comparisons = 1; // approximation: one hashed lookup comparison
    s.found = (st.find(key) != st.end());
    s.time_us = chrono::duration_cast<us>(Clock::now() - start).count();
    return s;
}

// ---------- Running and collecting stats ----------

vector<SortStats> run_all_sorts(const vector<int>& data) {
    vector<SortStats> results;
    vector<pair<string, function<bool(vector<int>&, unsigned long long&, unsigned long long&)>>> sorts;

    // Add functions: the boolean indicates success (true) or not (false)
    sorts.push_back({"Counting Sort", [](vector<int>& a, unsigned long long &cmp, unsigned long long &swp){ return countingSort(a, cmp, swp); }});
    sorts.push_back({"Radix Sort",   [](vector<int>& a, unsigned long long &cmp, unsigned long long &swp){ return radixSort(a, cmp, swp); }});
    sorts.push_back({"Merge Sort",   [](vector<int>& a, unsigned long long &cmp, unsigned long long &swp){ mergeSort(a, cmp, swp); return true; }});
    sorts.push_back({"Quick Sort",   [](vector<int>& a, unsigned long long &cmp, unsigned long long &swp){ quickSort(a, cmp, swp); return true; }});
    sorts.push_back({"Heap Sort",    [](vector<int>& a, unsigned long long &cmp, unsigned long long &swp){ heapSort(a, cmp, swp); return true; }});
    sorts.push_back({"Insertion Sort",[](vector<int>& a, unsigned long long &cmp, unsigned long long &swp){ insertionSort(a, cmp, swp); return true; }});
    sorts.push_back({"Selection Sort",[](vector<int>& a, unsigned long long &cmp, unsigned long long &swp){ selectionSort(a, cmp, swp); return true; }});
    sorts.push_back({"Bubble Sort",  [](vector<int>& a, unsigned long long &cmp, unsigned long long &swp){ bubbleSort(a, cmp, swp); return true; }});

    for (auto &p : sorts) {
        string name = p.first;
        vector<int> arr = data; // fresh copy
        unsigned long long cmp = 0, swp = 0;
        auto start = Clock::now();
        bool ok = p.second(arr, cmp, swp);
        auto stop = Clock::now();
        long long t = chrono::duration_cast<us>(stop - start).count();
        // Validate sortedness if ok
        bool sorted = true;
        if (ok) {
            for (size_t i = 1; i < arr.size(); ++i) if (arr[i-1] > arr[i]) { sorted = false; break; }
            if (!sorted) ok = false;
        }
        results.push_back({name, t, cmp, swp, ok});
    }
    return results;
}

vector<SearchStats> run_all_searches(const vector<int>& data, int key, bool assume_sorted) {
    vector<SearchStats> results;
    // We'll run linear and hash on original data; others require sortedness
    results.push_back(linearSearch(data, key));
    results.push_back(hashSearch(data, key));

    if (assume_sorted) {
        results.push_back(binarySearch(data, key));
        results.push_back(jumpSearch(data, key));
        results.push_back(interpolationSearch(data, key));
        results.push_back(exponentialSearch(data, key));
        results.push_back(ternarySearch(data, key));
        results.push_back(fibonacciSearch(data, key));
    } else {
        // If not sorted, still attempt but mark success=false for those needing sortedness
        vector<int> data_sorted = data;
        sort(data_sorted.begin(), data_sorted.end());
        results.push_back(SearchStats{"Binary Search (on sorted copy)", 0,0,false,false});
        results.back() = binarySearch(data_sorted, key);
        results.back().name += " [on sorted copy]";
        results.push_back(jumpSearch(data_sorted, key)); results.back().name += " [on sorted copy]";
        results.push_back(interpolationSearch(data_sorted, key)); results.back().name += " [on sorted copy]";
        results.push_back(exponentialSearch(data_sorted, key)); results.back().name += " [on sorted copy]";
        results.push_back(ternarySearch(data_sorted, key)); results.back().name += " [on sorted copy]";
        results.push_back(fibonacciSearch(data_sorted, key)); results.back().name += " [on sorted copy]";
    }
    return results;
}

// ---------- Theoretical time complexity calculations ----------
struct ComplexityCost {
    string algorithm_name;
    double complexity_value;  // approximate "cost" based on Big-O
};

// Calculate theoretical time complexity for sorting algorithms
double calculate_sort_complexity(const string& algo_name, long long n) {
    string name_lower = algo_name;
    transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
    
    if (name_lower.find("counting") != string::npos) {
        return n;  // O(n)
    } else if (name_lower.find("radix") != string::npos) {
        return n;  // O(n)
    } else if (name_lower.find("merge") != string::npos) {
        return n * log2(max(1LL, n));  // O(n log n)
    } else if (name_lower.find("quick") != string::npos) {
        return n * log2(max(1LL, n));  // O(n log n) average
    } else if (name_lower.find("heap") != string::npos) {
        return n * log2(max(1LL, n));  // O(n log n)
    } else if (name_lower.find("insertion") != string::npos) {
        return n * n;  // O(n^2)
    } else if (name_lower.find("selection") != string::npos) {
        return n * n;  // O(n^2)
    } else if (name_lower.find("bubble") != string::npos) {
        return n * n;  // O(n^2)
    }
    return 1e18;  // unknown
}

// Calculate theoretical time complexity for searching algorithms (single query)
// For single query, hash search is O(n) to build hash table + O(1) lookup = O(n)
double calculate_search_complexity(const string& algo_name, long long n) {
    if (algo_name.find("Hash") != string::npos || algo_name.find("hash") != string::npos) {
        return n;  // O(n) for single query (build hash table cost)
    } else if (algo_name.find("Binary") != string::npos) {
        return log2(max(1LL, n));  // O(log n)
    } else if (algo_name.find("Interpolation") != string::npos) {
        // Average case O(log log n), but we use a small constant for simplicity
        double loglogn = log2(max(2LL, n));
        return log2(max(2.0, loglogn));  // O(log log n) average
    } else if (algo_name.find("Exponential") != string::npos) {
        return log2(max(1LL, n));  // O(log n)
    } else if (algo_name.find("Jump") != string::npos) {
        return sqrt(n);  // O(sqrt(n))
    } else if (algo_name.find("Ternary") != string::npos) {
        return log2(max(1LL, n)) / log2(3);  // O(log_3 n) ≈ O(log n)
    } else if (algo_name.find("Fibonacci") != string::npos) {
        return log2(max(1LL, n));  // O(log n)
    } else if (algo_name.find("Linear") != string::npos) {
        return n / 2.0;  // O(n) average case (search until found, average n/2)
    }
    return 1e18;  // unknown
}

// Find best sorting algorithm based on theoretical complexity
string find_best_sort_by_complexity(long long n, long long range) {
    vector<pair<string, double>> sort_complexities;
    
    // Consider all sorting algorithms
    sort_complexities.push_back({"Bubble Sort", calculate_sort_complexity("Bubble", n)});
    sort_complexities.push_back({"Selection Sort", calculate_sort_complexity("Selection", n)});
    sort_complexities.push_back({"Insertion Sort", calculate_sort_complexity("Insertion", n)});
    sort_complexities.push_back({"Merge Sort", calculate_sort_complexity("Merge", n)});
    sort_complexities.push_back({"Quick Sort", calculate_sort_complexity("Quick", n)});
    sort_complexities.push_back({"Heap Sort", calculate_sort_complexity("Heap", n)});
    
    // Counting and Radix only if range is reasonable
    if (range > 0 && range < 20000000) {
        sort_complexities.push_back({"Counting Sort", calculate_sort_complexity("Counting", n)});
        sort_complexities.push_back({"Radix Sort", calculate_sort_complexity("Radix", n)});
    }
    
    // Find minimum complexity
    string best = sort_complexities[0].first;
    double min_cost = sort_complexities[0].second;
    for (auto &p : sort_complexities) {
        if (p.second < min_cost) {
            min_cost = p.second;
            best = p.first;
        }
    }
    
    return best;
}

// ---------- Recommendation logic ----------
SortStats recommend_sort(const vector<SortStats>& stats) {
    // pick fastest by time among successful algorithms; tie-breaker comparisons then swaps
    SortStats best = {"<none>", LLONG_MAX, ULLONG_MAX, ULLONG_MAX, false};
    for (auto &s: stats) {
        if (!s.success) continue;
        if (!best.success || s.time_us < best.time_us ||
            (s.time_us == best.time_us && s.comparisons < best.comparisons) ||
            (s.time_us == best.time_us && s.comparisons == best.comparisons && s.swaps < best.swaps)) {
            best = s;
        }
    }
    return best;
}

SearchStats recommend_search(const vector<SearchStats>& stats) {
    // pick fastest (time) among successful algorithms; prefer true found ones? Here choose fastest which also found key if any found.
    bool anyFound = false;
    for (auto &s: stats) if (s.found) anyFound = true;
    SearchStats best = {"<none>", LLONG_MAX, ULLONG_MAX, false, false};
    for (auto &s: stats) {
        if (!s.success) continue;
        if (anyFound && !s.found) continue; // if any algorithm found, prefer only those that found
        if (!best.success || s.time_us < best.time_us ||
            (s.time_us == best.time_us && s.comparisons < best.comparisons)) {
            best = s;
        }
    }
    return best;
}

// ---------- Pretty printing ----------
void print_sort_table(const vector<SortStats>& stats) {
    cout << left << setw(20) << "Algorithm" << setw(12) << "Time(us)" << setw(15) << "Comparisons" << setw(12) << "Swaps" << "Status\n";
    print_sep(70);
    for (auto &s : stats) {
        cout << left << setw(20) << s.name
             << setw(12) << s.time_us
             << setw(15) << s.comparisons
             << setw(12) << s.swaps
             << (s.success ? "OK" : "SKIPPED/FAILED") << "\n";
    }
}

void print_search_table(const vector<SearchStats>& stats) {
    cout << left << setw(35) << "Algorithm" << setw(12) << "Time(us)" << setw(15) << "Comparisons" << setw(10) << "Found" << "Status\n";
    print_sep(85);
    for (auto &s : stats) {
        cout << left << setw(35) << s.name
             << setw(12) << s.time_us
             << setw(15) << s.comparisons
             << setw(10) << (s.found ? "Yes" : "No")
             << (s.success ? "OK" : "SKIPPED/FAILED") << "\n";
    }
}

// ---------- Main interactive flow ----------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    srand((unsigned)time(nullptr));

    // Flag-based ML extension
    if (argc > 1) {
        string flag = argv[1];
        if (flag == "--train") {
            cout << "Generating training CSV (50 datasets) at results/dataset_features.csv ...\n";
            generate_training_csv("results/dataset_features.csv", 50);
            cout << "Done. Now run: python train_model.py\n";
            return 0;
        } else if (flag == "--ml") {
            cout << "Reading dataset from dataset.txt ...\n";
            vector<int> data = read_dataset_txt("dataset.txt");
            if (data.empty()) {
                cerr << "dataset.txt is empty or missing. Provide comma-separated integers.\n";
                return 1;
            }
            Features f = compute_features(data);
            cout << "Feature Summary:\n";
            cout << "- Size: " << f.size << "\n";
            cout << "- Sortedness ratio: " << fixed << setprecision(6) << f.sortednessRatio << "\n";
            cout << "- Uniqueness ratio: " << fixed << setprecision(6) << f.uniquenessRatio << "\n";
            cout << "- Range: " << f.valueRange << "\n";

            // For sorting: Use ML prediction (no algorithm execution)
            cout << "\nQuerying ML model for sorting algorithm (predict.py) ...\n";
            string sort_pred = run_python_predict(f);
            string best_sort_algo;
            
            // Check if prediction is valid (not an error message)
            bool valid_pred = !sort_pred.empty() && 
                             sort_pred.find("can't") == string::npos &&
                             sort_pred.find("error") == string::npos &&
                             sort_pred.find("Error") == string::npos &&
                             sort_pred.find("Errno") == string::npos &&
                             sort_pred.find("python:") == string::npos;
            
            if (valid_pred) {
                // Capitalize first letter to match our algorithm names
                if (!sort_pred.empty() && islower(sort_pred[0])) {
                    sort_pred[0] = toupper(sort_pred[0]);
                }
                // Add " Sort" suffix if not present
                if (sort_pred.find("Sort") == string::npos) {
                    sort_pred += " Sort";
                }
                best_sort_algo = sort_pred;
                cout << "Recommended Sorting Algorithm: " << best_sort_algo << "\n";
            } else {
                cout << "ML prediction unavailable, using theoretical complexity analysis...\n";
                best_sort_algo = find_best_sort_by_complexity(f.size, f.valueRange);
                cout << "Recommended Sorting Algorithm (theoretical): " << best_sort_algo << "\n";
            }

            // For searching: Check if sorted
            cout << "\n--- Searching Algorithm Recommendation ---\n";
            bool is_sorted = (f.sortednessRatio > 0.95);
            
            if (is_sorted) {
                // If sorted, directly recommend Binary Search
                cout << "Array is sorted (sortedness > 0.95).\n";
                cout << "Recommended Searching Algorithm: Binary Search\n";
                cout << "Reason: Sorted arrays benefit from O(log n) binary search.\n";
            } else {
                // If unsorted, calculate theoretical complexity for each search algorithm
                // including the cost of sorting first
                cout << "Array is unsorted (sortedness <= 0.95).\n";
                cout << "Calculating theoretical time complexity for search algorithms...\n\n";
                
                // Remove " Sort" suffix if present for complexity calculation
                string sort_algo_key = best_sort_algo;
                if (sort_algo_key.find(" Sort") != string::npos) {
                    sort_algo_key = sort_algo_key.substr(0, sort_algo_key.find(" Sort"));
                }
                
                double sort_complexity = calculate_sort_complexity(sort_algo_key, f.size);
                
                // List of searching algorithms to consider
                vector<string> search_algos = {
                    "Linear Search",
                    "Hash Search",
                    "Binary Search",
                    "Jump Search",
                    "Interpolation Search",
                    "Exponential Search",
                    "Ternary Search",
                    "Fibonacci Search"
                };
                
                vector<ComplexityCost> total_costs;
                
                for (const string& search_algo : search_algos) {
                    // For algorithms that need sorted array, add sorting cost
                    bool needs_sorted = (search_algo.find("Linear") == string::npos && 
                                        search_algo.find("Hash") == string::npos);
                    
                    double search_complexity = calculate_search_complexity(search_algo, f.size);
                    double total_cost = search_complexity;
                    
                    if (needs_sorted) {
                        total_cost = sort_complexity + search_complexity;
                    }
                    
                    total_costs.push_back({search_algo, total_cost});
                }
                
                // Find minimum total cost
                ComplexityCost best_search = total_costs[0];
                for (const auto& cost : total_costs) {
                    if (cost.complexity_value < best_search.complexity_value) {
                        best_search = cost;
                    }
                }
                
                // Display results
                cout << "Best sorting algorithm to use: " << best_sort_algo << "\n";
                string sort_complexity_str;
                if (sort_complexity < f.size * 1.5) {
                    sort_complexity_str = "O(n)";
                } else if (sort_complexity < f.size * f.size * 0.1) {
                    sort_complexity_str = "O(n log n)";
                } else {
                    sort_complexity_str = "O(n²)";
                }
                cout << "Sorting complexity: " << sort_complexity_str << "\n\n";
                
                cout << "Search Algorithm Complexity Analysis (including sort cost if needed):\n";
                cout << left << setw(25) << "Algorithm" << setw(20) << "Total Complexity Cost" << "\n";
                print_sep(50);
                for (const auto& cost : total_costs) {
                    bool needs_sorted = (cost.algorithm_name.find("Linear") == string::npos && 
                                        cost.algorithm_name.find("Hash") == string::npos);
                    cout << left << setw(25) << cost.algorithm_name;
                    cout << setw(20) << fixed << setprecision(2) << cost.complexity_value;
                    if (needs_sorted) cout << " (includes sort cost)";
                    cout << "\n";
                }
                
                cout << "\nRecommended Searching Algorithm: " << best_search.algorithm_name << "\n";
                cout << "Reason: Lowest total time complexity (";
                cout << fixed << setprecision(2) << best_search.complexity_value << ")\n";
            }
            
            return 0;
        }
    }

    cout << "==== Sorting & Searching Algorithm Recommendation System (C++) ====\n\n";
    cout << "Choose dataset type:\n";
    cout << "1) Random\n2) Sorted\n3) Reverse sorted\n4) Many duplicates\nEnter choice (1-4): ";
    int choice;
    if (!(cin >> choice)) return 0;
    cout << "Enter dataset size n (e.g., 1000): ";
    int n; cin >> n;
    if (n < 0) { cerr << "Invalid n\n"; return 0; }
    cout << "Enter min value (e.g., 0): "; int lo; cin >> lo;
    cout << "Enter max value (e.g., 1000000): "; int hi; cin >> hi;

    vector<int> data;
    if (choice == 1) data = gen_random(n, lo, hi);
    else if (choice == 2) data = gen_sorted(n, lo, hi);
    else if (choice == 3) data = gen_reversed(n, lo, hi);
    else data = gen_many_duplicates(n, 5, lo, hi);

    cout << "Dataset generated. Sample (first 15 elements):\n";
    for (int i = 0; i < min(n, 15); ++i) cout << data[i] << (i+1==min(n,15) ? "\n" : " ");
    if (n > 15) cout << "...\n";

    // Ask for a search key
    cout << "\nEnter an integer key to search for (or type 'r' to pick a random element): ";
    string key_in;
    cin >> key_in;
    int key;
    if (key_in == "r") {
        if (!data.empty()) {
            random_device rd; mt19937 rng(rd());
            uniform_int_distribution<int> distIndex(0, (int)data.size()-1);
            key = data[distIndex(rng)];
            cout << "Random key chosen from dataset: " << key << "\n";
        } else {
            key = 0;
            cout << "Dataset empty; key=0\n";
        }
    } else {
        key = stoi(key_in);
    }

    // Determine if data is already sorted
    bool isSortedFlag = std::is_sorted(data.begin(), data.end());

    cout << "\nRunning sorts (each algorithm gets an independent copy of data)...\n";
    auto sort_results = run_all_sorts(data);

    cout << "\nRunning searches...\n";
    auto search_results = run_all_searches(data, key, isSortedFlag);

    cout << "\n\n--- Sorting Results ---\n";
    print_sort_table(sort_results);
    auto best_sort = recommend_sort(sort_results);
    if (best_sort.success) {
        cout << "\nRecommended sorting algorithm: " << best_sort.name << "\n";
        cout << "Reason: fastest measured time (" << best_sort.time_us << " us)";
        if (best_sort.name.find("Counting") != string::npos || best_sort.name.find("Radix") != string::npos) {
            cout << " — note: works best for integer datasets and limited ranges.";
        }
        cout << "\n";
    } else {
        cout << "\nNo successful sorting algorithm found (likely range issues for counting/radix or other failures).\n";
    }

    cout << "\n--- Searching Results ---\n";
    print_search_table(search_results);
    auto best_search = recommend_search(search_results);
    if (best_search.success) {
        cout << "\nRecommended searching algorithm: " << best_search.name << "\n";
        cout << "Reason: fastest measured time (" << best_search.time_us << " us)";
        if (best_search.name.find("Binary")!=string::npos || best_search.name.find("Interpolation")!=string::npos ||
            best_search.name.find("Fibonacci")!=string::npos) {
            cout << " — note: requires sorted array to be efficient.";
        }
        cout << "\n";
        if (!best_search.found) cout << "Note: recommended algorithm did NOT find the key; check dataset or key.\n";
    } else {
        cout << "\nNo successful searching algorithm found.\n";
    }

    // Extra tips based on data characteristics
    cout << "\n--- Quick Tips based on dataset ---\n";
    if (n <= 1000) cout << "- Small dataset: simple sorts (Insertion/Selection) may be acceptable.\n";
    if (n > 1000) cout << "- Large dataset: prefer Quick/Merge/Heap.\n";
    if (!isSortedFlag) cout << "- Array is not sorted: searching algorithms that need sorted input will operate on a sorted copy (cost not measured here).\n";
    if (choice == 4) cout << "- Dataset has many duplicates: Counting/Radix/Stable linear-time sorts may perform extremely well.\n";
    cout << "\nProgram finished. You can run again with different dataset parameters.\n";
    return 0;
}
