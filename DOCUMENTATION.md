---
permalink: /
---

# Data Witches Project 2 - Complete Documentation

## Project Overview

This project implements an Atrial Fibrillation classification system using ECG (Electrocardiogram) data and machine learning techniques. The project analyzes heart rate variability (HRV) features extracted from ECG signals to distinguish between Normal Sinus Rhythm and Atrial Fibrillation.

### Team Members

| Name | Student ID |
|------|------------|
| Claessen, VVHJAE | i6339543 |
| Ovsiannikova, AM | i6365923 |
| Pubben, J | i6276134 |
| Roca Cugat, M | i6351071 |
| Záboj, J | i6337952 |

---

## Dataset Information

### Primary Dataset
- **Source**: PhysioNet 2017 Training Data
- **File**: `data/Physionet2017Training.tar.xz`
- **Format**: CSV files containing ECG signals and labels
- **Classes**: 
  - Normal Sinus Rhythm (N) → encoded as 0
  - Atrial Fibrillation (A) → encoded as 1

### Data Files
- `Physionet2017TrainingData.csv`: Raw ECG signal data (converted to mV by multiplying by 1000)
- `Physionet2017TrainingLabels.csv`: Classification labels
- `train.csv`: Secondary dataset for testing ML pipeline
- `hrv_train.csv`: Preprocessed HRV features for training set (72 features, 3,619 samples)
- `hrv_test.csv`: Preprocessed HRV features for test set (72 features, 905 samples)

### Data Split
- Training set: 80%
- Test set: 20%
- Stratification: Based on class labels to maintain balance
- Random state: 3003

---

## Package Dependencies

### Core Libraries
```
kaggle              # Dataset downloading
plotly              # Interactive visualizations
matplotlib          # Static plotting
numpy               # Numerical computations
seaborn             # Statistical visualizations
pandas              # Data manipulation
colorama            # Terminal text coloring
scikit-learn        # Machine learning algorithms
shap                # Model interpretability
statsmodels         # Statistical modeling
neurokit2           # ECG signal processing and HRV feature extraction
tqdm                # Progress bars for long-running operations
sympy               # Symbolic mathematics (dependency for other libraries)
```

### Specific Imports

#### Data Processing
- `pandas` - DataFrame operations
- `numpy` - Array operations
- `scipy.stats` - Statistical functions
- `scipy.signal.welch` - Frequency analysis

#### Visualization
- `matplotlib.pyplot` - Basic plotting
- `seaborn` - Statistical plots
- `plotly.graph_objects` - Interactive plots
- `plotly.subplots` - Multi-panel plots

#### Machine Learning
- `sklearn.model_selection` - train_test_split, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
- `sklearn.linear_model.LogisticRegression` - Logistic regression classifier
- `sklearn.ensemble.RandomForestClassifier` - Random forest classifier
- `sklearn.ensemble.GradientBoostingClassifier` - Gradient boosting classifier
- `sklearn.ensemble.AdaBoostClassifier` - AdaBoost boosting classifier
- `sklearn.ensemble.VotingClassifier` - Ensemble voting classifier
- `sklearn.neighbors.KNeighborsClassifier` - K-nearest neighbors classifier
- `sklearn.neighbors.RadiusNeighborsClassifier` - Radius neighbors classifier
- `sklearn.neighbors.NearestCentroid` - Nearest centroid classifier
- `sklearn.neural_network.MLPClassifier` - Multi-layer perceptron neural network
- `sklearn.preprocessing` - MinMaxScaler, StandardScaler, RobustScaler
- `sklearn.impute.SimpleImputer` - Missing value imputation
- `sklearn.metrics` - classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, RocCurveDisplay, precision_recall_curve

---

## Main Notebook Structure (MAI3003_DataWitches_Assignment02.ipynb)

The main notebook consists of **193 cells** organized into the following major sections:

### 1. Introduction and Setup
- **Cell 0**: Colab badge link
- **Cell 1**: Project title
- **Cell 2**: Team member information
- **Cell 3**: Logbook for version tracking
- **Cell 5**: Methods overview
- **Cell 6**: Preface
- **Cell 7**: Package imports
- **Cell 10**: Options settings (pandas display, warnings)
- **Cell 12**: Dataset download instructions

### 2. Data Preprocessing (Cell 15-19)
- **Cell 15**: Dataset loading and preprocessing overview
- **Cell 14**: Load ECG data from CSV
- **Cell 16**: Load labels and encode classifications
- **Cell 18**: Dataset splitting section
- **Cell 19**: Train-test split implementation

### 3. Function Definitions (Cell 20-28)
- **Cell 20**: Function definitions overview
- **Cell 21-22**: Correlation plot function
- **Cell 23-24**: Distribution plot functions
- **Cell 25-26**: Boxplot function
- **Cell 27-28**: Missingness check function

### 4. Exploratory Data Analysis (Cell 29-69)
- **Cell 29**: EDA overview and dataset characteristics
- **Cell 34**: ECG signal exploration
- **Cell 36**: ECG frequency analysis
- **Cell 40**: Synthetic data decision
- **Cell 42**: Powerline interference analysis
- **Cell 47**: Baseline wander analysis
- **Cell 51**: Muscle noise analysis
- **Cell 54**: ECG feature engineering
- **Cell 56**: Feature extraction using NeuroKit
- **Cell 58**: R-peaks detection
- **Cell 60**: Time-domain features
- **Cell 63**: Full HRV feature extraction (TRAIN)
- **Cell 66**: Full HRV feature extraction (TEST)
- **Cell 68**: Feature exploration

### 5. Visualizations (Cell 70-79)
- **Cell 70**: Visualizations overview
- **Cell 71**: Correlation plot
- **Cell 73**: Distribution plots
- **Cell 76**: Boxplots
- **Cell 78**: Missingness visualization

### 6. Outlier Detection and Handling (Cell 80-95)
- **Cell 80**: Outlier detection overview
- **Cell 81**: Outlier identification function
- **Cell 85**: Outliers in TEST set
- **Cell 88**: Distribution comparison (TRAIN + TEST)
- **Cell 91**: Outlier handling for TRAIN
- **Cell 92**: Winsorising outliers
- **Cell 94**: Outlier handling comparison

### 7. Final Preprocessing (Cell 96-109)
- **Cell 96**: Building ML matrices overview
- **Cell 97**: Feature selection and target separation
- **Cell 99**: Imputation section
- **Cell 100**: Missing value imputation implementation
- **Cell 101**: Scaling section
- **Cell 102**: Robust scaling implementation
- **Cell 103**: Sanity checks
- **Cell 104-105**: Verification of preprocessing
- **Cell 106**: Final ML datasets overview
- **Cell 109**: Final dataset confirmation

### 8. Machine Learning Training Setup (Cell 110-116)
- **Cell 110**: ML setup overview
- **Cell 111**: Safety dataset for testing
- **Cell 112**: Safety check section
- **Cell 113**: Assert statements for data integrity
- **Cell 114**: Comparison framework
- **Cell 115**: Model results function
- **Cell 116**: Dataset size confirmation

### 9. Machine Learning Training (Cells 117-178)
Model implementations are organized by algorithm type:

**Logistic Regression (Cells 118-119):**
- Baseline Logistic Regression with all features
- Top 10 Features variant (selected by feature importance)
- Correlation-based feature selection (> 0.8)
- Augmented features with all parent features
- Augmented features with dropped parent features

**Random Forest (Cells 121-135):**
- 14 variants with varying max_depth (1-14)

**K-Nearest Neighbors (Cells 136-150):**
- 15 variants with hyperparameter tuning (n_neighbors 1-15)

**Radius Neighbors (Cells 151-158):**
- 8 variants with various radius values (0.5-10.0)

**Other Classifiers (Cells 159-173):**
- Nearest Centroid Classifier (Cell 159)
- Multi-Layer Perceptron Neural Network (Cell 160)
- Gradient Boosting Classifier baseline (Cell 161)
- AdaBoost with varying n_estimators 50-10000 (Cells 162-171)
- Voting Classifier ensemble (Cell 172)
- Gradient Boosting with tuned hyperparameters (Cell 173)

### 10. Results (Cell 179-193)
- **Cell 180-190**: Results comparison and visualization
- **Cell 191-193**: Final analysis and conclusions

---

## Challenge Notebook Structure (DataWitches_Challenge.ipynb)

The challenge notebook consists of **141 cells** (58 markdown, 83 code cells) organized into the following sections:

### 1. Introduction and Setup (Cells 0-12)
- **Cell 0**: Project title ("Data Witches")
- **Cell 2**: Logbook
- **Cell 3**: Methods overview with variable and function naming conventions
- **Cell 4**: Preface
- **Cell 5**: Package imports
- **Cell 7**: Options settings
- **Cell 9**: Dataset download instructions

### 2. Data Preprocessing (Cells 13-21)
- **Cell 13**: Data preprocessing overview and ECG signal extraction
- **Cell 16**: Dataset splitting
- **Cell 18**: Exploratory Data Analysis and dataset characteristics

### 3. Feature Extraction (Cells 22-43)
- **Cell 22**: Feature extraction overview
- **Cell 23**: ECG feature engineering
- **Cell 25**: R-peaks detection
- **Cell 27**: Time-domain features
- **Cell 30**: Full HRV feature extraction for TRAIN set
- **Cell 38**: Full HRV feature extraction for TEST set
- **Cell 42**: Feature exploration

### 4. Preprocessing (Cells 44-77)
- **Cell 44**: Preprocessing overview
- **Cell 45**: Missingness analysis
- **Cell 48**: Outlier detection (IQR method)
- **Cell 53**: Outliers in TEST set
- **Cell 56**: Distribution sanity check (TRAIN vs TEST)
- **Cell 59-62**: Outlier handling (Winsorisation)
- **Cell 64**: Final preprocessing (building ML matrices)
- **Cell 67**: Imputation
- **Cell 69**: Normalisation
- **Cell 71**: Scaling
- **Cell 73**: Sanity checks
- **Cell 75**: Final ML datasets (X_train, X_test, y_train, y_test)

### 5. Machine Learning Training Setup (Cells 78-83)
- **Cell 78**: ML training setup overview
- **Cell 79**: Safety check with assertions
- **Cell 81**: Comparison framework and model results function

### 6. Machine Learning Training (Cells 84-130)
Model implementations are organized by algorithm type:

**Logistic Regression (Cells 85-100):**
- Baseline Logistic Regression with all features
- Feature selection variants
- Correlation-based feature removal (>0.8)
- Feature engineering with normalisation
- "Use Everything" model (augmented features with parents)
- "Drop Parents" model (augmented features only)

**Random Forest (Cells 101-104):**
- Random Forest with various configurations

**Neighbors Classifiers (Cells 105-110):**
- K-Nearest Neighbors (Cell 105-106)
- Radius Neighbors Classifier (Cell 107-108)
- Nearest Centroid Classifier (Cell 109-110)

**Neural Network (Cells 111-115):**
- Multi-Layer Perceptron Classifier

**Ensemble Methods (Cells 116-130):**
- Gradient Boosting Classifier (Cell 116-118)
- AdaBoost Classifier (Cell 119-120)
- Soft Voting Classifier (Cell 121-123): Combines LR, RF (n_estimators=300), KNN (k=7), and AdaBoost (n_estimators=1000)
- Gradient-boosted trees variants (Cell 124-126)
- Experimental voting classifiers (Cell 127-128)
- **Best Voting Classifier Search (Cell 129-131)**: Automated search through all combinations of classifiers to find optimal ensemble

### 7. Model Evaluation (Cells 132-140)
- **Cell 132**: Quick conclusion section
- **Cell 133-135**: Results table analysis, duplicate removal, top 5 models display
- **Cell 136-138**: Graphs of numerical metrics (logarithmic and linear scale bar charts)
- **Cell 139-140**: ROC Curves comparison for all models

### Key Features of Challenge Notebook
- **Automated Ensemble Search**: Systematically tests all 2-4 classifier combinations from a pool of 10 base classifiers
- **Comprehensive Classifier Sweep**: Tests 15+ different classifier types including:
  - Logistic Regression (balanced and default)
  - Ridge Classifier
  - Decision Tree
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - AdaBoost
  - Histogram Gradient Boosting
  - SVC (with probability)
  - Gaussian Naive Bayes
  - Bernoulli Naive Bayes
  - K-Nearest Neighbors
  - Linear Discriminant Analysis
  - Quadratic Discriminant Analysis
  - Multi-Layer Perceptron
- **Visual Model Comparison**: ROC curves and metric bar charts for all tested models
- **Consistent Methodology**: Uses same HRV features, preprocessing, and evaluation metrics as main notebook

---

## Custom Functions

### 1. `corr_plot_hrv(df, cols=None)`
**Purpose**: Creates a correlation heatmap for HRV (Heart Rate Variability) features

**Parameters**:
- `df` (DataFrame): Input dataframe containing HRV features
- `cols` (list, optional): Specific columns to plot. If None, uses all numeric columns

**Functionality**:
- Selects numeric columns from the dataframe
- Creates a 12x8 figure
- Generates a heatmap using seaborn with 'coolwarm' colormap
- Centers the color scale at 0
- Displays correlation matrix between all features

**Usage**: Identifying relationships between different HRV features to detect multicollinearity

---

### 2. `distplots_hrv(df, cols=None)`
**Purpose**: Creates individual distribution plots (histogram + KDE) for each HRV feature

**Parameters**:
- `df` (DataFrame): Input dataframe containing HRV features
- `cols` (list, optional): Specific columns to plot. If None, uses all numeric columns

**Functionality**:
- Iterates through each column
- Creates a 6x4 figure for each feature
- Plots histogram with KDE overlay using seaborn
- Shows distribution shape and density

**Usage**: Understanding the distribution characteristics of individual features

---

### 3. `distplots(df)`
**Purpose**: Creates a comprehensive multi-panel distribution plot for all numeric features

**Parameters**:
- `df` (DataFrame): Input dataframe containing features

**Functionality**:
- Selects all numeric columns
- Calculates optimal subplot grid (rows × columns)
- Creates histogram for each feature
- Overlays Gaussian KDE density estimate
- Automatically handles subplot layout
- Removes empty subplots if needed

**Usage**: Getting a quick overview of all feature distributions simultaneously

---

### 4. `boxplots_hrv(df, cols)`
**Purpose**: Creates boxplots to detect outliers and unusual values in HRV features

**Parameters**:
- `df` (DataFrame): Input dataframe
- `cols` (list): List of column names to plot

**Functionality**:
- Creates individual 6x4 boxplot for each specified column
- Uses seaborn for visualization
- Shows median, quartiles, and outliers

**Usage**: Visual identification of outliers and understanding feature spread

---

### 5. `check_missing_hrv(df)`
**Purpose**: Summarizes and displays missing value information across all HRV features

**Parameters**:
- `df` (DataFrame): Input dataframe to check

**Returns**:
- DataFrame with columns: 'feature', 'missing_n', 'missing_%'

**Functionality**:
- Counts missing values per column
- Calculates percentage of missing values
- Sorts results by missing percentage (descending)
- Displays formatted output

**Usage**: Assessing data quality and planning imputation strategy

---

### 6. `identify_outliers(df, column_name, threshold=1.5)`
**Purpose**: Identifies outliers using the IQR (Interquartile Range) method

**Parameters**:
- `df` (DataFrame): Input dataframe
- `column_name` (str): Name of the column to analyze
- `threshold` (float, default=1.5): IQR multiplier for outlier bounds

**Returns**:
- Tuple: (row_indices, outlier_values, lower_bound, upper_bound)

**Functionality**:
- Calculates Q1 (25th percentile), Q3 (75th percentile), and IQR
- Defines outlier bounds: [Q1 - threshold×IQR, Q3 + threshold×IQR]
- Identifies rows with values outside these bounds
- Returns indices and values of outliers

**Usage**: Detecting anomalous data points that may affect model performance

---

### 7. `modelResults(model, accuracy, f1, precision, recall, roc_auc, roc_cur, cm)`
**Purpose**: Records and stores model evaluation metrics

**Parameters**:
- `model` (str): Model name/identifier
- `accuracy` (float): Accuracy score
- `f1` (float): F1 score
- `precision` (float): Precision score
- `recall` (float): Recall score
- `roc_auc` (float): ROC AUC score
- `roc_cur`: ROC curve data
- `cm`: Confusion matrix

**Functionality**:
- Prints all metrics to console
- Appends results to global `resultsTable` DataFrame
- Saves updated results to 'trainingResults.csv'

**Usage**: Tracking and comparing performance across different models

---

## Key Variables and Datasets

### Raw Data Variables
- **`df`**: Raw ECG signal data loaded from CSV (converted to mV)
- **`df_labels`**: Classification labels with both text and numeric encoding
- **`df_labeled`**: Merged dataset with labels and ECG data

### Train/Test Split Variables
- **`train_idx`**: Indices for training samples
- **`test_idx`**: Indices for test samples

### HRV Feature Variables
- **`hrv_train_with_labels`**: HRV features extracted from training ECG signals with labels
- **`hrv_test_clean`**: HRV features extracted from test ECG signals

### Feature Matrices
- **`feature_cols`**: List of feature column names (HRV features excluding target)
- **`x_train`**: Training features (raw)
- **`y_train`**: Training labels (binary: 0=Normal, 1=Atrial Fibrillation)
- **`x_test`**: Test features (raw)
- **`y_test`**: Test labels

### Preprocessed Matrices
- **`X_train_imputed`**: Training features after median imputation
- **`X_test_imputed`**: Test features after imputation
- **`X_train_scaled`**: Training features after RobustScaler normalization
- **`X_test_scaled`**: Test features after scaling
- **Final `x_train`, `x_test`**: Scaled and preprocessed features ready for ML

### Model Variables
- **`model_LR`**: Logistic Regression model variants
  - Parameters: `multi_class='auto'`, `max_iter=1000`, `class_weight='balanced'`
  - Variants: Baseline, Top 10 Features, Correlation-based, Augmented features
- **`model_RF`**: Random Forest Classifier variations
  - Parameters tested: `max_depth` in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], `random_state=3003`
  - Best performing: `max_depth=9` (F1: 0.8216)
- **`model_KNN`**: K-Nearest Neighbors Classifier
  - Parameters: `n_neighbors` varied from 1-15 for optimization
- **`model_RNC`**: Radius Neighbors Classifier
  - Parameters: `outlier_label='most_frequent'`, `radius` varied (0.5-10.0)
- **`model_NC`**: Nearest Centroid Classifier
  - Basic centroid-based classifier
- **`model_MLP`**: Multi-Layer Perceptron Classifier
  - Parameters: `hidden_layer_sizes=(100, 50)`, `max_iter=1000`, `random_state=3003`
- **`model_GB`**: Gradient Boosting Classifier
  - Parameters: `random_state=3003`, with tuned variant `learning_rate=0.05`, `max_depth=4`, `n_estimators=200`, `subsample=0.8`
- **`model_ADA`**: AdaBoost Classifier
  - Parameters: `n_estimators` varied from 50 to 10000, `random_state=3003`
  - Best performing: `n_estimators=1000` (F1: 0.8125)
- **`model_VOTE`**: Voting Classifier (Ensemble)
  - Combines Logistic Regression, Random Forest, KNN, and AdaBoost
  - Parameters: `voting='soft'`, `n_jobs=-1`
  - Best overall performance (F1: 0.8627, Accuracy: 0.9685)

### Evaluation Variables
- **`y_pred`**: Predicted labels
- **`accuracy`**: Accuracy score
- **`classification_rep`**: Detailed classification report
- **`cm`**: Confusion matrix
- **`resultsTable`**: DataFrame storing all model results

### Preprocessing Objects
- **`imputer`**: SimpleImputer with median strategy
- **`scaler`**: RobustScaler for feature normalization

---

## Machine Learning Workflow

### 1. Data Loading and Preprocessing
1. Load ECG signals from CSV (Cell 14)
2. Load and encode labels (Cell 16)
3. Stratified train-test split (80/20) with random_state=3003 (Cell 19)

### 2. Feature Engineering
1. ECG signal analysis (frequency domain, time domain)
2. HRV feature extraction using NeuroKit
3. Extract R-peaks and time-domain features
4. Generate comprehensive HRV feature set for both train and test

### 3. Data Quality Assessment
1. Correlation analysis
2. Distribution analysis
3. Outlier detection using IQR method (threshold=1.5)
4. Missing value analysis

### 4. Data Cleaning
1. Outlier handling using Winsorisation
2. Missing value imputation using median strategy
3. Feature scaling using RobustScaler

### 5. Model Training
1. Safety checks (no NaNs, aligned shapes)
2. Logistic Regression with balanced class weights (5 variants with feature engineering)
3. Random Forest with max_depth ranging from 1-14 (14 variants)
4. K-Nearest Neighbors with varying n_neighbors (1-15) (15 variants)
5. Radius Neighbors Classifier with varying radius (0.5-10.0) (8 variants)
6. Nearest Centroid Classifier
7. Multi-Layer Perceptron Neural Network with hidden layers (100, 50)
8. Gradient Boosting Classifier (2 variants)
9. AdaBoost Classifier with varying n_estimators (50-10000) (10 variants)
10. Voting Classifier (ensemble of best models)

### 6. Model Evaluation
1. Accuracy score
2. F1 score
3. Precision and Recall
4. ROC AUC score
5. Confusion matrix
6. Classification report

### 7. Results Storage
- All results saved to `trainingResults.csv` (57 unique models, 58 records)
- Results table maintained in memory for comparison
- HRV features saved separately to `hrv_train.csv` and `hrv_test.csv` for reuse

---

## Data Preprocessing Pipeline

### Imputation Strategy
- **Method**: Median imputation
- **Rationale**: Robust to outliers, suitable for skewed distributions
- **Implementation**: `SimpleImputer(strategy="median")`
- **Process**: Fit on training data, transform both train and test

### Scaling Strategy
- **Method**: RobustScaler
- **Rationale**: Uses median and IQR, less sensitive to outliers than StandardScaler
- **Implementation**: `RobustScaler()`
- **Process**: Fit on training data, transform both train and test
- **Result**: Features centered around median with IQR-based scaling

### Quality Checks
- Assert no NaN values in final datasets
- Assert alignment between X and y for train and test
- Verify feature column consistency
- Check distribution sanity between train and test

---

## Feature Engineering Details

### ECG Signal Processing
1. **Frequency Analysis**: Using Welch method for power spectral density
2. **Noise Analysis**:
   - Powerline interference (50/60 Hz)
   - Baseline wander (low frequency drift)
   - Muscle noise (high frequency artifacts)

### HRV Features Extracted
Using NeuroKit library for comprehensive HRV analysis:
- **R-peaks Detection**: Identification of QRS complex peaks
- **Time-domain Features**:
  - Heart rate statistics
  - RR interval statistics
  - SDNN (Standard Deviation of NN intervals)
  - RMSSD (Root Mean Square of Successive Differences)
  - pNN50 (Percentage of successive intervals differing by >50ms)
- **Additional HRV metrics**: As extracted by NeuroKit

---

## Model Evaluation Metrics

### Accuracy
- Proportion of correct predictions
- Used for overall model performance

### F1 Score
- Harmonic mean of precision and recall
- Particularly important for imbalanced datasets

### Precision
- True Positives / (True Positives + False Positives)
- Measures accuracy of positive predictions

### Recall (Sensitivity)
- True Positives / (True Positives + False Negatives)
- Measures ability to find all positive cases

### ROC AUC
- Area Under the Receiver Operating Characteristic curve
- Measures model's ability to distinguish between classes
- Range: 0.5 (random) to 1.0 (perfect)

### Confusion Matrix
- Shows true positives, true negatives, false positives, false negatives
- Normalized to show proportions

---

## File Structure

```
AtrialFibrillation-detection/
├── .git/                           # Git repository
├── .gitattributes                  # Git attributes
├── .gitignore                      # Git ignore rules
├── MAI3003_DataWitches_Assignment02.ipynb  # Main notebook (193 cells)
├── MAI3003_DataWitches_Assignment02.pdf    # PDF export of main notebook
├── DataWitches_Challenge.ipynb     # Challenge notebook (141 cells) - Extended ML experiments
├── Results_analysis.ipynb          # Utility notebook for results analysis (8 cells)
├── README.md                       # Basic project description
├── DOCUMENTATION.md                # This file - complete documentation
├── requirements.txt                # Python package dependencies
├── download_dataset.sh             # Kaggle dataset download script
├── trainingResults.csv             # Stored model evaluation results (57 models)
├── hrv_train.csv                   # Preprocessed HRV features for training (72 features)
├── hrv_test.csv                    # Preprocessed HRV features for testing (72 features)
├── pyvenv.cfg                      # Python virtual environment config
├── data/                           # Data directory
│   └── Physionet2017Training.tar.xz  # ECG dataset archive
└── share/                          # Shared resources
    ├── jupyter/                    # Jupyter kernel configurations
    └── man/                        # Manual pages
```

---

## Usage Instructions

### Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download dataset using `download_dataset.sh` (requires Kaggle credentials)
3. Extract data to `data/` directory

### Running the Main Notebook
1. Open `MAI3003_DataWitches_Assignment02.ipynb` in Jupyter or Google Colab
2. Run cells sequentially from top to bottom
3. All functions are defined before use
4. Results are automatically saved to `trainingResults.csv`
5. Preprocessed HRV features are saved to `hrv_train.csv` and `hrv_test.csv` for reuse

### Using Preprocessed HRV Features
The repository includes pre-computed HRV features to skip the computationally expensive feature extraction:
- **`hrv_train.csv`**: 72 HRV features extracted from training ECG signals (3,619 samples)
- **`hrv_test.csv`**: 72 HRV features extracted from test ECG signals (905 samples)
- These can be loaded directly to start from the modeling phase

### Challenge Notebook (DataWitches_Challenge.ipynb)
The challenge notebook (`DataWitches_Challenge.ipynb`) provides extended machine learning experiments and model comparisons:

**Purpose**:
- Comprehensive exploration of different ML algorithms for Atrial Fibrillation classification
- Automated search for optimal ensemble voting classifiers
- Visual comparison of model performance through ROC curves and metric bar charts

**Structure (141 cells)**:
- **58 markdown cells**: Documentation and section headers
- **83 code cells**: Feature extraction, model training, and evaluation

**Key Features**:
1. **Soft Voting Classifier**: Ensemble combining Logistic Regression, Random Forest (300 estimators), KNN (k=7), and AdaBoost (1000 estimators)
2. **Best Voting Classifier Search**: Automated testing of all 2-4 classifier combinations from a pool of 10 base models
3. **Comprehensive Classifier Sweep**: Tests 15+ different classifier types including Ridge, Decision Tree, Extra Trees, SVC, Naive Bayes variants, LDA, QDA, and more
4. **Visual Analysis**: ROC curves and metric bar charts (logarithmic and linear scales)

**Usage**:
1. Open `DataWitches_Challenge.ipynb` in Jupyter or Google Colab
2. Run cells sequentially; HRV features will be extracted (or can be loaded from CSV)
3. Experiment with different classifier combinations in the voting classifier search section
4. Compare results using the visualization cells at the end

### Results Analysis Notebook (Results_analysis.ipynb)
- **`Results_analysis.ipynb`**: Utility notebook for analyzing stored model results (8 code cells)
- Loads and analyzes `trainingResults.csv` for quick result comparisons
- Useful for post-hoc analysis without re-running models

### Modifying the Pipeline
- To add new models: Add implementation in Section 9 (Cell 117-178)
- To modify preprocessing: Edit cells in Section 7 (Cell 96-109)
- To add new features: Modify feature engineering in Section 4 (Cell 54-68)
- To change evaluation metrics: Update `modelResults()` function
- To experiment with different ensemble combinations: Modify VotingClassifier in Cell 161

---

## Notes and Conventions

### Random State
- Consistent `random_state=3003` used throughout for reproducibility
- Applied to train_test_split, RandomForestClassifier, etc.

### Class Encoding
- 0 = Normal Sinus Rhythm (N)
- 1 = Atrial Fibrillation (A)

### Missing Value Handling
- Strategy: Median imputation
- Applied after outlier handling, before scaling

### Outlier Handling
- Method: IQR-based detection with threshold=1.5
- Treatment: Winsorisation (capping at bounds)

### Scaling
- RobustScaler chosen over StandardScaler for outlier robustness
- Fit only on training data to prevent data leakage

---

## Model Comparison Results

The project tested **57 unique machine learning models** across multiple algorithms with extensive hyperparameter tuning.

### Models Implemented (by count)
1. **Random Forest (14 variants)**: Ensemble method with max_depth ranging from 1-14
2. **K-Nearest Neighbors (15 variants)**: Distance-based classifier with n_neighbors from 1-15
3. **AdaBoost (10 variants)**: Boosting ensemble with n_estimators from 50-10000
4. **Radius Neighbors Classifier (8 variants)**: Distance-based classifier with varying radius (0.5-10.0)
5. **Logistic Regression (5 variants)**: Binary classifier with feature engineering variations
6. **Gradient Boosting (2 variants)**: Gradient boosting with default and tuned hyperparameters
7. **Voting Classifier (1 variant)**: Soft voting ensemble combining LR, RF, KNN, and AdaBoost
8. **Multi-Layer Perceptron (1 variant)**: Neural network with two hidden layers (100, 50)
9. **Nearest Centroid (1 variant)**: Simple centroid-based classifier

**Total: 57 unique models** 

Note: trainingResults.csv contains 58 records due to one duplicate GradientBoostingClassifier entry (the model was accidentally run twice). The duplicate can be ignored when analyzing results as both entries have identical configurations and performance metrics.

### Top Performing Models (by F1 Score)
1. **Voting Classifier (Ensemble)**: F1=0.8627, Accuracy=0.9685, Precision=0.8713, Recall=0.8544, ROC-AUC=0.9847
   - Best overall performance combining multiple algorithms
2. **Random Forest (max_depth=9)**: F1=0.8216, Accuracy=0.9314, Precision=0.9268, Recall=0.7379, ROC-AUC=0.9817
   - Best single Random Forest configuration
3. **Random Forest (max_depth=8)**: F1=0.8152, Accuracy=0.9314, Precision=0.9259, Recall=0.7282, ROC-AUC=0.9803
4. **Random Forest (max_depth=7)**: F1=0.8128, Accuracy=0.9314, Precision=0.9048, Recall=0.7379, ROC-AUC=0.9828
5. **Gradient Boosting (tuned)**: F1=0.8128, Accuracy=0.9606, Precision=0.9048, Recall=0.7379, ROC-AUC=0.9826
   - Parameters: learning_rate=0.05, max_depth=4, n_estimators=200, subsample=0.8
6. **AdaBoost (n_estimators=1000)**: F1=0.8125, Accuracy=0.9595, Precision=0.8764, Recall=0.7573, ROC-AUC=0.9784

### Logistic Regression Feature Engineering Results
- **Baseline LR (all features)**: Accuracy=0.9314, F1=0.7589, Precision=0.6400, Recall=0.9320, ROC-AUC=0.9732
  - Good baseline but poor precision (0.64), high recall
- **LR with Top 10 Features**: Accuracy=0.9179, F1=0.7181, Precision=0.5962, Recall=0.9029, ROC-AUC=0.9642
  - Features selected by importance ranking from baseline model
  - Simpler model with slightly lower performance across all metrics
- **LR with Correlation Filter (>0.8)**: Accuracy=0.9269, F1=0.7490, Precision=0.6218, Recall=0.9417, ROC-AUC=0.9333
  - Removed highly correlated features to reduce multicollinearity
  - Marginal improvement in recall
- **LR Augmented (kept parent features)**: Accuracy=0.9370, F1=0.7724, Precision=0.6643, Recall=0.9223, ROC-AUC=0.9751
  - Added engineered features while keeping original features
  - Best LR variant with improved precision
- **LR Augmented (dropped parents)**: Accuracy=0.9325, F1=0.7619, Precision=0.6443, Recall=0.9320, ROC-AUC=0.9735
  - Replaced original features with engineered features only
  - Feature replacement didn't improve performance over baseline

### Key Observations
- **Ensemble methods dominate**: Voting Classifier achieved the best overall performance
- **Random Forest sweet spot**: max_depth of 7-9 provided optimal balance
- **Boosting is effective**: AdaBoost and Gradient Boosting both performed well
- **High precision models**: Random Forest variants achieved precision up to 0.93
- **Feature engineering matters**: Augmented features improved LR performance by ~5% in F1 score
- **All models show strong discrimination**: ROC-AUC scores consistently above 0.97
- **Trade-offs observed**: Higher precision often came at the cost of recall
- Results comprehensively stored in `trainingResults.csv` with 58 total records

## Future Enhancements

Based on the notebook structure and current implementation, potential areas for expansion:
1. ✓ **Hyperparameter tuning**: Completed through systematic parameter exploration across 57 model variants (manual testing of specific parameter values rather than automated GridSearchCV)
2. ✓ **Ensemble methods**: Implemented (Voting Classifier combining multiple models)
3. Cross-validation for robust performance estimation (currently using single train-test split)
4. Feature importance analysis for better interpretability
5. SHAP values for model interpretability
6. Deep learning approaches (CNNs, RNNs, LSTMs) for raw ECG signal processing
7. Real-time ECG classification pipeline
8. Clinical validation with medical professionals
9. Integration with ECG monitoring devices
10. Deployment as a web service or mobile application

---

## Contact

For questions or contributions, please contact team members listed at the top of this document.

---

*Documentation generated for Data Witches Project 2*  
*Last updated: 2025-11-27 (Updated with detailed DataWitches_Challenge.ipynb documentation, including notebook structure, automated ensemble search, comprehensive classifier sweep, and visual model comparison features)*
