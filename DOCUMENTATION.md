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
- `sklearn.preprocessing` - MinMaxScaler, StandardScaler, RobustScaler
- `sklearn.impute.SimpleImputer` - Missing value imputation
- `sklearn.metrics` - classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, RocCurveDisplay, precision_recall_curve

---

## Notebook Structure

The notebook consists of **127 cells** organized into the following major sections:

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

### 9. Machine Learning Training (Cell 117-124)
- **Cell 117**: ML training overview
- **Cell 118**: Logistic Regression section
- **Cell 119**: Logistic Regression implementation
- **Cell 121**: Random Forest section
- **Cell 122**: Random Forest implementation
- **Cell 124**: Additional models placeholder

### 10. Results (Cell 125-126)
- **Cell 125**: Results comparison
- **Cell 126**: Results visualization

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
- **`model_LR`**: Logistic Regression model
  - Parameters: `multi_class='auto'`, `max_iter=1000`, `class_weight='balanced'`
- **`model_RF`**: Random Forest Classifier
  - Parameters: `max_depth=5`, `random_state=3003`

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
2. Logistic Regression with balanced class weights
3. Random Forest with max_depth=5
4. Additional models (placeholder for expansion)

### 6. Model Evaluation
1. Accuracy score
2. F1 score
3. Precision and Recall
4. ROC AUC score
5. Confusion matrix
6. Classification report

### 7. Results Storage
- All results saved to `trainingResults.csv`
- Results table maintained in memory for comparison

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
Data-Witches_Project2/
├── .git/                           # Git repository
├── .gitattributes                  # Git attributes
├── .gitignore                      # Git ignore rules
├── MAI3003_DataWitches_Assignment02.ipynb  # Main notebook (127 cells)
├── README.md                       # Basic project description
├── DOCUMENTATION.md                # This file - complete documentation
├── requirements.txt                # Python package dependencies
├── download_dataset.sh             # Kaggle dataset download script
├── trainingResults.csv             # Stored model evaluation results
├── pyvenv.cfg                      # Python virtual environment config
├── data/                           # Data directory
│   └── Physionet2017Training.tar.xz  # ECG dataset archive
└── share/                          # Shared resources
```

---

## Usage Instructions

### Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download dataset using `download_dataset.sh` (requires Kaggle credentials)
3. Extract data to `data/` directory

### Running the Notebook
1. Open `MAI3003_DataWitches_Assignment02.ipynb` in Jupyter or Google Colab
2. Run cells sequentially from top to bottom
3. All functions are defined before use
4. Results are automatically saved to `trainingResults.csv`

### Modifying the Pipeline
- To add new models: Add implementation in Section 9 (Cell 117+)
- To modify preprocessing: Edit cells in Section 7 (Cell 96-109)
- To add new features: Modify feature engineering in Section 4 (Cell 54-68)
- To change evaluation metrics: Update `modelResults()` function

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

## Future Enhancements

Based on the notebook structure, potential areas for expansion:
1. Additional machine learning models (Cell 124 placeholder)
2. Hyperparameter tuning using GridSearchCV
3. Cross-validation for robust performance estimation
4. Feature importance analysis
5. SHAP values for model interpretability
6. Ensemble methods combining multiple models

---

## Contact

For questions or contributions, please contact team members listed at the top of this document.

---

*Documentation generated for Data Witches Project 2*  
*Last updated: 2025-11-21*
