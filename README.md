# ML Basics to Advanced Algorithms

A comprehensive machine learning project demonstrating the progression from basic linear regression implementations to advanced algorithmic techniques.

## Project Overview

This repository contains implementations of machine learning algorithms, starting with fundamental linear regression concepts and advancing to sophisticated techniques including regularization, feature selection, and cross-validation.

## Current Implementation

### Linear Regression (`linerregression.ipynb`)

The main notebook covers a complete journey through linear regression, from basic mathematical implementation to advanced optimization techniques:

#### 1. **Basic Linear Regression with NumPy**
- Manual implementation of the linear regression formula: `Y = MX + C`
- Matrix operations using NumPy for parameter calculation
- Understanding the mathematical foundation: `θ = (X^T·X)^(-1)·X^T·Y`

#### 2. **Scikit-Learn Implementation**
- Using `LinearRegression` from scikit-learn
- Model training with `.fit()` method
- Predictions and parameter extraction
- Performance comparison with manual implementation

#### 3. **Data Visualization**
- Scatter plots of training data
- Regression line visualization
- Extrapolation demonstrations
- Using Matplotlib for comprehensive plotting

#### 4. **Real-World Dataset Analysis**
- **Diabetes Dataset**: Multi-feature regression analysis
- 10-feature dataset with 442 samples
- Comprehensive data exploration using Pandas

#### 5. **Advanced Techniques**

##### **Regularization Methods**
- **Lasso Regression**: L1 regularization for feature selection
- **ElasticNet**: Combined L1 and L2 regularization
- Cross-validation for optimal hyperparameter selection

##### **Feature Engineering**
- **Polynomial Features**: Creating higher-degree feature combinations
- **Feature Selection**: SelectKBest, RFECV (Recursive Feature Elimination)
- **Automated Feature Selection**: SelectFromModel with various estimators

##### **Model Optimization**
- **Cross-Validation**: 5-fold CV for robust model evaluation
- **Hyperparameter Tuning**: Automated alpha selection with CV
- **Performance Metrics**: R², MAE, MSE, RMSE evaluation

## Key Features

- 📚 **Educational Progression**: From basic concepts to advanced implementations
- 🔬 **Multiple Approaches**: NumPy manual implementation vs. scikit-learn
- 📊 **Comprehensive Visualization**: Clear plots for understanding model behavior
- 🎯 **Feature Engineering**: Advanced feature selection and transformation
- 🔄 **Cross-Validation**: Robust model evaluation techniques
- 📈 **Performance Metrics**: Complete evaluation suite

## Technologies Used

- **Python 3.11**
- **NumPy**: Mathematical computations and array operations
- **Scikit-Learn**: Machine learning algorithms and utilities
- **Matplotlib**: Data visualization and plotting
- **Pandas**: Data manipulation and analysis

## Getting Started

### Prerequisites
```bash
pip install numpy scikit-learn matplotlib pandas
```

### Running the Notebook
1. Clone the repository
2. Open `linerregression.ipynb` in Jupyter Notebook or any compatible environment
3. Run cells sequentially to follow the learning progression

## Learning Outcomes

After working through this notebook, you will understand:

- Mathematical foundations of linear regression
- Implementation differences between manual and library approaches
- Feature engineering and selection techniques
- Regularization methods for preventing overfitting
- Cross-validation for model evaluation
- Advanced optimization techniques in machine learning

## Future Enhancements

This project is designed to expand with additional algorithms:
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines
- Neural Networks
- Deep Learning implementations

## Contributing

Contributions are welcome! Please feel free to submit pull requests for:
- Additional algorithm implementations
- Improved documentation
- Code optimizations
- New datasets and examples

## License

This project is open source and available under the MIT License. 