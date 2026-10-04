import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    precision_score, recall_score, f1_score, accuracy_score, 
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import BaseEstimator, ClassifierMixin,RegressorMixin
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor


class MLPRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Hybrid MLP feature extractor + Random Forest classifier"""
    def __init__(self, mlp_hidden_layers=(50, 25), rf_n_estimators=100, random_state=42):
        self.mlp_hidden_layers = mlp_hidden_layers
        self.rf_n_estimators = rf_n_estimators
        self.random_state = random_state
        self.mlp_feature_extractor = None
        self.rf_classifier = None
        self.scaler = None
        
    def fit(self, X, y):
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train MLP for feature extraction
        self.mlp_feature_extractor = MLPClassifier(
            hidden_layer_sizes=self.mlp_hidden_layers,
            random_state=self.random_state,
            max_iter=1000
        )
        self.mlp_feature_extractor.fit(X_scaled, y)
        
        # Extract features from the last hidden layer
        mlp_features = self._extract_features(X_scaled)
        
        # Train Random Forest on MLP features
        self.rf_classifier = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            random_state=self.random_state
        )
        self.rf_classifier.fit(mlp_features, y)
        
        return self
    
    def _extract_features(self, X):
        """Extract features from MLP hidden layers"""
        # Get activations from the last hidden layer
        activations = X.copy()
        for i, layer in enumerate(self.mlp_feature_extractor.coefs_[:-1]):
            activations = np.maximum(0, np.dot(activations, layer) + self.mlp_feature_extractor.intercepts_[i])
        return activations
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        mlp_features = self._extract_features(X_scaled)
        return self.rf_classifier.predict(mlp_features)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        mlp_features = self._extract_features(X_scaled)
        return self.rf_classifier.predict_proba(mlp_features)


class MLPRegressionRandomForest(BaseEstimator, RegressorMixin):
    """Hybrid MLP feature extractor + Random Forest Regressor"""
    def __init__(self, mlp_hidden_layers=(50, 25), rf_n_estimators=100, random_state=42):
        self.mlp_hidden_layers = mlp_hidden_layers
        self.rf_n_estimators = rf_n_estimators
        self.random_state = random_state
        self.mlp_feature_extractor = None
        self.rf_regressor = None
        self.scaler = None
        
    def fit(self, X, y):
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train MLP for feature extraction
        self.mlp_feature_extractor = MLPRegressor(
            hidden_layer_sizes=self.mlp_hidden_layers,
            random_state=self.random_state,
            max_iter=1000
        )
        self.mlp_feature_extractor.fit(X_scaled, y)
        
        # Extract features from the last hidden layer
        mlp_features = self._extract_features(X_scaled)
        
        # Train Random Forest Regressor on MLP features
        self.rf_regressor = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            random_state=self.random_state
        )
        self.rf_regressor.fit(X, y)
        
        return self
    
    def _extract_features(self, X):
        """Extract features from MLP hidden layers"""
        activations = X.copy()
        for i, layer in enumerate(self.mlp_feature_extractor.coefs_[:-1]):
            activations = np.maximum(0, np.dot(activations, layer) + self.mlp_feature_extractor.intercepts_[i])
        return activations
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        mlp_features = self._extract_features(X_scaled)
        return self.rf_regressor.predict(X)


class VANETMLPipeline:
    def __init__(self):
        self.classification_models = {}
        self.regression_models = {}
        self.scalers = {}
        
        # Available model configurations
        self.available_classification_models = {
            'SGD Classifier': SGDClassifier(loss='log_loss', max_iter=50, tol=1e-2, alpha=1.0, learning_rate='constant', eta0=1e-4),
            'GP Classifier': GaussianProcessClassifier(kernel=RBF(length_scale=1e12, length_scale_bounds="fixed"), optimizer=None, max_iter_predict=1, n_restarts_optimizer=0, multi_class='one_vs_one'),
            'KNN Classifier': KNeighborsClassifier(n_neighbors=30, weights='uniform', metric='cosine'),
            'FusionMind Classifier': MLPRandomForestClassifier()
        }
        
        self.available_regression_models = {
            'SGD Regressor': SGDRegressor(),
            'GP Regressor': GaussianProcessRegressor(kernel=RBF(length_scale=1e-6, length_scale_bounds="fixed"), optimizer=None, n_restarts_optimizer=0, alpha=10, random_state=42),
            'KNN Regressor': KNeighborsRegressor(n_neighbors=30, weights='uniform', metric='cosine'),
            'FusionMind Regressor':MLPRegressionRandomForest(random_state=42)
        }
        
    def train_classification_models(self, X, y, selected_models=None):
        """Train selected classification models and return performance metrics."""
        if selected_models is None:
            selected_models = list(self.available_classification_models.keys())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['classification'] = scaler
        
        results = {}
        labels = ['Not optimal', 'Optimal']
        
        for name in selected_models:
            if name not in self.available_classification_models:
                continue
                
            model_file = f'models/{name.replace(" ", "_")}_classifier.pkl'
            scaler_file = f'models/{name.replace(" ", "_")}_classifier_scaler.pkl'
            
            # Check if model exists
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                print(f"Loading existing {name} model...")
                try:
                    model = joblib.load(model_file)
                    model_scaler = joblib.load(scaler_file)
                    self.classification_models[name] = model
                    
                    # Make predictions with loaded model
                    if name in ['SGD Classifier']:
                        y_pred = model.predict(X_test_scaled)
                        y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                    else:
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                    
                    # Calculate metrics
                    metrics = self._calculate_classification_metrics(y_test, y_pred, y_proba, name, labels)
                    results[name] = metrics
                    
                except Exception as e:
                    print(f"Error loading {name}: {str(e)}. Training new model...")
                    # Fall through to training
                    pass
                else:
                    continue
            
            # Train new model
            print(f"Training {name} model...")
            try:
                model = self.available_classification_models[name]
                
                # Train model
                if name in ['SGD Classifier']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                    model_scaler = scaler
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                    model_scaler = None
                
                # Calculate metrics
                metrics = self._calculate_classification_metrics(y_test, y_pred, y_proba, name, labels)
                results[name] = metrics
                
                # Save model
                self.classification_models[name] = model
                joblib.dump(model, model_file)
                if model_scaler:
                    joblib.dump(model_scaler, scaler_file)
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        # Save scaler
        joblib.dump(scaler, 'models/classification_scaler.pkl')
        
        # Generate comparison plots
        self._plot_classification_comparison(results)
        
        return results
    def train_regression_models(self, X, y, selected_models=None):
        """Train selected regression models (including Hybrid MLP+RF) and return performance metrics."""
        if selected_models is None:
            selected_models = list(self.available_regression_models.keys())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features (for models that require it)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['regression'] = scaler
        
        results = {}
        
        for name in selected_models:
            if name not in self.available_regression_models:
                continue
                
            model_file = f'models/{name.replace(" ", "_")}_regressor.pkl'
            scaler_file = f'models/{name.replace(" ", "_")}_regressor_scaler.pkl'
            
            # Check if model exists
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                print(f"Loading existing {name} model...")
                try:
                    model = joblib.load(model_file)
                    model_scaler = joblib.load(scaler_file)
                    self.regression_models[name] = model
                    
                    # Predictions
                    if name in ['SGD Regressor']:
                        y_pred = model.predict(X_test_scaled)
                    elif name == "MLP+RF Regressor":
                        y_pred = model.predict(X_test)  # internally scales
                    else:
                        y_pred = model.predict(X_test)
                    
                    # Metrics
                    metrics = self._calculate_regression_metrics(y_test, y_pred, name)
                    results[name] = metrics
                    
                except Exception as e:
                    print(f"Error loading {name}: {str(e)}. Training new model...")
                    pass
                else:
                    continue
            
            # Train new model
            print(f"Training {name} model...")
            try:
                model = self.available_regression_models[name]
                
                if name in ['SGD Regressor']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    model_scaler = scaler
                
                elif name == "MLP+RF Regressor":
                    # Hybrid model handles scaling internally
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    model_scaler = None
                
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    model_scaler = None
                
                # Metrics
                metrics = self._calculate_regression_metrics(y_test, y_pred, name)
                results[name] = metrics
                
                # Save
                self.regression_models[name] = model
                joblib.dump(model, model_file)
                if model_scaler:
                    joblib.dump(model_scaler, scaler_file)
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        # Save global scaler for consistency
        joblib.dump(scaler, 'models/regression_scaler.pkl')
        
        # Plot comparison
        self._plot_regression_comparison(results)
        
        return results

    
    def _calculate_classification_metrics(self, y_true, y_pred, y_proba, model_name, labels):
        """Calculate classification metrics and generate plots."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred) * 100,
            'precision': precision_score(y_true, y_pred, average='macro') * 100,
            'recall': recall_score(y_true, y_pred, average='macro') * 100,
            'f1_score': f1_score(y_true, y_pred, average='macro') * 100,
            'classification_report': classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        }
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,   # Disable colorbar
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 14, "weight": "bold"}  # Increase internal font size
        )

        plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)

        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12, rotation=0)

        plt.tight_layout()
        plt.savefig(f'static/plots/{model_name.replace(" ", "_")}_confusion_matrix.png')
        plt.close()
        # ROC Curve
        if y_proba is not None:
            try:
                if len(labels) == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    plt.figure(figsize=(10, 8))
                    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'{model_name} - ROC Curve')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f'static/plots/{model_name.replace(" ", "_")}_roc_curve.png')
                    plt.close()
                    
                    metrics['auc'] = roc_auc
            except Exception as e:
                print(f"Error generating ROC curve for {model_name}: {str(e)}")
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true, y_pred, model_name):
        """Calculate regression metrics and generate plots."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2
        }
        
        # Scatter plot of predictions vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Predictions vs Actual\nR² = {r2:.3f}, RMSE = {rmse:.3f}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'static/plots/{model_name.replace(" ", "_")}_predictions.png')
        plt.close()
        
        return metrics
    
    def _plot_classification_comparison(self, results):
        """Generate classification models comparison plot."""
        if not results:
            return
            
        # Extract metrics for plotting
        models = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                models.append(model_name)
                accuracies.append(metrics['accuracy'])
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1_scores.append(metrics['f1_score'])
        
        if not models:
            return
        
        # Create comparison plot
        x = np.arange(len(models))
        width = 0.2
        
        plt.figure(figsize=(12, 8))
        plt.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        plt.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
        plt.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score (%)')
        plt.title('Classification Models Performance Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('static/plots/classification_comparison.png')
        plt.close()
    
    def _plot_regression_comparison(self, results):
        """Generate regression models comparison plot."""
        if not results:
            return
            
        # Extract metrics for plotting
        models = []
        mae_values = []
        rmse_values = []
        r2_values = []
        
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                models.append(model_name)
                mae_values.append(metrics['mae'])
                rmse_values.append(metrics['rmse'])
                r2_values.append(metrics['r2_score'])
        
        if not models:
            return
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MAE and RMSE
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
        ax1.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Error')
        ax1.set_title('Model Error Comparison (Lower is Better)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # R² Score
        ax2.bar(models, r2_values, alpha=0.8, color='green')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Model R² Score Comparison (Higher is Better)')
        ax2.set_xticklabels(models, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('static/plots/regression_comparison.png')
        plt.close()
    
    def predict_classification(self, X_test):
        """Make classification predictions on test data."""
        if not self.classification_models:
            return None
        
        # Scale test data
        if 'classification' in self.scalers:
            X_test_scaled = self.scalers['classification'].transform(X_test)
        else:
            X_test_scaled = X_test
        
        predictions = {}
        for name, model in self.classification_models.items():
            try:
                if name in ['SGD Classifier']:
                    pred = model.predict(X_test_scaled)
                else:
                    pred = model.predict(X_test)
                
                # Convert numerical predictions back to labels
                pred_labels = ['Not optimal' if p == 0 else 'Optimal' for p in pred]
                predictions[name] = pred_labels
            except Exception as e:
                print(f"Error predicting with {name}: {str(e)}")
                predictions[name] = [f"Error: {str(e)}"] * len(X_test)
        
        return predictions
    
    def predict_regression(self, X_test):
        """Make regression predictions on test data."""
        if not self.regression_models:
            return None
        
        # Scale test data
        if 'regression' in self.scalers:
            X_test_scaled = self.scalers['regression'].transform(X_test)
        else:
            X_test_scaled = X_test
        
        predictions = {}
        for name, model in self.regression_models.items():
            try:
                if name in ['SGD Regressor']:
                    pred = model.predict(X_test_scaled)
                else:
                    pred = model.predict(X_test)
                
                predictions[name] = pred.tolist()
            except Exception as e:
                print(f"Error predicting with {name}: {str(e)}")
                predictions[name] = [f"Error: {str(e)}"] * len(X_test)
        
        return predictions
    
    def get_available_models(self):
        """Get list of available models for selection."""
        return {
            'classification': list(self.available_classification_models.keys()),
            'regression': list(self.available_regression_models.keys())
        }
