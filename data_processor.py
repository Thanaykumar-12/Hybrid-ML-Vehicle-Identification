import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

class DataProcessor:
    def __init__(self):
        pass
    
    def preprocess_data(self, df, is_train=True, label_encoders=None):
        """Preprocess the data for machine learning."""
        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Handle missing values
        df.dropna(inplace=True)
        
        # Detect and handle datetime columns
        datetime_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                datetime_col = col
                break
        
        if is_train:
            label_encoders = {}
            
            # Encode categorical variables
            for col in df.select_dtypes(include='object').columns:
                if col not in ['optimal_route_chosen', 'avg_vehicle_spacing']:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    label_encoders[col] = le
            
            # Handle target variables
            if 'optimal_route_chosen' in df.columns:
                le_target = LabelEncoder()
                df['optimal_route_chosen'] = le_target.fit_transform(df['optimal_route_chosen'].astype(str))
                label_encoders['optimal_route_chosen'] = le_target
            
            # Prepare features and targets
            feature_cols = [col for col in df.columns if col not in ['optimal_route_chosen', 'avg_vehicle_spacing']]
            X = df[feature_cols]
            y = df['optimal_route_chosen'] if 'optimal_route_chosen' in df.columns else None
            y1 = df['avg_vehicle_spacing'] if 'avg_vehicle_spacing' in df.columns else None
            
            return X, y, y1, label_encoders
        
        else:
            if label_encoders is None:
                raise ValueError("label_encoders must be provided for test/inference.")
            
            # Encode categorical variables using training encoders
            for col in df.select_dtypes(include='object').columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    # Handle unseen categories
                    try:
                        df[col] = le.transform(df[col].astype(str))
                    except ValueError:
                        # Replace unseen categories with most frequent class
                        most_frequent = le.classes_[0]
                        df[col] = df[col].astype(str).apply(
                            lambda x: x if x in le.classes_ else most_frequent
                        )
                        df[col] = le.transform(df[col])
            
            # Select only feature columns (exclude target variables if present)
            feature_cols = [col for col in df.columns if col not in ['optimal_route_chosen', 'avg_vehicle_spacing']]
            return df[feature_cols]
    
    def perform_eda(self, X, y, y1):
        """Perform Exploratory Data Analysis with 5 plots."""
        # Ensure plots directory exists
        os.makedirs('static/plots', exist_ok=True)
        
        # Set style
        plt.style.use('default')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('VANET Routing Dataset - Exploratory Data Analysis', fontsize=16, y=0.98)
        
        # 1. Count Plot of Classification Target
        ax1 = axes[0, 0]
        y_labels = ['Not optimal' if val == 0 else 'Optimal' for val in y]
        y_series = pd.Series(y_labels)
        sns.countplot(x=y_series, ax=ax1)
        ax1.set_title('Distribution of Optimal Route Chosen')
        ax1.set_xlabel('Route Optimality')
        ax1.set_ylabel('Count')
        
        # 2. Histogram of Regression Target
        ax2 = axes[0, 1]
        ax2.hist(y1, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Distribution of Average Vehicle Spacing')
        ax2.set_xlabel('Average Vehicle Spacing')
        ax2.set_ylabel('Frequency')
        
        # 3. Violin Plot: Distance to Destination vs Route Optimality
        ax3 = axes[0, 2]
        plot_data = pd.DataFrame({
            'distance_to_destination': X['distance_to_destination'],
            'optimal_route': y_labels
        })
        sns.violinplot(data=plot_data, x='optimal_route', y='distance_to_destination', ax=ax3)
        ax3.set_title('Distance to Destination vs Route Optimality')
        ax3.set_xlabel('Route Optimality')
        ax3.set_ylabel('Distance to Destination')
        
        # 4. Box Plot: Lane Count vs Average Vehicle Spacing
        ax4 = axes[1, 0]
        plot_data2 = pd.DataFrame({
            'lane_count': X['lane_count'],
            'avg_vehicle_spacing': y1
        })
        sns.boxplot(data=plot_data2, x='lane_count', y='avg_vehicle_spacing', ax=ax4)
        ax4.set_title('Lane Count vs Average Vehicle Spacing')
        ax4.set_xlabel('Lane Count')
        ax4.set_ylabel('Average Vehicle Spacing')
        
        # 5. Scatter Plot: Vehicle Speed vs Traffic Density
        ax5 = axes[1, 1]
        colors = ['red' if val == 0 else 'blue' for val in y]
        ax5.scatter(X['vehicle_speed'], X['traffic_density'], c=colors, alpha=0.6)
        ax5.set_title('Vehicle Speed vs Traffic Density')
        ax5.set_xlabel('Vehicle Speed')
        ax5.set_ylabel('Traffic Density')
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Not Optimal')
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Optimal')
        ax5.legend(handles=[red_patch, blue_patch])
        
        # 6. Correlation Heatmap
        ax6 = axes[1, 2]
        # Select numerical columns for correlation
        corr_data = X.copy()
        corr_data['avg_vehicle_spacing'] = y1
        correlation_matrix = corr_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                   cbar_kws={'shrink': 0.8}, ax=ax6)
        ax6.set_title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('static/plots/eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate additional individual plots for better visibility
        self._generate_individual_plots(X, y, y1)
        
        return True
    
    def _generate_individual_plots(self, X, y, y1):
        """Generate individual plots for better visibility."""
        
        # Feature distributions
        numerical_features = ['vehicle_speed', 'traffic_density', 'route_stability_score', 
                            'packet_loss_rate', 'signal_strength', 'distance_to_destination', 
                            'rsu_coverage_score']
        
        # Create feature distribution plots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Feature Distributions', fontsize=14)
        axes = axes.ravel()
        
        for i, feature in enumerate(numerical_features):
            if feature in X.columns and i < len(axes):
                axes[i].hist(X[feature], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
                axes[i].set_title(f'{feature.replace("_", " ").title()}')
                axes[i].set_xlabel(feature.replace("_", " ").title())
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numerical_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('static/plots/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance plot (using correlation with targets)
        y_labels = ['Not optimal' if val == 0 else 'Optimal' for val in y]
        y_numeric = pd.Series(y)
        
        # Calculate correlations
        correlations_classification = {}
        correlations_regression = {}
        
        for feature in X.columns:
            if X[feature].dtype in ['int64', 'float64']:
                correlations_classification[feature] = abs(np.corrcoef(X[feature], y_numeric)[0, 1])
                correlations_regression[feature] = abs(np.corrcoef(X[feature], y1)[0, 1])
        
        # Plot feature importance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Classification correlations
        features = list(correlations_classification.keys())
        values = list(correlations_classification.values())
        ax1.barh(features, values, alpha=0.7, color='lightcoral')
        ax1.set_title('Feature Correlation with Route Optimality')
        ax1.set_xlabel('Absolute Correlation')
        
        # Regression correlations
        features = list(correlations_regression.keys())
        values = list(correlations_regression.values())
        ax2.barh(features, values, alpha=0.7, color='lightgreen')
        ax2.set_title('Feature Correlation with Vehicle Spacing')
        ax2.set_xlabel('Absolute Correlation')
        
        plt.tight_layout()
        plt.savefig('static/plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
