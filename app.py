import os
import logging
from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
from ml_models import VANETMLPipeline
from data_processor import DataProcessor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Initialize ML pipeline and data processor
ml_pipeline = VANETMLPipeline()
data_processor = DataProcessor()

# Global variables to store processed data
processed_data = {}

@app.route('/')
def home():
    """Home page with dataset overview."""
    try:
        # Load and process training data
        df = pd.read_csv('vanet_routing_dataset.csv')
        
        # Basic dataset statistics
        dataset_info = {
            'total_records': len(df),
            'features': len(df.columns) - 2,  # Excluding target variables
            'classification_target': 'optimal_route_chosen',
            'regression_target': 'avg_vehicle_spacing',
            'missing_values': df.isnull().sum().sum(),
            'feature_names': [col for col in df.columns if col not in ['optimal_route_chosen', 'avg_vehicle_spacing']]
        }
        
        # Class distribution for classification target
        class_distribution = df['optimal_route_chosen'].value_counts().to_dict()
        
        # Basic statistics for regression target
        regression_stats = {
            'mean': round(df['avg_vehicle_spacing'].mean(), 3),
            'std': round(df['avg_vehicle_spacing'].std(), 3),
            'min': round(df['avg_vehicle_spacing'].min(), 3),
            'max': round(df['avg_vehicle_spacing'].max(), 3)
        }
        
        return render_template('home.html', 
                             dataset_info=dataset_info,
                             class_distribution=class_distribution,
                             regression_stats=regression_stats)
    except Exception as e:
        app.logger.error(f"Error in home route: {str(e)}")
        flash(f"Error loading dataset: {str(e)}", 'error')
        return render_template('home.html', dataset_info=None)

@app.route('/eda')
def eda():
    """Exploratory Data Analysis page."""
    try:
        # Load and preprocess data
        df = pd.read_csv('vanet_routing_dataset.csv')
        X, y, y1, label_encoders = data_processor.preprocess_data(df.copy(), is_train=True)
        
        # Store processed data globally
        global processed_data
        processed_data = {
            'X': X, 'y': y, 'y1': y1, 
            'label_encoders': label_encoders,
            'original_df': df
        }
        
        # Generate EDA plots
        eda_plots = data_processor.perform_eda(X, y, y1)
        
        return render_template('eda.html', plots_generated=True)
    except Exception as e:
        app.logger.error(f"Error in EDA route: {str(e)}")
        flash(f"Error generating EDA plots: {str(e)}", 'error')
        return render_template('eda.html', plots_generated=False)

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    """Classification performance page."""
    try:
        # Check if data is processed
        if not processed_data:
            flash("Please visit EDA page first to process data.", 'warning')
            return redirect(url_for('eda'))
        
        # Get available models
        available_models = ml_pipeline.get_available_models()
        
        # Handle model selection
        selected_models = None
        if request.method == 'POST':
            selected_models = request.form.getlist('selected_models')
            if not selected_models:
                flash("Please select at least one model.", 'warning')
                return render_template('classification.html', 
                                     available_models=available_models['classification'],
                                     results=None,
                                     test_predictions=None)
        
        # Train classification models
        classification_results = ml_pipeline.train_classification_models(
            processed_data['X'], processed_data['y'], selected_models
        )
        
        # Make predictions on test data
        test_predictions = None
        try:
            test_df = pd.read_csv('testdata.csv')
            test_processed = data_processor.preprocess_data(
                test_df.copy(), is_train=False, 
                label_encoders=processed_data['label_encoders']
            )
            test_predictions = ml_pipeline.predict_classification(test_processed)
        except Exception as e:
            app.logger.error(f"Error processing test data: {str(e)}")
            flash(f"Could not process test data: {str(e)}", 'warning')
        
        return render_template('classification.html', 
                             available_models=available_models['classification'],
                             selected_models=selected_models,
                             results=classification_results,
                             test_predictions=test_predictions)
    except Exception as e:
        app.logger.error(f"Error in classification route: {str(e)}")
        flash(f"Error training classification models: {str(e)}", 'error')
        return render_template('classification.html', 
                             available_models=ml_pipeline.get_available_models()['classification'],
                             results=None,
                             test_predictions=None)

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    """Regression performance page."""
    try:
        # Check if data is processed
        if not processed_data:
            flash("Please visit EDA page first to process data.", 'warning')
            return redirect(url_for('eda'))
        
        # Get available models
        available_models = ml_pipeline.get_available_models()
        
        # Handle model selection
        selected_models = None
        if request.method == 'POST':
            selected_models = request.form.getlist('selected_models')
            if not selected_models:
                flash("Please select at least one model.", 'warning')
                return render_template('regression.html', 
                                     available_models=available_models['regression'],
                                     results=None,
                                     test_predictions=None)
        
        # Train regression models
        regression_results = ml_pipeline.train_regression_models(
            processed_data['X'], processed_data['y1'], selected_models
        )
        
        # Make predictions on test data
        test_predictions = None
        try:
            test_df = pd.read_csv('testdata.csv')
            test_processed = data_processor.preprocess_data(
                test_df.copy(), is_train=False, 
                label_encoders=processed_data['label_encoders']
            )
            test_predictions = ml_pipeline.predict_regression(test_processed)
        except Exception as e:
            app.logger.error(f"Error processing test data: {str(e)}")
            flash(f"Could not process test data: {str(e)}", 'warning')
        
        return render_template('regression.html', 
                             available_models=available_models['regression'],
                             selected_models=selected_models,
                             results=regression_results,
                             test_predictions=test_predictions)
    except Exception as e:
        app.logger.error(f"Error in regression route: {str(e)}")
        flash(f"Error training regression models: {str(e)}", 'error')
        return render_template('regression.html', 
                             available_models=ml_pipeline.get_available_models()['regression'],
                             results=None,
                             test_predictions=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page for single sample input."""
    try:
        # Check if data is processed and models are trained
        if not processed_data:
            flash("Please visit EDA page first to process data.", 'warning')
            return redirect(url_for('eda'))
        
        if not ml_pipeline.classification_models and not ml_pipeline.regression_models:
            flash("Please train some models first by visiting Classification or Regression pages.", 'warning')
            return redirect(url_for('home'))
        
        prediction_results = None
        
        if request.method == 'POST':
            try:
                # Get form data
                input_data = {
                    'vehicle_speed': float(request.form['vehicle_speed']),
                    'traffic_density': float(request.form['traffic_density']),
                    'route_stability_score': float(request.form['route_stability_score']),
                    'packet_loss_rate': float(request.form['packet_loss_rate']),
                    'signal_strength': float(request.form['signal_strength']),
                    'distance_to_destination': float(request.form['distance_to_destination']),
                    'rsu_coverage_score': float(request.form['rsu_coverage_score']),
                    'lane_count': int(request.form['lane_count']),
                    'road_priority': int(request.form['road_priority'])
                }
                
                # Create DataFrame with single row
                input_df = pd.DataFrame([input_data])
                
                # Process the input data using the same preprocessing as training
                processed_input = data_processor.preprocess_data(
                    input_df.copy(), is_train=False, 
                    label_encoders=processed_data['label_encoders']
                )
                
                # Make predictions
                classification_predictions = ml_pipeline.predict_classification(processed_input)
                regression_predictions = ml_pipeline.predict_regression(processed_input)
                
                prediction_results = {
                    'input_data': input_data,
                    'classification': classification_predictions,
                    'regression': regression_predictions
                }
                
                flash("Predictions generated successfully!", 'success')
                
            except ValueError as e:
                flash(f"Invalid input values: {str(e)}", 'error')
            except Exception as e:
                app.logger.error(f"Error making predictions: {str(e)}")
                flash(f"Error making predictions: {str(e)}", 'error')
        
        return render_template('predict.html', results=prediction_results)
    
    except Exception as e:
        app.logger.error(f"Error in predict route: {str(e)}")
        flash(f"Error in prediction page: {str(e)}", 'error')
        return render_template('predict.html', results=None)

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('static/plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
