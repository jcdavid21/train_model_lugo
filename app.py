from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json
import time
import io
from flask_cors import CORS
import os


app = Flask(__name__)
app.config['SECRET_KEY'] = 'ml_training_secret'
CORS(app, origins="*")

socketio = SocketIO(app, cors_allowed_origins="*")

class MLModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def preprocess_data(self, df):
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['Region', 'Municipality', 'Barangay']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col])
            else:
                processed_df[col] = self.label_encoders[col].transform(processed_df[col])
        
        # Ensure all required columns exist and handle any remaining NaNs
        required_columns = ['Region', 'Municipality', 'Barangay', 'Microenterprise Development', 'Employment Facilitation', 'Total']
        for col in required_columns:
            if col not in processed_df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")
            # Fill any remaining NaN values with 0 for numeric columns or appropriate values for categorical
            if col in ['Microenterprise Development', 'Employment Facilitation', 'Total']:
                processed_df[col] = processed_df[col].fillna(0)
                
        # Prepare features and target
        X = processed_df[['Region', 'Municipality', 'Barangay', 'Microenterprise Development', 'Employment Facilitation']]
        y = processed_df['Total']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, df, epochs=100, socketio_instance=None):
        print("\n========== STARTING MODEL TRAINING ==========")
        print(f"Training with {len(df)} samples and {epochs} epochs")
        
        X, y = self.preprocess_data(df)
        print(f"Features: {self.feature_names}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        self.model = RandomForestRegressor(n_estimators=epochs, random_state=42)
        
        # Train with epoch simulation
        training_history = {'epoch': [], 'accuracy': [], 'loss': []}
        
        print("\nTraining Progress:")
        print("-" * 60)
        print(f"{'Epoch':<8}{'Progress':<15}{'Accuracy (R²)':<20}{'Loss (MSE)':<15}")
        print("-" * 60)
        
        for epoch in range(1, epochs + 1):
            # Train model with subset of estimators
            current_model = RandomForestRegressor(
                n_estimators=max(1, int(epoch * 100 / epochs)), 
                random_state=42
            )
            current_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = current_model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store history
            training_history['epoch'].append(epoch)
            training_history['accuracy'].append(max(0, r2))  # R² can be negative, but we'll show 0 as minimum
            training_history['loss'].append(mse)
            
            # Terminal progress output
            progress_percent = (epoch / epochs) * 100
            progress_bar = "█" * int(progress_percent / 5) + "░" * (20 - int(progress_percent / 5))
            
            # Print only every 5th epoch or first/last epoch to avoid console spam
            if epoch == 1 or epoch == epochs or epoch % 5 == 0:
                print(f"{epoch:<8}{progress_bar:<15}{max(0, r2):<20.4f}{mse:<15.4f}")
            
            # Emit progress to frontend
            if socketio_instance:
                socketio_instance.emit('training_progress', {
                    'epoch': epoch,
                    'total_epochs': epochs,
                    'accuracy': max(0, r2),
                    'loss': mse,
                    'progress': progress_percent
                })
            
            # Add small delay to simulate training time
            time.sleep(0.1)
        
        # Final training with all estimators
        print("\nFinalizing model with full estimators...")
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Final evaluation
        final_pred = self.model.predict(X_test_scaled)
        final_accuracy = r2_score(y_test, final_pred)
        final_loss = mean_squared_error(y_test, final_pred)
        
        print("\n========== TRAINING COMPLETED ==========")
        print(f"Final Accuracy (R²): {max(0, final_accuracy):.4f}")
        print(f"Final Loss (MSE): {final_loss:.4f}")
        print("=" * 40)
        
        return {
            'training_history': training_history,
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'test_size': len(X_test)
        }
    
    def predict(self, input_data):
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        # Preprocess input
        processed_input = input_data.copy()
        for col, encoder in self.label_encoders.items():
            if col in processed_input:
                processed_input[col] = encoder.transform([processed_input[col]])[0]
        
        # Create feature array
        features = np.array([[
            processed_input.get('Region', 0),
            processed_input.get('Municipality', 0),
            processed_input.get('Barangay', 0),
            processed_input.get('Microenterprise Development', 0),
            processed_input.get('Employment Facilitation', 0)
        ]])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        
        return prediction

# Global model instance
ml_model = MLModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and file.filename.endswith('.csv'):
            # Read CSV file
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            
            # Use pandas with thousands separator to handle comma-formatted numbers
            df = pd.read_csv(stream, thousands=',')
            
            # Ensure all numeric columns are properly converted to float
            numeric_columns = ['Microenterprise Development', 'Employment Facilitation', 'Total']
            for col in numeric_columns:
                if col in df.columns:
                    # Convert any string numbers with commas to float
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with NaN values in critical columns to prevent training errors
            critical_columns = ['Region', 'Municipality', 'Barangay', 'Microenterprise Development', 'Employment Facilitation', 'Total']
            df = df.dropna(subset=critical_columns)
            
            # Log data shape after cleaning
            print(f"\n===== DATASET UPLOADED =====")
            print(f"Original data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First few rows:")
            print(df.head(3))
            print("=" * 30)
            
            # Store dataset info
            dataset_info = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'sample_data': df.head().to_dict('records')
            }
            
            # Store dataframe for training
            app.config['dataset'] = df
            
            return jsonify({'success': True, 'dataset_info': dataset_info})
        else:
            return jsonify({'success': False, 'error': 'Please upload a CSV file'})
    
    except Exception as e:
        print(f"\n===== ERROR DURING UPLOAD =====")
        print(f"Error: {str(e)}")
        print("=" * 30)
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('start_training')
def handle_training(data):
    try:
        if 'dataset' not in app.config:
            print("\n===== TRAINING ERROR =====")
            print("No dataset uploaded")
            print("=" * 30)
            emit('training_error', {'error': 'No dataset uploaded'})
            return
        
        epochs = data.get('epochs', 100)
        df = app.config['dataset']
        
        print(f"\n===== TRAINING REQUESTED =====")
        print(f"Epochs: {epochs}")
        print(f"Dataset size: {len(df)} rows")
        print("=" * 30)
        
        # Additional check for empty dataset after cleaning
        if len(df) == 0:
            print("\n===== TRAINING ERROR =====")
            print("Dataset is empty after cleaning NaN values")
            print("=" * 30)
            emit('training_error', {'error': 'Dataset is empty after cleaning NaN values'})
            return
        
        # Double check for NaN values
        if df['Total'].isnull().any():
            # If there are still NaN values in target, drop those rows
            nan_count = df['Total'].isnull().sum()
            print(f"Found {nan_count} NaN values in target column. Removing...")
            df = df.dropna(subset=['Total'])
            app.config['dataset'] = df  # Update the stored dataset
            
            if len(df) == 0:
                print("\n===== TRAINING ERROR =====")
                print("No valid data remains after removing NaN target values")
                print("=" * 30)
                emit('training_error', {'error': 'No valid data remains after removing NaN target values'})
                return
                
        emit('training_started', {'epochs': epochs})
        
        # Train model
        results = ml_model.train(df, epochs=epochs, socketio_instance=socketio)
        
        emit('training_completed', results)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n===== TRAINING ERROR =====")
        print(f"Error: {str(e)}")
        print(f"Details:\n{error_details}")
        print("=" * 30)
        emit('training_error', {'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not ml_model.is_trained:
            return jsonify({'success': False, 'error': 'Model is not trained yet'})
        
        input_data = request.json
        print(f"\n===== PREDICTION REQUEST =====")
        print(f"Input data: {input_data}")
        
        prediction = ml_model.predict(input_data)
        
        print(f"Prediction result: {prediction}")
        print("=" * 30)
        
        return jsonify({'success': True, 'prediction': prediction})
    
    except Exception as e:
        print(f"\n===== PREDICTION ERROR =====")
        print(f"Error: {str(e)}")
        print("=" * 30)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_unique_values')
def get_unique_values():
    try:
        if 'dataset' not in app.config:
            return jsonify({'success': False, 'error': 'No dataset uploaded'})
        
        df = app.config['dataset']
        
        # Convert NaN values to strings to avoid JSON serialization errors
        unique_regions = df['Region'].dropna().unique().tolist()
        unique_municipalities = df['Municipality'].dropna().unique().tolist()
        unique_barangays = df['Barangay'].dropna().unique().tolist()
        
        unique_values = {
            'Region': unique_regions,
            'Municipality': unique_municipalities,
            'Barangay': unique_barangays
        }
        
        return jsonify({'success': True, 'unique_values': unique_values})
    
    except Exception as e:
        print(f"\n===== GET UNIQUE VALUES ERROR =====")
        print(f"Error: {str(e)}")
        print("=" * 30)
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n========== ML TRAINING SERVER STARTED ==========")
    print(f"Server running at: http://0.0.0.0:8800")
    print("Upload a CSV file and train your model")
    print("=" * 50)
    socketio.run(app, debug=True, host='0.0.0.0', port=8800)