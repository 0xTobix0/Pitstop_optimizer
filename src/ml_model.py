"""
Machine learning model for F1 pitstop prediction.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Tuple, List
from config import MODEL_CONFIG, TRACK_PARAMS, RACE_CONFIG
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class PitstopPredictor:
    def __init__(self, track: str):
        """Initialize predictor for a specific track."""
        self.track = track
        self.model = None
        self.feature_importance = {}
        self.params = TRACK_PARAMS[track]
        self.feature_names = [
            'LapNumber', 'TyreLife', 'Compound', 'LapTime',
            'TrackTemp', 'AirTemp', 'Humidity', 'Position'
        ]
        
    def train(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Train the model using historical data."""
        # Prepare features
        df['CompoundEncoded'] = df['Compound'].map({'SOFT': 1, 'MEDIUM': 2, 'HARD': 3})
        
        # One-hot encode compounds to better capture performance differences
        compound_dummies = pd.get_dummies(df['Compound'], prefix='Compound')
        df = pd.concat([df, compound_dummies], axis=1)
        
        # Calculate stint progression relative to optimal stint length
        df['StintProgression'] = df.apply(
            lambda x: x['TyreLife'] / self.params['max_stint'][x['Compound']]['optimal'],
            axis=1
        )
        
        # Calculate base tire degradation with compound-specific factors
        base_deg = df['TyreLife'] * RACE_CONFIG['base_tire_deg_rate']
        compound_factors = {'SOFT': 1.3, 'MEDIUM': 1.1, 'HARD': 0.9}
        df['TyreDegradation'] = df.apply(
            lambda x: base_deg.loc[x.name] * self.params['tire_deg_factor'] * compound_factors[x['Compound']],
            axis=1
        )
        
        # Add interaction features
        df['TyreLife_X_LapNumber'] = df['TyreLife'] * df['LapNumber']
        df['StintProg_X_CompoundEnc'] = df['StintProgression'] * df['CompoundEncoded']
        df['Position_X_LapTime'] = df['Position'] * df['LapTime']
        
        # Add stint-based features
        df['RemainingOptimal'] = df.apply(
            lambda x: max(0, self.params['max_stint'][x['Compound']]['optimal'] - x['TyreLife']),
            axis=1
        )
        
        # Prepare training data
        feature_names = [
            'LapNumber', 'TyreLife', 'CompoundEncoded', 'LapTime',
            'TrackTemp', 'AirTemp', 'Humidity', 'Position',
            'TyreDegradation', 'StintProgression',
            'TyreLife_X_LapNumber', 'StintProg_X_CompoundEnc',
            'Position_X_LapTime', 'Compound_SOFT', 'Compound_MEDIUM',
            'Compound_HARD', 'RemainingOptimal'
        ]
        
        X = df[feature_names].fillna(0)
        y = df['RemainingLaps']
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model with early stopping
        self.model = lgb.train(
            params=MODEL_CONFIG,
            train_set=train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10)]
        )
        
        # Save model
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_model(os.path.join(model_dir, f'pitstop_model_{self.track}.txt'))
        
        # Calculate metrics
        val_preds = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, val_preds)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        
        return mae, rmse
        
    def predict(self, conditions: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions for given conditions."""
        # Load model if not loaded
        if self.model is None:
            self._load_model()
            
        # Copy conditions to avoid modifying input
        conditions = conditions.copy()
        
        # Encode compound
        conditions['CompoundEncoded'] = conditions['Compound'].map({'SOFT': 1, 'MEDIUM': 2, 'HARD': 3})
        
        # One-hot encode compounds and ensure all columns exist
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            conditions[f'Compound_{compound}'] = (conditions['Compound'] == compound).astype(int)
        
        # Calculate stint progression
        conditions['StintProgression'] = conditions.apply(
            lambda x: x['TyreLife'] / self.params['max_stint'][x['Compound']]['optimal'],
            axis=1
        )
        
        # Calculate tire degradation
        base_deg = conditions['TyreLife'] * RACE_CONFIG['base_tire_deg_rate']
        compound_factors = {'SOFT': 1.3, 'MEDIUM': 1.1, 'HARD': 0.9}
        conditions['TyreDegradation'] = conditions.apply(
            lambda x: base_deg.loc[x.name] * self.params['tire_deg_factor'] * compound_factors[x['Compound']],
            axis=1
        )
        
        # Add interaction features
        conditions['TyreLife_X_LapNumber'] = conditions['TyreLife'] * conditions['LapNumber']
        conditions['StintProg_X_CompoundEnc'] = conditions['StintProgression'] * conditions['CompoundEncoded']
        conditions['Position_X_LapTime'] = conditions['Position'] * conditions['LapTime']
        
        # Calculate remaining optimal laps
        conditions['RemainingOptimal'] = conditions.apply(
            lambda x: max(0, self.params['max_stint'][x['Compound']]['optimal'] - x['TyreLife']),
            axis=1
        )
        
        # Prepare features for prediction
        feature_names = [
            'LapNumber', 'TyreLife', 'CompoundEncoded', 'LapTime',
            'TrackTemp', 'AirTemp', 'Humidity', 'Position',
            'TyreDegradation', 'StintProgression',
            'TyreLife_X_LapNumber', 'StintProg_X_CompoundEnc',
            'Position_X_LapTime', 'Compound_SOFT', 'Compound_MEDIUM',
            'Compound_HARD', 'RemainingOptimal'
        ]
        
        X = conditions[feature_names].fillna(0)
        
        # Get prediction from model
        remaining_laps = self.model.predict(X)[0]
        
        # Calculate confidence based on feature importance and prediction variance
        feature_importances = self.model.feature_importance()
        important_features = sum(1 for imp in feature_importances if imp > 0)
        base_confidence = min(0.85, 0.65 + (important_features / len(feature_names)) * 0.2)
        
        # Adjust confidence based on stint progression
        stint_progress = conditions['StintProgression'].iloc[0]
        if stint_progress > 0.8:
            base_confidence *= 0.9
        elif stint_progress < 0.2:
            base_confidence *= 0.95
            
        # Determine pit window size based on overtaking difficulty
        window_size = 4  # Default window size
        if self.params['overtaking_difficulty'] < 0.4:
            window_size = 3  # Smaller window for easy overtaking
        elif self.params['overtaking_difficulty'] > 0.7:
            window_size = 5  # Larger window for difficult overtaking
            
        # Adjust window for safety car probability
        if self.params['safety_car_probability'] > 0.35:
            window_size += 2
        
        # Calculate pit window
        optimal_pit_lap = max(1, int(conditions['LapNumber'].iloc[0] + remaining_laps))
        window_start = max(1, optimal_pit_lap - window_size // 2)
        window_end = min(self.params['total_laps'], optimal_pit_lap + window_size // 2)
        
        return {
            'optimal_pit_lap': optimal_pit_lap,
            'laps_until_pit': remaining_laps,
            'pit_window_start': window_start,
            'pit_window_end': window_end,
            'confidence': base_confidence,
            'uncertainty': window_size / 2
        }

    def _load_model(self):
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_path = os.path.join(model_dir, f'pitstop_model_{self.track}.txt')
        if not os.path.exists(model_path):
            raise ValueError("Model not trained. Please train the model first.")
        self.model = lgb.Booster(model_file=model_path)

    def calculate_prediction_confidence(self, df: pd.DataFrame, pit_lap: int) -> float:
        """Calculate confidence score for prediction."""
        # Base confidence starts at 0.8
        confidence = 0.8
        
        # Reduce confidence if tire life is beyond typical stint length
        compound = df['Compound'].iloc[-1]
        typical_stint = self.params['max_stint'][compound]
        current_life = df['TyreLife'].iloc[-1]
        total_stint = current_life + (pit_lap - df['LapNumber'].iloc[-1])
        
        if total_stint > typical_stint:
            confidence *= 0.7  # Significant reduction for exceeding typical stint
        elif total_stint > typical_stint * 0.9:
            confidence *= 0.85  # Moderate reduction when close to limit
            
        # Reduce confidence if lap times are very inconsistent
        lap_time_std = df['LapTime'].std()
        if lap_time_std > 2.0:  # More than 2 seconds variation
            confidence *= 0.9
            
        # Reduce confidence if track evolution is significant
        if self.params['track_evolution'] > 0.1:  # More than 10% improvement
            confidence *= 0.95
            
        # Reduce confidence for extreme temperatures
        track_temp = df['TrackTemp'].iloc[-1]
        if track_temp > 45 or track_temp < 15:
            confidence *= 0.9
            
        return round(confidence, 2)
