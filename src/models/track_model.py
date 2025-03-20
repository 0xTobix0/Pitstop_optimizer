"""
Track characteristics model based on 2022-2023 F1 data with ML-based pit window optimization.
"""
from dataclasses import dataclass # Dataclass for track data
from typing import Dict, List, Tuple 
import numpy as np # Numerical operations
from sklearn.preprocessing import StandardScaler # Feature scaling for ML model
from lightgbm import LGBMRegressor # ML model for pit window prediction

# Tire compound characteristics
TIRE_COMPOUNDS = {
    'SOFT': {
        'max_life': 20, # Maximum life in laps
        'grip_level': 1.0, # Maximum grip
        'deg_rate': 1.3, # Higher degradation rate for softs
        'optimal_temp': 90  # degrees C
    },
    'MEDIUM': {
        'max_life': 30, # Maximum life in laps
        'grip_level': 0.85, # Maximum grip
        'deg_rate': 1.0, # Normal degradation rate
        'optimal_temp': 85  # degrees C
    },
    'HARD': {
        'max_life': 40, # Maximum life in laps
        'grip_level': 0.7, # Maximum grip
        'deg_rate': 0.8, # Lower degradation rate
        'optimal_temp': 80  # degrees C
    }
}

@dataclass
class TrackData:
    """Track configuration data."""
    tire_deg_factor: float  # Base degradation multiplier
    track_type: str        # Track type (High-speed, Technical, Street)
    track_evolution: float # Evolution rate per lap
    overtaking_diff: float # Overtaking difficulty (lower is easier)
    safety_car_prob: float # Safety car probability
    total_laps: int       # Standard race distance in laps

# Track characteristics from historical data
TRACK_CHARACTERISTICS = {
    'Monza': {
        'tire_deg_factor': 1.3,  # Highest due to max speeds
        'track_type': 'High-speed',
        'track_evolution': 0.08,  # High rubber laid from speeds
        'overtaking_diff': 0.25,  # Easiest - DRS zones
        'safety_car_prob': 0.35,  # First chicane incidents
        'total_laps': 53         # 5.793 km circuit length
    },
    'Spa': {
        'tire_deg_factor': 1.25,  # High speeds but flowing
        'track_type': 'High-speed',
        'track_evolution': 0.09,  # Longest track, more rubber
        'overtaking_diff': 0.30,  # Multiple zones but technical
        'safety_car_prob': 0.40,  # Weather + Eau Rouge
        'total_laps': 44         # 7.004 km circuit length
    },
    'Silverstone': {
        'tire_deg_factor': 1.2,   # Mix of high speed/technical
        'track_type': 'Technical',
        'track_evolution': 0.07,  # Balanced rubber laying
        'overtaking_diff': 0.35,  # Good but technical
        'safety_car_prob': 0.30,  # Wider runoffs
        'total_laps': 52         # 5.891 km circuit length
    }
}

# Track type characteristics
TRACK_TYPE_DATA = {
    'Street': {
        'sc_prob_range': (0.50, 0.80),
        'evolution_range': (0.14, 0.18),
        'pit_stops': (1, 2),
        'strategy': 'Prioritize track position'
    },
    'High-speed': {
        'sc_prob_range': (0.20, 0.45),
        'evolution_range': (0.07, 0.10),
        'pit_stops': (2, 3),
        'strategy': 'Allow aggressive strategies'
    },
    'Technical': {
        'sc_prob_range': (0.20, 0.35),
        'evolution_range': (0.06, 0.09),
        'pit_stops': (2, 2),
        'strategy': 'Balance position and pace'
    }
}

class PitWindowPredictor:
    """ML-based predictor for optimal pit windows."""
    
    def __init__(self):
        """Initialize the pit window predictor."""
        # Initialize the LightGBM Regressor model with predefined hyperparameters
        self.model = LGBMRegressor(
            n_estimators=200,        # Number of boosting iterations (trees)
            learning_rate=0.05,      # Conservative learning rate for stability
            max_depth=6,             # Maximum tree depth to control model complexity
            num_leaves=20,           # Number of leaves in each tree
            min_child_samples=30,    # Minimum samples per leaf to prevent overfitting
            feature_fraction=0.8,    # Fraction of features used for each tree
            bagging_fraction=0.8,    # Fraction of data used for each tree (row subsampling)
            bagging_freq=5,          # Frequency of applying bagging
            random_state=42          # Ensures reproducibility of results
        )
        
        # Initialize a StandardScaler for feature scaling (normalization)
        self.scaler = StandardScaler()
        
        # Flag to track whether the model has been trained
        self.is_trained = False

    
    def prepare_features(self, data: Dict) -> np.ndarray:
        """Prepare features for pit window prediction."""
        track_data = data['track_data']
        current_lap = data['current_lap']
        tire_age = data['tire_age']
        compound = data.get('compound', 'MEDIUM')  # Default to MEDIUM if not specified
        
        # Get tire compound characteristics
        tire_info = TIRE_COMPOUNDS[compound]
        
        # Calculate track evolution effect with starting value as 1
        evolution_factor = 1.0 + (track_data.track_evolution * current_lap / 10) # Divide by 10 to normalize
        
        # Basic features
        features = [
            data['current_position'],
            data['gap_ahead'],
            data['gap_behind'], 
            track_data.overtaking_diff,
            track_data.safety_car_prob, 
            track_data.total_laps - current_lap,  # remaining laps
            tire_age,
            track_data.track_evolution,
            track_data.tire_deg_factor
        ]
        
        # Engineered features
        features.extend([
            # Tire compound characteristics
            tire_info['grip_level'],
            tire_info['deg_rate'],
            tire_age / tire_info['max_life'],  # Tire life percentage (normalized)
            
            # Race progress features
            current_lap / track_data.total_laps,  # Race progress (normalized)
            evolution_factor,  # Track evolution effect
            
            # Track type encoding
            1.0 if track_data.track_type == 'High-speed' else 0.0,
            1.0 if track_data.track_type == 'Technical' else 0.0,
            1.0 if track_data.track_type == 'Street' else 0.0,
            
            # Combined characteristics
            track_data.tire_deg_factor * track_data.track_evolution,
            track_data.overtaking_diff * track_data.safety_car_prob,
            tire_info['deg_rate'] * track_data.tire_deg_factor,  # Combined tire degradation
            
            # Strategy indicators
            1.0 if track_data.overtaking_diff < 0.3 else 0.0,  # Easy overtaking
            1.0 if track_data.safety_car_prob > 0.35 else 0.0,  # High SC risk
            1.0 if current_lap < track_data.total_laps * 0.3 else 0.0,  # Early race
            1.0 if current_lap > track_data.total_laps * 0.7 else 0.0   # Late race
        ])
        
        features = np.array(features).reshape(1, -1)
        return self.scaler.transform(features) if self.is_trained else features
    
    def train(self, historical_data: List[Dict]):
        """Train the pit window prediction model."""
        X = np.array([self.prepare_features(data)[0] for data in historical_data])
        y = np.array([data['optimal_pit_lap'] for data in historical_data])
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict_window(self, race_data: Dict) -> Tuple[int, int]:
        """
        Predict optimal pit window based on current race situation.
        
        Args:
            race_data: Dictionary containing:
                - current_position: int
                - gap_ahead: float (seconds)
                - gap_behind: float (seconds)
                - track_data: TrackData
                - current_lap: int
                - tire_age: int
                - compound: str (optional, defaults to 'MEDIUM')
        
        Returns:
            Tuple of (window_start, window_end) in laps
        """
        if not self.is_trained:
            # Fallback to heuristic-based prediction if model isn't trained
            return self._heuristic_prediction(race_data)
        
        features = self.prepare_features(race_data)
        optimal_lap = int(self.model.predict(features)[0])
        
        # Adjust window based on track characteristics
        track_data = race_data['track_data']
        compound = race_data.get('compound', 'MEDIUM')
        tire_info = TIRE_COMPOUNDS[compound]
        
        # Base window size depends on track type and tire compound
        if track_data.track_type == 'High-speed':
            window_size = 4  # Wider windows due to overtaking opportunities
        elif track_data.track_type == 'Street':
            window_size = 2  # Narrow windows due to track position importance
        else:
            window_size = 3  # Balanced window for technical tracks
        
        # Adjust window based on tire compound
        if compound == 'SOFT':
            window_size = max(2, window_size - 1)  # Narrower window for softs
        elif compound == 'HARD':
            window_size += 1  # Wider window for hards
        
        # Further adjust based on track characteristics
        if track_data.overtaking_diff < 0.3:
            window_size += 1  # Easier overtaking allows wider windows
        if track_data.safety_car_prob > 0.35:
            window_size += 1  # High SC probability suggests flexible windows
        
        window_start = max(race_data['current_lap'] + 1, optimal_lap - window_size)
        window_end = min(track_data.total_laps - 1, optimal_lap + window_size)
        
        return window_start, window_end
    
    def _heuristic_prediction(self, race_data: Dict) -> Tuple[int, int]:
        """Fallback heuristic-based prediction when model isn't trained."""
        track_data = race_data['track_data']
        current_lap = race_data['current_lap']
        compound = race_data.get('compound', 'MEDIUM')
        tire_age = race_data['tire_age']
        
        # Get base stint length from tire compound
        base_window = TIRE_COMPOUNDS[compound]['max_life']
        
        # Adjust for track degradation
        base_window = int(base_window * (1 / track_data.tire_deg_factor))
        
        # Add evolution bonus
        if track_data.track_evolution >= 0.09:
            base_window += 3
        elif track_data.track_evolution >= 0.08:
            base_window += 2
        else:
            base_window += 1
        
        # Calculate optimal pit lap
        optimal_lap = current_lap + (base_window - tire_age)
        
        # Ensure within race distance
        optimal_lap = max(current_lap + 1, min(optimal_lap, track_data.total_laps - 5))
        
        # Consider track characteristics
        if track_data.tire_deg_factor > 1.25:
            optimal_lap = max(current_lap + 1, optimal_lap - 2)  # Earlier stops for high deg
        
        if track_data.track_evolution >= 0.09:
            optimal_lap = min(track_data.total_laps - 5, optimal_lap + 1)  # Later stops possible
        
        # Define window around optimal lap
        window_size = 3 if track_data.track_type == 'High-speed' else 2
        window_start = max(current_lap + 1, optimal_lap - window_size)
        window_end = min(track_data.total_laps - 1, optimal_lap + window_size)
        
        return window_start, window_end

class TrackPredictor:
    """Track characteristics predictor using historical data and ML."""
    
    def __init__(self):
        """Initialize predictors."""
        self.pit_window_predictor = PitWindowPredictor()
    
    def predict_from_characteristics(self, track_data: Dict) -> TrackData:
        """Create TrackData object from historical characteristics."""
        return TrackData(
            tire_deg_factor=track_data['tire_deg_factor'],
            track_type=track_data['track_type'],
            track_evolution=track_data['track_evolution'],
            overtaking_diff=track_data['overtaking_diff'],
            safety_car_prob=track_data['safety_car_prob'],
            total_laps=track_data['total_laps']
        )
    
    def predict_pit_window(self, race_data: Dict) -> Tuple[int, int]:
        """Predict optimal pit window using ML model."""
        return self.pit_window_predictor.predict_window(race_data)
