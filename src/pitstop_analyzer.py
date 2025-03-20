"""
Pit stop strategy analyzer for F1 races.
"""
import fastf1
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from models.track_model import TrackPredictor, TRACK_CHARACTERISTICS
import numpy as np

# Constants for tire compounds and effects
TIRE_COMPOUNDS = {
    'SOFT': {
        'max_life': 20,
        'deg_rate': 1.3,
        'grip_level': 1.0,
        'optimal_temp': 90
    },
    'MEDIUM': {
        'max_life': 30,
        'deg_rate': 1.0,
        'grip_level': 0.85,
        'optimal_temp': 85
    },
    'HARD': {
        'max_life': 40,
        'deg_rate': 0.8,
        'grip_level': 0.7,
        'optimal_temp': 80
    }
}

# Fuel load effects on tire degradation
FUEL_EFFECTS = {
    'initial_weight': 110,  # kg of fuel at start
    'consumption_rate': 1.8,  # kg per lap
    'recovery_rate': 0.03,   # Base rate for exponential decay
    'deg_multiplier': 1.4,   # Maximum impact on tire degradation
    'weight_sensitivity': {  # Impact of weight on different compounds
        'SOFT': 1.2,
        'MEDIUM': 1.0,
        'HARD': 0.9
    }
}

class PitStopAnalyzer:
    """
    Analyzes pit stop strategies based on track characteristics and tire compounds.
    """
    
    def __init__(self, track_predictor: TrackPredictor):
        """
        Initialize the analyzer.
        
        Args:
            track_predictor: TrackPredictor instance for getting track characteristics
        """
        self.track_predictor = track_predictor
        self.track_data = None
        self.session = None
    
    def load_race_data(self, year: int, track_name: str):
        """
        Load historical race data for analysis.
        
        Args:
            year: Year of the race
            track_name: Name of the track
        """
        try:
            self.session = fastf1.get_session(year, track_name, 'R')
            self.session.load()
            self.track_data = self.track_predictor.predict(self.session)
        except Exception as e:
            logging.error(f"Error loading race data: {str(e)}")
            raise
    
    def calculate_fuel_effect(self, lap_number: int, compound: str) -> float:
        """
        Calculate the effect of fuel load on tire performance using exponential decay.
        
        Args:
            lap_number: Current lap number
            compound: Tire compound (SOFT, MEDIUM, HARD)
        
        Returns:
            float: Adjustment factor for tire performance
        """
        # Calculate remaining fuel using exponential decay
        initial_effect = 1.0  # Maximum effect at race start
        recovery_rate = FUEL_EFFECTS['recovery_rate']
        
        # Apply compound-specific weight sensitivity
        weight_factor = FUEL_EFFECTS['weight_sensitivity'][compound]
        adjusted_rate = recovery_rate * weight_factor
        
        # Calculate fuel effect using exponential decay
        fuel_factor = initial_effect * np.exp(-lap_number * adjusted_rate)
        
        # Ensure fuel factor stays within reasonable bounds
        fuel_factor = max(0, min(1.0, fuel_factor))
        
        # Calculate compound-specific fuel effect
        base_deg = TIRE_COMPOUNDS[compound]['deg_rate']
        fuel_adj = 1.0 + (FUEL_EFFECTS['deg_multiplier'] - 1.0) * fuel_factor
        
        return base_deg * fuel_adj
    
    def analyze_stint_length(self, compound: str, start_lap: int) -> Dict:
        """
        Analyze optimal stint length for a given compound.
        
        Args:
            compound: Tire compound to analyze
            start_lap: Lap number at start of stint
        
        Returns:
            Dict containing stint analysis
        """
        max_life = TIRE_COMPOUNDS[compound]['max_life']
        base_deg = TIRE_COMPOUNDS[compound]['deg_rate']
        
        # Get track-specific adjustments
        track_factor = self.track_data.tire_deg_factor
        
        # Calculate fuel-adjusted degradation
        fuel_effect = self.calculate_fuel_effect(start_lap, compound)
        adjusted_deg = base_deg * track_factor * fuel_effect
        
        # Estimate optimal stint length
        optimal_length = int(max_life / adjusted_deg)
        
        return {
            'compound': compound,
            'optimal_length': optimal_length,
            'degradation_rate': adjusted_deg,
            'fuel_effect': fuel_effect
        }
    
    def get_optimal_strategy(self, race_laps: int, current_position: int, 
                           gaps_ahead: List[float], gaps_behind: List[float]) -> Dict:
        """
        Get optimal pit stop strategy based on current race situation.
        
        Args:
            race_laps: Total number of race laps
            current_position: Current race position
            gaps_ahead: Time gaps to cars ahead (seconds)
            gaps_behind: Time gaps to cars behind (seconds)
        
        Returns:
            Dict containing strategy recommendations
        """
        if not self.track_data:
            raise ValueError("Track data not loaded. Call load_race_data first.")
        
        strategy = {
            'recommended_stops': [],
            'tire_choices': [],
            'risks': [],
            'opportunities': []
        }
        
        # Early race strategy
        if race_laps >= 40:  # Still early in the race
            if self.track_data.track_evolution >= 0.08:
                strategy['tire_choices'].append('MEDIUM')
                strategy['risks'].append('Higher initial degradation')
                strategy['opportunities'].append('Track evolution benefit')
            else:
                strategy['tire_choices'].append('HARD')
                strategy['risks'].append('Lower initial pace')
                strategy['opportunities'].append('Strategic flexibility')
        
        # Mid race strategy
        if self.track_data.tire_deg_factor > 1.25:
            strategy['tire_choices'].append('HARD')
            strategy['risks'].append('High track degradation')
        elif self.track_data.safety_car_prob > 0.35:
            strategy['tire_choices'].extend(['MEDIUM', 'SOFT'])
            strategy['opportunities'].append('Safety car opportunity')
        
        # Late race strategy
        if race_laps <= 20:
            if self.track_data.overtaking_difficulty < 0.4:
                strategy['tire_choices'].append('SOFT')
                strategy['opportunities'].append('Overtaking potential')
            else:
                strategy['tire_choices'].append('MEDIUM')
                strategy['risks'].append('Track position critical')
        
        return strategy
