"""
F1 Pit Stop Analyzer

Uses historical FastF1 data to analyze and predict optimal pit stop windows
based on track characteristics and tire performance for dry race conditions.
"""

import fastf1
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

# Enable FastF1 cache
fastf1.Cache.enable_cache('cache')

@dataclass
class TireCompound:
    """Tire compound characteristics for dry conditions."""
    max_life: int
    grip_level: float
    degradation_rate: float
    warning_age: int
    critical_age: int

@dataclass
class TrackConfig:
    """Track configuration and characteristics."""
    name: str
    length: float
    laps: int
    pit_loss_time: float
    tire_deg_factor: float
    track_evolution: float
    safety_car_prob: float
    overtaking_difficulty: float

# Define track configurations based on 2022-2023 data
SIMILAR_TRACKS = {
    'monza': TrackConfig(
        name="Monza",
        length=5.793,
        laps=53,
        pit_loss_time=21.5,
        tire_deg_factor=1.3,      # Highest due to max speeds
        track_evolution=0.08,     # High rubber laid from speeds
        safety_car_prob=0.35,     # First chicane incidents
        overtaking_difficulty=0.25 # Easiest - DRS zones
    ),
    'spa': TrackConfig(
        name="Spa",
        length=7.004,
        laps=44,
        pit_loss_time=20.0,
        tire_deg_factor=1.25,     # High speeds but flowing
        track_evolution=0.09,     # Longest track, more rubber
        safety_car_prob=0.40,     # Weather + Eau Rouge
        overtaking_difficulty=0.30 # Multiple zones but technical
    ),
    'silverstone': TrackConfig(
        name="Silverstone",
        length=5.891,
        laps=52,
        pit_loss_time=23.0,
        tire_deg_factor=1.2,      # Mix of high speed/technical
        track_evolution=0.07,     # Balanced rubber laying
        safety_car_prob=0.30,     # Wider runoffs
        overtaking_difficulty=0.35 # Good but technical
    )
}

# Define tire compound characteristics based on real F1 data
TIRE_COMPOUNDS = {
    'SOFT': TireCompound(
        max_life=20,        # Maximum life ~15-20 laps
        grip_level=1.0,     # Highest grip
        degradation_rate=0.12,  # High degradation
        warning_age=12,     # 60% of max life
        critical_age=16     # 80% of max life
    ),
    'MEDIUM': TireCompound(
        max_life=30,        # Maximum life ~25-30 laps
        grip_level=0.85,    # Balanced grip
        degradation_rate=0.08,  # Moderate degradation
        warning_age=20,     # ~67% of max life
        critical_age=25     # ~83% of max life
    ),
    'HARD': TireCompound(
        max_life=40,        # Maximum life ~35-40 laps
        grip_level=0.75,    # Lower grip level
        degradation_rate=0.05,  # Low degradation
        warning_age=30,     # 75% of max life
        critical_age=35     # ~88% of max life
    )
}

class FuelEffect:
    """Fuel effect characteristics for different tire compounds."""
    def __init__(self, base_effect: float, sensitivity: float, weight_factor: float, performance_impact: float):
        self.base_effect = base_effect  # Base fuel effect multiplier
        self.sensitivity = sensitivity   # How sensitive the compound is to fuel load
        self.weight_factor = weight_factor  # Impact of fuel weight on tire performance
        self.performance_impact = performance_impact  # Overall performance impact

# Define fuel effects for each compound
FUEL_EFFECTS = {
    'SOFT': FuelEffect(2.0, 0.02, 1.6, 1.55),    # Most sensitive to fuel load
    'MEDIUM': FuelEffect(1.5, 0.015, 1.3, 1.3),  # Balanced sensitivity
    'HARD': FuelEffect(1.0, 0.01, 0.8, 1.25)     # Least sensitive to fuel load
}

class PitStopAnalyzer:
    """
    Analyzes historical race data to predict optimal pit stop windows.
    """
    
    def __init__(self, track_name: str):
        """Initialize PitStopAnalyzer with track configuration."""
        if track_name.lower() not in SIMILAR_TRACKS:
            raise ValueError(f"Unknown track: {track_name}")
            
        self.track_name = track_name
        self.track_data = SIMILAR_TRACKS[track_name.lower()]
        self.current_compound = None
        self.race_data = {}
        self.pit_stop_data = {}
        self.total_fuel = 110.0  # Standard F1 fuel load in kg
        
    def load_race_data(self, year: int) -> None:
        """
        Load race data for a specific year and track.
        
        Args:
            year: Year of the race data to load
        """
        try:
            # Load race session data
            race = fastf1.get_session(year, self.track_name, 'R')
            race.load()
            
            # Store race data
            self.race_data[year] = race
            
            # Get pit stop data
            pit_stops = []
            for driver in race.drivers:
                try:
                    driver_stops = race.laps.pick_driver(driver).get_car_data()
                    pit_stops.extend([{
                        'driver': driver,
                        'lap': stop.lap,
                        'compound': stop.compound,
                        'time': stop.time
                    } for stop in driver_stops if hasattr(stop, 'pit_in_time')])
                except Exception as e:
                    logging.warning(f"Could not get pit stop data for driver {driver}: {e}")
                    continue
            
            # Store pit stop data
            self.pit_stop_data[year] = pit_stops
            
        except Exception as e:
            logging.warning(f"Could not load {year} {self.track_name} GP data: {e}")
            
    def calculate_fuel_effect(self, current_lap: int, compound: str) -> float:
        """
        Calculate the effect of fuel load on tire performance.
        
        Args:
            current_lap: Current lap number
            compound: Current tire compound
            
        Returns:
            float: Fuel effect multiplier (higher means more impact)
        """
        if compound not in FUEL_EFFECTS:
            raise ValueError(f"Unknown compound: {compound}")
            
        # Calculate current fuel load based on lap number
        fuel_per_lap = self.total_fuel / max(1, self.track_data.laps)  # Prevent division by zero
        current_fuel = max(0, self.total_fuel - (current_lap * fuel_per_lap))
        
        # Get compound-specific fuel effects
        effect = FUEL_EFFECTS[compound]
        
        # Calculate base fuel effect (normalized to 1.0)
        fuel_ratio = current_fuel / self.total_fuel if self.total_fuel > 0 else 0
        fuel_effect = (fuel_ratio * effect.base_effect) + 0.83
        
        # Apply compound-specific sensitivities (capped)
        sensitivity_impact = min(0.5, effect.sensitivity * current_fuel)  # Cap sensitivity impact
        fuel_effect *= (1 + sensitivity_impact)
        
        # Factor in weight impact (normalized)
        weight_impact = (fuel_ratio * effect.weight_factor) / 2  # Normalize weight impact
        
        # Calculate final effect with performance impact
        final_effect = min(1.5, fuel_effect + weight_impact) * effect.performance_impact
        
        # Ensure result is between 1.0 and 2.0
        return max(1.0, min(2.0, final_effect))
        
    def _analyze_tire_degradation(self, compound: str, tire_age: int, current_lap: int) -> float:
        """
        Calculate tire degradation based on compound, age, track characteristics and fuel load.
        
        Args:
            compound: Current tire compound
            tire_age: Age of current tires in laps
            current_lap: Current lap number
            
        Returns:
            float: Tire degradation percentage (0.0 - 1.0)
        """
        # Get compound characteristics
        compound_data = TIRE_COMPOUNDS[compound]
        
        # Base degradation from tire age
        base_deg = tire_age / max(1, compound_data.max_life)  # Prevent division by zero
        
        # Track specific degradation factor
        track_factor = self.track_data.tire_deg_factor
        
        # Track evolution benefit (reduces degradation)
        evolution_benefit = min(0.5, self.track_data.track_evolution * current_lap)  # Cap evolution benefit
        
        # Calculate fuel effect
        fuel_effect = self.calculate_fuel_effect(current_lap, compound)
        
        # Calculate final degradation including fuel effect
        degradation = base_deg * track_factor * (1.0 - evolution_benefit) * (fuel_effect / 2)  # Normalize fuel effect
        
        # Cap at 100%
        return min(1.0, max(0.0, degradation))
        
    def _assess_risks(self, compound: str, tire_age: int, degradation: float) -> str:
        """
        Assess risk level based on tire state and track characteristics.
        
        Args:
            compound: Current tire compound
            tire_age: Age of current tires in laps
            degradation: Current tire degradation
            
        Returns:
            str: Risk level (low, medium, high, critical)
        """
        # Get compound characteristics
        compound_data = TIRE_COMPOUNDS[compound]
        
        # Base risk from tire age
        if tire_age >= compound_data.critical_age:
            return "critical"
        elif tire_age >= compound_data.warning_age:
            return "high"
            
        # Risk from degradation
        if degradation >= 0.9:
            return "critical"
        elif degradation >= 0.7:
            return "high"
        elif degradation >= 0.5:
            return "medium"
            
        return "low"
        
    def _calculate_pit_window(self, current_lap: int, remaining_laps: int, degradation: float) -> Dict[str, int]:
        """
        Calculate optimal pit window based on current situation.
        
        Args:
            current_lap: Current lap number
            remaining_laps: Number of laps remaining
            degradation: Current tire degradation
            
        Returns:
            Dict containing start, optimal, and end lap for pit window
        """
        # Base window calculation
        optimal_lap = max(current_lap + 5, min(
            current_lap + int(15 * (1 - degradation)),  # More laps if less degraded
            self.track_data.laps - 10  # Don't pit too late
        ))
        
        # Adjust window based on track characteristics
        window_size = int(10 * (1 - self.track_data.overtaking_difficulty))  # Larger window if easier overtaking
        
        # Calculate start and end of window
        start_lap = max(current_lap + 1, optimal_lap - window_size)
        end_lap = min(self.track_data.laps - 5, optimal_lap + window_size)
        
        # Ensure valid window
        if start_lap >= end_lap:
            start_lap = current_lap + 1
            optimal_lap = current_lap + 3
            end_lap = current_lap + 5
            
        return {
            'start_lap': start_lap,
            'optimal_lap': optimal_lap,
            'end_lap': end_lap
        }
        
    def summarize_situation(self, current_lap: int, compound: str, tire_age: int) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the current race situation.
        
        Args:
            current_lap: Current lap number
            compound: Current tire compound
            tire_age: Current tire age
            
        Returns:
            Dict containing situation summary
        """
        # Calculate tire degradation with fuel effect
        degradation = self._analyze_tire_degradation(compound, tire_age, current_lap)
        
        # Get risk assessment
        risk_level = self._assess_risks(compound, tire_age, degradation)
        
        # Calculate remaining laps
        remaining_laps = self.track_data.laps - current_lap
        
        # Get pit window
        pit_window = self._calculate_pit_window(current_lap, remaining_laps, degradation)
        
        # Calculate fuel effect
        fuel_effect = self.calculate_fuel_effect(current_lap, compound)
        
        # Generate warnings
        warnings = []
        
        # Tire age warnings
        compound_data = TIRE_COMPOUNDS[compound]
        if tire_age >= compound_data.critical_age:
            warnings.append(f"CRITICAL: {compound} tires are beyond critical age")
        elif tire_age >= compound_data.warning_age:
            warnings.append(f"WARNING: {compound} tires are nearing critical age")
            
        # Degradation warnings
        if degradation >= 0.9:
            warnings.append("CRITICAL: Extreme tire degradation")
        elif degradation >= 0.7:
            warnings.append("WARNING: High tire degradation")
            
        # Fuel effect warnings
        if fuel_effect > 1.4:
            warnings.append("CAUTION: High fuel load significantly affecting tire performance")
        elif fuel_effect > 1.2:
            warnings.append("INFO: Moderate fuel effect on tire performance")
            
        # Track specific warnings
        if self.track_data.safety_car_prob > 0.4:
            warnings.append("High safety car probability - consider flexible strategy")
            
        if self.track_data.overtaking_difficulty > 0.7 and remaining_laps > 10:
            warnings.append("Difficult overtaking - track position crucial")
            
        # Calculate expected stint lengths
        stint_lengths = {}
        for comp in TIRE_COMPOUNDS:
            max_life = float(TIRE_COMPOUNDS[comp].max_life)  # Convert to float
            # Adjust for track characteristics and fuel effect
            fuel_impact = self.calculate_fuel_effect(current_lap, comp)
            # Use floor division to get integer result
            adjusted_life = int(max_life / (self.track_data.tire_deg_factor * fuel_impact))
            # Ensure minimum stint length
            stint_lengths[comp] = max(5, adjusted_life)
            
        # Generate strategy options
        strategy_options = []
        
        # Basic strategy options
        if remaining_laps <= stint_lengths['HARD']:
            strategy_options.append("Consider running to end on current tires")
            
        if self.track_data.overtaking_difficulty < 0.4:
            strategy_options.append("Track good for overtaking - aggressive strategy possible")
            
        if self.track_data.safety_car_prob > 0.4:
            strategy_options.append("High safety car chance - maintain strategy flexibility")
            
        # Fuel-based strategy options
        if current_lap < 10 and fuel_effect > 1.3:
            strategy_options.append("High fuel load - consider conservative driving style")
        elif current_lap > self.track_data.laps - 15:
            strategy_options.append("Low fuel load - more aggressive driving possible")
            
        # Return comprehensive summary
        return {
            'current_situation': {
                'lap': current_lap,
                'compound': compound,
                'tire_age': tire_age,
                'degradation': degradation,
                'fuel_effect': fuel_effect,
                'risk_level': risk_level
            },
            'track_characteristics': {
                'length': self.track_data.length,
                'laps': self.track_data.laps,
                'tire_deg_factor': self.track_data.tire_deg_factor,
                'track_evolution': self.track_data.track_evolution,
                'overtaking_difficulty': self.track_data.overtaking_difficulty,
                'safety_car_prob': self.track_data.safety_car_prob
            },
            'pit_window': pit_window,
            'stint_lengths': stint_lengths,
            'warnings': warnings,
            'strategy_options': strategy_options
        }
