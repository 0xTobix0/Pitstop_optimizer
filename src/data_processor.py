"""
Data processor for F1 pitstop optimization.
"""
import fastf1
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from config import TRACK_PARAMS, RACE_CONFIG

class F1DataProcessor:
    def __init__(self):
        """Initialize data processor."""
        fastf1.Cache.enable_cache('cache')
        
    def calculate_track_characteristics(self, track: str, years: List[int]) -> Dict[str, Any]:
        """Calculate track characteristics from historical data."""
        if track not in TRACK_PARAMS:
            raise ValueError(f"Invalid track: {track}")
            
        # Load race data for specified years
        races = []
        for year in years:
            try:
                race = fastf1.get_session(year, track, 'R')
                race.load()
                races.append(race)
            except Exception as e:
                print(f"Warning: Could not load {year} {track} race: {str(e)}")
                
        if not races:
            raise ValueError(f"No valid race data found for {track}")
            
        # Initialize characteristics
        characteristics = {
            'total_laps': races[0].laps.LapNumber.max(),
            'fuel_per_lap': 0.0,  # Will calculate from track length
            'tire_deg_factor': 1.0,  # Will calculate from stint data
            'track_evolution': 0.0,  # Will calculate from lap time progression
            'overtaking_diff': 0.5,  # Will calculate from position changes
            'max_stint': {  # Will update based on track characteristics
                'HARD': 40,
                'MEDIUM': 30,
                'SOFT': 20
            }
        }
        
        # Calculate fuel consumption based on track length
        # Hardcoded track lengths in km
        TRACK_LENGTHS = {
            'Suzuka': 5.807,
            'Monza': 5.793,
            'Spa': 7.004,
            'Silverstone': 5.891
        }
        track_length = TRACK_LENGTHS.get(track, 5.0)  # Default to 5km if unknown
        characteristics['fuel_per_lap'] = track_length * RACE_CONFIG['fuel_per_km']
        
        # Analyze tire degradation from stint data
        all_stints = []
        for race in races:
            for driver in race.drivers:
                stints = self._get_driver_stints(race, driver)
                for stint in stints:
                    stint_data = self._analyze_stint(stint)
                    if stint_data:
                        all_stints.append(stint_data)
                        
        if all_stints:
            # Calculate average degradation rate
            deg_rates = [stint['DegRate'] for stint in all_stints]
            characteristics['tire_deg_factor'] = np.mean(deg_rates) / RACE_CONFIG['base_tire_deg_rate']
            
            # Update max stint lengths based on tire degradation
            for compound in characteristics['max_stint']:
                base_life = characteristics['max_stint'][compound]
                characteristics['max_stint'][compound] = int(base_life / characteristics['tire_deg_factor'])
                
        # Calculate track evolution
        lap_times = pd.concat([race.laps['LapTime'] for race in races])
        if not lap_times.empty:
            # Calculate percentage improvement per lap
            time_progression = lap_times.pct_change()
            characteristics['track_evolution'] = -np.mean(time_progression.dropna())
            
        # Calculate overtaking difficulty
        position_changes = []
        for race in races:
            for driver in race.drivers:
                driver_laps = race.laps.pick_driver(driver)
                if not driver_laps.empty:
                    changes = abs(driver_laps['Position'].diff().sum())
                    position_changes.append(changes)
                    
        if position_changes:
            # Normalize position changes to 0-1 scale where higher means harder to overtake
            avg_changes = np.mean(position_changes)
            characteristics['overtaking_diff'] = 1.0 / (1.0 + avg_changes/10)
            
        return characteristics
        
    def load_historical_data(self, years: List[int]) -> pd.DataFrame:
        """Load and process historical race data."""
        races = []
        for year in years:
            try:
                race = fastf1.get_session(year, 'Australian Grand Prix', 'R')
                race.load()
                races.append(race)
            except Exception as e:
                print(f"Warning: Could not load {year} Australian GP: {str(e)}")
                
        if not races:
            raise ValueError("No valid race data found")
            
        # Process each race
        all_stints = []
        for race in races:
            laps_df = race.laps
            
            # Get unique driver stints
            drivers = laps_df['Driver'].unique()
            for driver in drivers:
                driver_laps = laps_df[laps_df['Driver'] == driver].sort_values('LapNumber')
                
                # Find stint changes
                compound_changes = driver_laps['Compound'].ne(driver_laps['Compound'].shift()).fillna(True)
                stint_starts = driver_laps[compound_changes].index
                
                # Process each stint
                for i in range(len(stint_starts)):
                    start_idx = stint_starts[i]
                    end_idx = stint_starts[i+1] if i < len(stint_starts)-1 else driver_laps.index[-1]
                    
                    stint_laps = driver_laps.loc[start_idx:end_idx]
                    compound = stint_laps.iloc[0]['Compound']
                    
                    # Skip if compound is unknown
                    if pd.isna(compound) or compound not in ['SOFT', 'MEDIUM', 'HARD']:
                        continue
                        
                    # Calculate stint features
                    stint_length = len(stint_laps)
                    avg_lap_time = stint_laps['LapTime'].mean().total_seconds()
                    
                    # Get weather data
                    weather = race.weather_data
                    avg_track_temp = weather['TrackTemp'].mean()
                    avg_air_temp = weather['AirTemp'].mean()
                    avg_humidity = weather['Humidity'].mean()
                    
                    # Calculate remaining laps for each lap in stint
                    for lap_idx, lap in stint_laps.iterrows():
                        lap_num = lap['LapNumber']
                        tire_life = len(stint_laps.loc[:lap_idx])
                        position = lap['Position']
                        
                        # Apply compound-specific limits
                        max_remaining = {
                            'SOFT': min(15, 20 - tire_life),
                            'MEDIUM': min(25, 30 - tire_life),
                            'HARD': min(35, 40 - tire_life)
                        }[compound]
                        
                        # Calculate actual remaining laps
                        remaining_laps = len(stint_laps.loc[lap_idx:])
                        remaining_laps = min(remaining_laps, max_remaining)
                        
                        all_stints.append({
                            'LapNumber': lap_num,
                            'TyreLife': tire_life,
                            'Compound': compound,
                            'LapTime': avg_lap_time,
                            'TrackTemp': avg_track_temp,
                            'AirTemp': avg_air_temp,
                            'Humidity': avg_humidity,
                            'Position': position,
                            'RemainingLaps': remaining_laps
                        })
                        
        return pd.DataFrame(all_stints)
        
    def _get_driver_stints(self, race, driver) -> List[pd.DataFrame]:
        """Get list of stint DataFrames for a driver."""
        driver_laps = race.laps.pick_driver(driver)
        if driver_laps.empty:
            return []
            
        # Group laps by stint
        stints = []
        current_stint = []
        prev_compound = None
        
        for _, lap in driver_laps.iterrows():
            compound = lap['Compound']
            
            # Start new stint if compound changes
            if compound != prev_compound and prev_compound is not None:
                if current_stint:
                    stints.append(pd.DataFrame(current_stint))
                current_stint = []
                
            current_stint.append(lap)
            prev_compound = compound
            
        # Add final stint
        if current_stint:
            stints.append(pd.DataFrame(current_stint))
            
        return stints
        
    def _analyze_stint(self, stint: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a single stint for degradation and performance."""
        if stint['LapTime'].isna().any():
            return None
            
        compound = stint.iloc[0]['Compound']
        if not compound or compound not in ['SOFT', 'MEDIUM', 'HARD']:  # Only dry tires
            return None
            
        # Calculate lap time progression
        stint['LapTimeDelta'] = stint['LapTime'].diff()
        stint['NormalizedTime'] = stint['LapTime'] / stint['LapTime'].iloc[0]
        
        # Calculate degradation
        deg_rate = (stint['NormalizedTime'].iloc[-1] - 1) / len(stint)
        
        return {
            'Compound': compound,
            'StintLength': len(stint),
            'AvgLapTime': stint['LapTime'].mean(),
            'DegRate': deg_rate,
            'InitialLapTime': stint['LapTime'].iloc[0],
            'FinalLapTime': stint['LapTime'].iloc[-1],
            'Position': stint['Position'].mean(),
            'PositionChange': stint['Position'].iloc[-1] - stint['Position'].iloc[0]
        }
        
    def _process_stint_for_training(self, stint: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process a stint into training data format."""
        if stint['LapTime'].isna().any():
            return None
            
        compound = stint.iloc[0]['Compound']
        if not compound or compound not in ['SOFT', 'MEDIUM', 'HARD']:
            return None
            
        # Calculate stint length
        stint_length = len(stint)
        
        # Calculate features for each lap
        training_data = []
        
        for i in range(len(stint)):
            lap = stint.iloc[i]
            
            # Basic features
            features = {
                'LapNumber': lap['LapNumber'],
                'TyreLife': i,  # Laps since start of stint
                'Compound': compound,
                'LapTime': lap['LapTime'].total_seconds(),
                'TrackTemp': lap.get('TrackTemp', 25),  # Default if missing
                'AirTemp': lap.get('AirTemp', 20),
                'Humidity': lap.get('Humidity', 50),
                'Position': lap['Position'],
                'RemainingLaps': stint_length - i  # Target is remaining laps in stint
            }
            
            training_data.append(features)
        
        return training_data
        
    def _calculate_fuel_correction(self, lap_time: float, lap_number: int, fuel_per_lap: float) -> float:
        """Calculate fuel-corrected lap time."""
        initial_fuel = RACE_CONFIG['initial_fuel']
        current_fuel = max(0, initial_fuel - (lap_number * fuel_per_lap))
        fuel_penalty = RACE_CONFIG['fuel_penalty_per_kg'] * current_fuel
        return lap_time - fuel_penalty
