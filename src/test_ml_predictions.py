"""
Test script to compare ML predictions with baseline track configurations.
"""
from models.track_model import TrackPredictor, TRACK_CHARACTERISTICS, TRACK_TYPE_DATA
from pitstop_analyzer import PitStopAnalyzer, TIRE_COMPOUNDS, FUEL_EFFECTS
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, List

def normalize_track_name(track_name: str) -> str:
    """Normalize track name for consistent lookup."""
    track_map = {
        'monza': 'Monza',
        'spa': 'Spa',
        'silverstone': 'Silverstone'
    }
    return track_map.get(track_name.lower(), track_name)

def test_fuel_effects(track_name: str):
    """Test and visualize fuel effects on tire degradation."""
    track_name = normalize_track_name(track_name)
    if track_name not in TRACK_CHARACTERISTICS:
        print(f"Track {track_name} not found in database")
        return
    
    # Initialize analyzers
    predictor = TrackPredictor()
    analyzer = PitStopAnalyzer(predictor)
    analyzer.track_data = predictor.predict_from_characteristics(TRACK_CHARACTERISTICS[track_name])
    
    # Test fuel effects across laps
    laps = range(0, 50)
    compounds = list(TIRE_COMPOUNDS.keys())  # Use compounds from constants
    fuel_effects = {compound: [] for compound in compounds}
    
    for lap in laps:
        for compound in compounds:
            effect = analyzer.calculate_fuel_effect(lap, compound)
            fuel_effects[compound].append(effect)
    
    # Plot fuel effects with consistent compound case
    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')
    
    colors = {
        'SOFT': '#FF3333',  # Bright red
        'MEDIUM': '#FFFF33',  # Bright yellow
        'HARD': '#FFFFFF'  # White
    }
    
    line_styles = {
        'SOFT': '-',
        'MEDIUM': '--',
        'HARD': '-.'
    }
    
    for compound in compounds:
        plt.plot(laps, fuel_effects[compound], 
                label=compound.capitalize(), 
                color=colors[compound],
                linestyle=line_styles[compound],
                linewidth=2.5)
    
    plt.title(f'Fuel Effect on Tire Degradation - {track_name}', fontsize=14, pad=20)
    plt.xlabel('Lap Number', fontsize=12)
    plt.ylabel('Degradation Multiplier', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, framealpha=0.8)
    plt.tight_layout()
    
    # Set background color
    ax = plt.gca()
    ax.set_facecolor('#1C1C1C')  # Dark gray background
    plt.gcf().set_facecolor('#1C1C1C')
    
    # Adjust tick parameters for better visibility
    plt.tick_params(axis='both', colors='white', labelsize=10)
    
    plt.savefig('fuel_effects.png', 
                facecolor='#1C1C1C',
                edgecolor='none',
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    # Print analysis
    print(f"\nFuel Effect Analysis for {track_name}")
    print("=" * 50)
    
    for compound in compounds:
        initial_effect = fuel_effects[compound][0]
        mid_effect = fuel_effects[compound][25]
        final_effect = fuel_effects[compound][-1]
        
        print(f"\n{compound.capitalize()} Compound:")
        print(f"Initial degradation multiplier: {initial_effect:.3f}")
        print(f"Mid-race degradation multiplier: {mid_effect:.3f}")
        print(f"End-race degradation multiplier: {final_effect:.3f}")
        print(f"Total effect reduction: {((initial_effect - final_effect) / initial_effect) * 100:.1f}%")

def create_test_race_data(track_name: str) -> Dict:
    """Create test race data for predictions."""
    track_data = TRACK_CHARACTERISTICS[track_name]
    return {
        'current_position': 5,  # Mid-field position
        'gap_ahead': 2.5,      # Typical gap in seconds
        'gap_behind': 1.8,     # Typical gap in seconds
        'track_data': TrackPredictor().predict_from_characteristics(track_data),
        'current_lap': 15,     # Early-mid race
        'tire_age': 8,         # Mid-stint tire age
        'compound': 'MEDIUM'   # Default to medium compound
    }

def create_historical_data(track_name: str, n_samples: int = 500) -> List[Dict]:
    """Create synthetic historical data for training."""
    track_data = TRACK_CHARACTERISTICS[track_name]
    track_obj = TrackPredictor().predict_from_characteristics(track_data)
    
    historical_data = []
    compounds = [str(c) for c in TIRE_COMPOUNDS.keys()]  # Convert to regular Python strings
    
    for _ in range(n_samples):
        # Randomize race situations
        position = np.random.randint(1, 20)
        gap_ahead = np.random.uniform(0.5, 4.0)
        gap_behind = np.random.uniform(0.5, 4.0)
        current_lap = np.random.randint(10, 40)
        tire_age = np.random.randint(5, 20)
        compound = str(np.random.choice(compounds))  # Convert numpy string to Python string
        
        # Calculate optimal pit lap based on track characteristics
        base_window = TIRE_COMPOUNDS[compound]['max_life']
        base_window = int(base_window * (1 / track_data['tire_deg_factor']))
        
        # Add evolution bonus
        if track_data['track_evolution'] >= 0.09:
            base_window += 3
        elif track_data['track_evolution'] >= 0.08:
            base_window += 2
        else:
            base_window += 1
        
        # Add some noise to optimal lap
        optimal_lap = current_lap + (base_window - tire_age)
        optimal_lap += np.random.randint(-2, 3)  # Add noise
        
        # Ensure optimal lap is within race distance
        optimal_lap = max(current_lap + 1, min(optimal_lap, track_data['total_laps'] - 5))
        
        # Consider tire degradation
        if track_data['tire_deg_factor'] > 1.25:
            optimal_lap = max(current_lap + 1, optimal_lap - 2)  # Earlier stops for high deg
        
        # Consider track evolution
        if track_data['track_evolution'] >= 0.09:
            optimal_lap = min(track_data['total_laps'] - 5, optimal_lap + 1)  # Later stops possible
        
        historical_data.append({
            'current_position': position,
            'gap_ahead': gap_ahead,
            'gap_behind': gap_behind,
            'track_data': track_obj,
            'current_lap': current_lap,
            'tire_age': tire_age,
            'compound': compound,
            'optimal_pit_lap': optimal_lap
        })
    
    return historical_data

def format_track_stats(stats: Dict[str, Any]) -> str:
    """Format track statistics for display."""
    return (
        f"Tire Degradation: {stats['tire_deg_factor']:.3f}\n"
        f"Track Evolution: {stats['track_evolution']:.3f}\n"
        f"Safety Car Prob: {stats['safety_car_prob']:.3f}\n"
        f"Overtaking Diff: {stats['overtaking_diff']:.3f}"
    )

def validate_track_type(track_name: str, track_data: Dict):
    """Validate track data against track type characteristics."""
    track_type = track_data['track_type']
    type_data = TRACK_TYPE_DATA[track_type]
    
    print(f"\nValidating {track_name} as {track_type} track:")
    print("-" * 50)
    
    # Check safety car probability
    sc_min, sc_max = type_data['sc_prob_range']
    sc_valid = sc_min <= track_data['safety_car_prob'] <= sc_max
    print(f"Safety Car Probability: {track_data['safety_car_prob']:.2f}")
    print(f"Expected Range: {sc_min:.2f} - {sc_max:.2f}")
    print(f"Status: {'✓' if sc_valid else '✗'}\n")
    
    # Check track evolution
    evo_min, evo_max = type_data['evolution_range']
    evo_valid = evo_min <= track_data['track_evolution'] <= evo_max
    print(f"Track Evolution: {track_data['track_evolution']:.3f}")
    print(f"Expected Range: {evo_min:.3f} - {evo_max:.3f}")
    print(f"Status: {'✓' if evo_valid else '✗'}\n")
    
    # Print strategy implications
    print("Strategy Implications:")
    print(f"- {type_data['strategy']}")
    print(f"- Expected pit stops: {type_data['pit_stops'][0]}-{type_data['pit_stops'][1]}")

def test_ml_predictions(track_name: str):
    """Test ML predictions against baseline values."""
    track_name = normalize_track_name(track_name)
    if track_name not in TRACK_CHARACTERISTICS:
        print(f"Track {track_name} not found in database")
        print("Available tracks:", ", ".join(TRACK_CHARACTERISTICS.keys()))
        return
    
    print(f"\nTesting ML predictions for {track_name.upper()} GP")
    print("=" * 50)
    
    # Initialize predictor
    predictor = TrackPredictor()
    
    # Get track data
    track_data = TRACK_CHARACTERISTICS[track_name]
    print("\nTrack Configuration:")
    print("-" * 20)
    print(format_track_stats(track_data))
    
    # Test fuel effects
    test_fuel_effects(track_name)
    
    # Validate track type
    validate_track_type(track_name, track_data)
    
    # Create historical data and train model
    print("\nTraining ML model...")
    historical_data = create_historical_data(track_name)
    predictor.pit_window_predictor.train(historical_data)
    
    # Test predictions for each compound
    print("\nTesting predictions...")
    test_data = create_test_race_data(track_name)
    
    for compound in TIRE_COMPOUNDS.keys():
        test_data['compound'] = compound
        window_start, window_end = predictor.predict_pit_window(test_data)
        
        print(f"\n{compound.capitalize()} compound prediction for lap {test_data['current_lap']}:")
        print(f"Window Start: Lap {window_start}")
        print(f"Window End: Lap {window_end}")
        print(f"Window Size: {window_end - window_start} laps")
    
    # Print strategy considerations based on track type
    print("\nStrategy Considerations:")
    track_type = track_data['track_type']
    if track_type == 'High-speed':
        print("- Wider pit windows due to overtaking opportunities")
        print("- Can be more aggressive with strategy")
        if track_data['tire_deg_factor'] > 1.25:
            print("- Consider earlier stops due to high tire degradation")
        if track_data['track_evolution'] >= 0.09:
            print("- Later stops possible due to high track evolution")
    elif track_type == 'Technical':
        print("- Balance track position with pace")
        print("- Standard pit windows")
        if track_data['overtaking_diff'] > 0.3:
            print("- Track position critical due to overtaking difficulty")
    else:  # Street
        print("- Prioritize track position")
        print("- Conservative pit windows")
        print("- Safety car probability high - stay flexible")

def main():
    """Test ML predictions for all tracks."""
    logging.basicConfig(level=logging.INFO)
    
    tracks = ["Monza", "Spa", "Silverstone"]
    for track in tracks:
        test_ml_predictions(track)
        print("\n")
    
    choice = input("Would you like to test another track? (y/n): ")
    if choice.lower() == 'y':
        track = input("Enter track name: ")
        test_ml_predictions(track)

if __name__ == "__main__":
    main()
