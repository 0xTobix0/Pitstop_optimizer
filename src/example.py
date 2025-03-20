"""
Simple F1 pit stop timing optimizer with ML-based pit window predictions.
"""
import argparse
import fastf1
import os
from models.track_model import TrackPredictor, TRACK_CHARACTERISTICS
from dataclasses import dataclass

# Configure FastF1 cache
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# Tire compounds based on historical data
TIRE_COMPOUNDS = {
    'SOFT': {
        'max_life': 20,
        'warning_age': 15,
        'grip_level': 1.0,
        'deg_rate': 1.3,
        'optimal_temp': 90
    },
    'MEDIUM': {
        'max_life': 30,
        'warning_age': 23,
        'grip_level': 0.85,
        'deg_rate': 1.0,
        'optimal_temp': 85
    },
    'HARD': {
        'max_life': 40,
        'warning_age': 32,
        'grip_level': 0.7,
        'deg_rate': 0.8,
        'optimal_temp': 80
    }
}

# Fuel effect parameters
FUEL_EFFECTS = {
    'initial_multiplier': 1.4,  # Maximum effect at race start
    'decay_rate': 0.02,        # Rate of effect reduction per lap
    'min_multiplier': 1.0      # Minimum effect at end of race
}

def get_user_input():
    """Get current race situation from user."""
    print("\nEnter current race situation:")
    
    while True:
        track_name = input("Track name (Monza/Spa/Silverstone): ").strip().capitalize()
        if track_name in TRACK_CHARACTERISTICS:
            break
        print("Please enter a valid track name")
    
    total_laps = TRACK_CHARACTERISTICS[track_name]['total_laps']
    while True:
        try:
            current_lap = int(input(f"Current lap number (1-{total_laps}): "))
            if 0 < current_lap <= total_laps:
                break
            print(f"Current lap must be between 1 and {total_laps}")
        except ValueError:
            print("Please enter valid numbers")
    
    while True:
        try:
            position = int(input("Current position: "))
            if 1 <= position <= 20:
                break
            print("Position must be between 1 and 20")
        except ValueError:
            print("Please enter a valid position")
    
    while True:
        try:
            gap_ahead = float(input("Gap to car ahead (seconds, -1 if none): "))
            if gap_ahead >= -1:
                break
            print("Gap must be >= -1")
        except ValueError:
            print("Please enter a valid gap")
    
    while True:
        try:
            gap_behind = float(input("Gap to car behind (seconds, -1 if none): "))
            if gap_behind >= -1:
                break
            print("Gap must be >= -1")
        except ValueError:
            print("Please enter a valid gap")
    
    while True:
        compound = input("Current tire compound (SOFT/MEDIUM/HARD): ").strip().upper()
        if compound in TIRE_COMPOUNDS:
            break
        print("Please enter a valid tire compound")
    
    while True:
        try:
            tire_age = int(input("Current tire age (laps): "))
            if tire_age >= 0:
                break
            print("Tire age must be 0 or positive")
        except ValueError:
            print("Please enter a valid number")
    
    return {
        'track_name': track_name,
        'current_lap': current_lap,
        'current_position': position,
        'gap_ahead': gap_ahead,
        'gap_behind': gap_behind,
        'compound': compound,
        'tire_age': tire_age
    }

def calculate_fuel_effect(current_lap: int, total_laps: int) -> float:
    """Calculate fuel load effect on tire degradation."""
    race_progress = current_lap / total_laps
    effect = (FUEL_EFFECTS['initial_multiplier'] - FUEL_EFFECTS['min_multiplier']) * \
            (1 - race_progress) ** (FUEL_EFFECTS['decay_rate'] * total_laps)
    return FUEL_EFFECTS['min_multiplier'] + effect

def calculate_evolution_bonus(track_type: str, track_evolution: float) -> int:
    """Calculate lap bonus based on track evolution."""
    if track_evolution >= 0.09:
        return 3  # High evolution bonus (e.g. Spa)
    elif track_evolution >= 0.08:
        return 2  # Medium evolution bonus (e.g. Monza)
    return 1     # Low evolution bonus (e.g. Silverstone)

def calculate_degradation_level(tire_age: int, max_life: int, fuel_effect: float) -> float:
    """Calculate tire degradation level (0-1 scale) with fuel effect."""
    base_deg = min(1.0, tire_age / max_life)
    return min(1.0, base_deg * fuel_effect)

def get_risk_assessment(deg_level: float) -> str:
    """Get risk assessment based on degradation level."""
    if deg_level >= 0.9:
        return "CRITICAL"
    elif deg_level >= 0.75:
        return "HIGH"
    elif deg_level >= 0.5:
        return "MEDIUM"
    return "LOW"

def get_strategy_recommendation(track_data, race_info, remaining_life, evo_bonus, risk, pit_window):
    """Get detailed strategy recommendation based on track characteristics."""
    remaining_race = track_data.total_laps - race_info['current_lap']
    recommendations = []
    
    # Early race strategy (first 10 laps)
    if race_info['current_lap'] < 10:
        if track_data.track_evolution >= 0.08:
            recommendations.extend([
                "MEDIUM tires recommended for longer first stint",
                f"High track evolution (+{evo_bonus} laps) will help extend tire life",
                f"Good overtaking opportunities (difficulty: {track_data.overtaking_diff:.2f})"
            ])
        else:
            recommendations.extend([
                "HARD tires safer for consistent first stint",
                "Focus on building gap in clean air",
                f"Track evolution bonus: +{evo_bonus} laps per stint"
            ])
        
        # Add fuel effect consideration
        fuel_effect = calculate_fuel_effect(race_info['current_lap'], track_data.total_laps)
        if fuel_effect > 1.2:
            recommendations.append(f"High fuel load effect (+{(fuel_effect-1)*100:.1f}% deg)")
    
    # Mid race strategy (>20 laps remaining)
    elif remaining_race > 20:
        if track_data.tire_deg_factor > 1.25:
            recommendations.extend([
                "HARD tire recommended for high degradation phase",
                f"High tire stress (deg factor: {track_data.tire_deg_factor:.2f})",
                "Consider offset strategy for track position"
            ])
        else:
            recommendations.extend([
                "MEDIUM tire offers good balance",
                "Can extend stint with careful management",
                f"Moderate tire stress (deg factor: {track_data.tire_deg_factor:.2f})"
            ])
    
    # Late race strategy (≤20 laps remaining)
    else:
        if remaining_race <= remaining_life:
            recommendations.extend([
                f"Current {race_info['compound']} tires can finish the race",
                f"Remaining life ({remaining_life} laps) sufficient for {remaining_race} laps",
                "Focus on consistent lap times and defending position"
            ])
        elif track_data.overtaking_diff < 0.3 and remaining_race <= 15:
            recommendations.extend([
                "SOFT tires viable for final stint",
                f"Good overtaking potential (difficulty: {track_data.overtaking_diff:.2f})",
                f"{remaining_race} laps to finish, aggressive strategy possible"
            ])
        else:
            recommendations.extend([
                "MEDIUM tires safer for track position",
                "Focus on consistent lap times",
                f"Track evolution helping tire life (+{evo_bonus} laps)"
            ])
    
    # Add ML-based pit window recommendation
    window_start, window_end = pit_window
    if window_start <= race_info['current_lap'] <= window_end:
        recommendations.append("Currently in optimal pit window")
        if race_info['gap_ahead'] < 3 and race_info['gap_ahead'] > 0:
            if track_data.overtaking_diff < 0.4:
                recommendations.append("Undercut viable with good overtaking potential")
            else:
                recommendations.append("Consider undercut to gain track position")
        elif race_info['gap_behind'] < 2:
            if track_data.overtaking_diff > 0.7:
                recommendations.append("Overcut preferred due to difficult overtaking")
            else:
                recommendations.append("Defend position, prepare for overcut if needed")
    elif race_info['current_lap'] < window_start:
        recommendations.append(f"Optimal pit window in {window_start - race_info['current_lap']} laps")
    
    # Add risk-based recommendations
    if risk in ["HIGH", "CRITICAL"]:
        if track_data.safety_car_prob >= 0.35:
            recommendations.append(f"High safety car chance ({track_data.safety_car_prob:.0%}), could help tire management")
        if track_data.overtaking_diff < 0.3:
            recommendations.append("Good overtaking opportunities after pit stop")
    
    return recommendations

def get_live_track_data(track_name):
    """Get real-time track data using FastF1."""
    try:
        # Get latest session for the track
        session = fastf1.get_session(2024, track_name, 'R')
        session.load()
        
        # Extract track characteristics from live data
        track_data = TRACK_CHARACTERISTICS[track_name].copy()
        
        # Update with live data where available
        track_status = session.track_status
        if track_status is not None:
            track_data['tire_deg_factor'] = min(1.3, max(1.2, track_status.mean() / 10))
            track_evolution = track_status.diff().mean()
            if track_evolution is not None:
                track_data['track_evolution'] = min(0.09, max(0.07, track_evolution))
        
        return TrackPredictor().predict_from_characteristics(track_data)
        
    except Exception as e:
        print(f"\nWarning: Could not fetch live data ({str(e)})")
        print("Falling back to cached track characteristics")
        return TrackPredictor().predict_from_characteristics(TRACK_CHARACTERISTICS[track_name])

def analyze_pit_window(track_data, race_info):
    """Analyze optimal pit window based on current situation."""
    # Get compound characteristics
    compound_data = TIRE_COMPOUNDS[race_info['compound']]
    base_life = compound_data['max_life']
    warning_age = compound_data['warning_age']
    deg_rate = compound_data['deg_rate']
    
    # Calculate evolution bonus and fuel effect
    evo_bonus = calculate_evolution_bonus(track_data.track_type, track_data.track_evolution)
    fuel_effect = calculate_fuel_effect(race_info['current_lap'], track_data.total_laps)
    
    # Apply track-specific degradation factor and fuel effect
    adjusted_life = int(base_life * (1 / track_data.tire_deg_factor)) + evo_bonus
    warning_lap = int(warning_age * (1 / track_data.tire_deg_factor)) + evo_bonus
    
    # Calculate current tire state
    remaining_life = adjusted_life - race_info['tire_age']
    laps_to_warning = warning_lap - race_info['tire_age']
    deg_level = calculate_degradation_level(race_info['tire_age'], adjusted_life, fuel_effect)
    risk = get_risk_assessment(deg_level)
    
    # Get ML-based pit window prediction
    race_data = {
        'current_position': race_info['current_position'],
        'gap_ahead': race_info['gap_ahead'],
        'gap_behind': race_info['gap_behind'],
        'track_data': track_data,
        'current_lap': race_info['current_lap'],
        'tire_age': race_info['tire_age'],
        'compound': race_info['compound']
    }
    pit_window = TrackPredictor().predict_pit_window(race_data)
    
    print("\nRace Situation Analysis:")
    print("=" * 50)
    print(f"Track: {race_info['track_name']} ({track_data.total_laps} laps)")
    print(f"Track Type: {track_data.track_type}")
    print(f"Position: P{race_info['current_position']}")
    if race_info['gap_ahead'] > 0:
        print(f"Gap Ahead: {race_info['gap_ahead']:.1f}s")
    if race_info['gap_behind'] > 0:
        print(f"Gap Behind: {race_info['gap_behind']:.1f}s")
    
    print("\nTrack Characteristics:")
    print("-" * 50)
    print(f"Tire Degradation Factor: {track_data.tire_deg_factor:.2f}")
    print(f"Track Evolution Rate: {track_data.track_evolution:.3f}")
    print(f"Evolution Bonus: +{evo_bonus} laps")
    print(f"Overtaking Difficulty: {track_data.overtaking_diff:.2f}")
    print(f"Safety Car Probability: {track_data.safety_car_prob:.0%}")
    
    print("\nTire Analysis:")
    print("-" * 50)
    print(f"Current Lap: {race_info['current_lap']}/{track_data.total_laps}")
    print(f"Compound: {race_info['compound']}")
    print(f"Current Tire Age: {race_info['tire_age']} laps")
    print(f"Base Tire Life: {base_life} laps")
    print(f"Adjusted Life (with track factor): {adjusted_life} laps")
    print(f"Remaining Life: {max(0, remaining_life)} laps")
    print(f"Fuel Effect: +{(fuel_effect-1)*100:.1f}% degradation")
    print(f"Degradation Level: {deg_level:.2%}")
    print(f"Risk Assessment: {risk}")
    
    # Determine pit window
    if risk == "CRITICAL":
        print("\nCRITICAL: Tires are at critical wear level!")
        print("Recommend pitting immediately")
    elif risk == "HIGH":
        print("\nWARNING: Tires are in high wear phase")
        print("Recommend pitting in next 1-2 laps")
    else:
        print(f"\nML-Predicted Pit Window: Lap {pit_window[0]}-{pit_window[1]}")
    
    # Get and display strategy recommendations
    recommendations = get_strategy_recommendation(track_data, race_info, remaining_life, evo_bonus, risk, pit_window)
    print("\nStrategy Recommendations:")
    print("-" * 50)
    for rec in recommendations:
        print(f"- {rec}")
    
    # Add final pit window recommendation
    remaining_race = track_data.total_laps - race_info['current_lap']
    if remaining_race <= remaining_life:
        print("\nPit Window Status: No pit stop needed")
        print(f"Current {race_info['compound']} tires sufficient to finish the race")
    elif risk == "CRITICAL":
        print("\nPit Window Status: CRITICAL - Box this lap")
    elif risk == "HIGH":
        print("\nPit Window Status: HIGH RISK - Box within 2 laps")
    else:
        print(f"\nPit Window Status: Next window Lap {pit_window[0]}-{pit_window[1]}")

def main():
    """Run pit stop analysis with ML-based optimization."""
    parser = argparse.ArgumentParser(description='F1 Pit Stop Optimizer')
    parser.add_argument('--live-data', action='store_true', help='Use live session data instead of cached characteristics')
    args = parser.parse_args()
    
    print("\nF1 Pit Stop Optimizer")
    print("=" * 50)
    print("Enhanced with fuel effect and track-specific tire modeling")
    
    while True:
        # Get user input
        race_info = get_user_input()
        track_name = race_info['track_name']
        
        # Get track data (live or cached)
        if args.live_data:
            track_data = get_live_track_data(track_name)
        else:
            track_data = TrackPredictor().predict_from_characteristics(TRACK_CHARACTERISTICS[track_name])
        
        # Analyze pit window
        analyze_pit_window(track_data, race_info)
        
        print("\nAnalyze another situation? (y/n): ", end='')
        if input().lower() != 'y':
            break
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
