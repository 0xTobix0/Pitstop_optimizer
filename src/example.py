"""
Example usage of the F1 pit stop analyzer.
Demonstrates analyzing pit stop strategies for similar dry tracks.
"""

from pitstop_analyzer import PitStopAnalyzer, SIMILAR_TRACKS, TIRE_COMPOUNDS, FUEL_EFFECTS
from typing import Dict, Any, List

def validate_tire_age(compound: str, age: int) -> bool:
    """Validate if tire age is within compound's maximum life."""
    max_life = TIRE_COMPOUNDS[compound].max_life
    return 0 <= age <= max_life

def get_strategy_insights(track_config, current_lap, compound, tire_age, recommended_compound):
    """Get strategic insights based on track and tire characteristics."""
    insights = []
    
    # Track-specific insights
    if track_config.overtaking_difficulty < 0.4:
        insights.append("Multiple overtaking opportunities allow for aggressive strategy")
    elif track_config.overtaking_difficulty > 0.7:
        insights.append("Limited overtaking suggests prioritizing track position")
    
    # Safety car probability insights
    if track_config.safety_car_prob > 0.4:
        insights.append("High safety car probability - consider flexible strategy")
    
    # Tire compound specific insights
    remaining_laps = track_config.laps - current_lap
    current_max_life = TIRE_COMPOUNDS[compound].max_life
    recommended_max_life = TIRE_COMPOUNDS[recommended_compound].max_life
    
    # Current tire situation
    life_remaining = current_max_life - tire_age
    if life_remaining < 5:
        insights.append(f"Critical: {compound} tires near end of life ({life_remaining} laps remaining)")
    elif life_remaining < 10:
        insights.append(f"Warning: {compound} tires in final phase ({life_remaining} laps remaining)")
    
    # Recommended compound insights
    if recommended_compound == 'SOFT':
        if remaining_laps <= 15:
            insights.append("SOFT compound optimal for short final stint")
        else:
            insights.append("SOFT compound risky for long stint - consider alternatives")
    elif recommended_compound == 'HARD':
        if remaining_laps > 30:
            insights.append("HARD compound provides flexibility for long final stint")
        else:
            insights.append("HARD compound may sacrifice pace unnecessarily")
    
    # Track evolution impact
    if current_lap < 10:
        insights.append("Early race: Track evolution significant - harder compounds may struggle")
    elif current_lap > track_config.laps - 15:
        insights.append("Late race: Track well rubbered in - softer compounds more effective")
    
    return insights

def analyze_track_strategy() -> bool:
    """Analyze pit stop strategy for a specific track."""
    try:
        # Display track selection
        print("\nAvailable Tracks:")
        tracks = {
            1: "Monza",
            2: "Spa", 
            3: "Silverstone"
        }
        for num, track in tracks.items():
            print(f"{num}. {track}")
            
        track_num = int(input("\nSelect track (enter number): "))
        if track_num not in tracks:
            raise ValueError("Invalid track selection")
            
        track_name = tracks[track_num]
        analyzer = PitStopAnalyzer(track_name)
        
        # Display race information
        print(f"\nRace Information:")
        print(f"Total Laps: {analyzer.track_data.laps}")
        print(f"Track Length: {analyzer.track_data.length:.3f} km")
        print(f"Pit Loss Time: {analyzer.track_data.pit_loss_time}s")
        print(f"Starting Fuel: {analyzer.total_fuel:.1f} kg")
        
        # Get current lap
        current_lap = int(input(f"\nCurrent lap (1-{analyzer.track_data.laps}): "))
        if current_lap < 1 or current_lap > analyzer.track_data.laps:
            raise ValueError("Invalid lap number")
            
        # Display compound selection
        print("\nAvailable Compounds:")
        compounds = {
            1: "SOFT",
            2: "MEDIUM",
            3: "HARD"
        }
        for num, compound in compounds.items():
            max_life = TIRE_COMPOUNDS[compound].max_life
            fuel_effect = FUEL_EFFECTS[compound]
            print(f"{num}. {compound:<6} (Max life: ~{max_life} laps)")
            print(f"   Fuel sensitivity: {fuel_effect.sensitivity:.3f}, Weight impact: {fuel_effect.weight_factor:.1f}")
            
        compound_num = int(input("\nSelect current tire compound (enter number): "))
        if compound_num not in compounds:
            raise ValueError("Invalid compound selection")
            
        compound = compounds[compound_num]
        analyzer.current_compound = compound
        
        # Get tire age
        max_age = TIRE_COMPOUNDS[compound].max_life
        tire_age = int(input(f"\nCurrent tire age (0-{max_age} laps): "))
        if tire_age < 0 or tire_age > max_age:
            raise ValueError("Invalid tire age")
            
        print(f"\nAnalyzing {track_name} GP Strategy")
        print("=" * 50)
        
        # Load historical data
        analyzer.load_race_data(2022)
        analyzer.load_race_data(2023)
        
        # Get strategy analysis
        result = analyzer.summarize_situation(current_lap, compound, tire_age)
        
        # Display track characteristics
        track_info = result['track_characteristics']
        print(f"\nTrack Characteristics:")
        print(f"Length: {track_info['length']:.3f} km")
        print(f"Laps: {track_info['laps']}")
        print(f"Tire Degradation Factor: {track_info['tire_deg_factor']:.2f}")
        print(f"Track Evolution: +{track_info['track_evolution']*100:.1f}% grip per lap")
        print("Overtaking: " + ("Easy" if track_info['overtaking_difficulty'] < 0.4 
              else "Hard" if track_info['overtaking_difficulty'] > 0.7 else "Moderate"))
        print(f"Safety Car Probability: {int(track_info['safety_car_prob']*100)}%")
        
        # Display current situation
        current = result['current_situation']
        print(f"\nCurrent Situation:")
        print(f"Lap: {current['lap']}/{track_info['laps']}")
        print(f"Tire: {current['compound']} ({current['tire_age']} laps old)")
        print(f"Degradation: {current['degradation']*100:.1f}%")
        print(f"Fuel Effect: {current['fuel_effect']:.2f}x")
        print(f"Risk Level: {current['risk_level'].upper()}")
        
        # Calculate remaining fuel
        fuel_per_lap = analyzer.total_fuel / analyzer.track_data.laps
        remaining_fuel = max(0, analyzer.total_fuel - (current_lap * fuel_per_lap))
        print(f"Remaining Fuel: {remaining_fuel:.1f} kg")
        
        # Display warnings
        if result['warnings']:
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"- {warning}")
                
        # Display pit window
        window = result['pit_window']
        print(f"\nPit Window:")
        print(f"Start: Lap {window['start_lap']}")
        print(f"Optimal: Lap {window['optimal_lap']}")
        print(f"End: Lap {window['end_lap']}")
        
        # Display expected stint lengths
        print("\nExpected Stint Lengths (adjusted for fuel and track):")
        for compound, length in result['stint_lengths'].items():
            fuel_effect = FUEL_EFFECTS[compound]
            print(f"{compound}: ~{length} laps")
            print(f"  Base sensitivity: {fuel_effect.base_effect:.1f}x")
            print(f"  Weight impact: {fuel_effect.weight_factor:.1f}x")
            
        # Display strategy options
        if result['strategy_options']:
            print("\nStrategy Options:")
            for option in result['strategy_options']:
                print(f"- {option}")
        
        return True
        
    except ValueError as e:
        print(f"\nError: {str(e)}")
        return False
    except Exception as e:
        print(f"\nError analyzing strategy: {str(e)}")
        return False

def main():
    """Main function to run the pit stop analyzer."""
    print("F1 Pit Stop Strategy Analyzer")
    print("=" * 30)
    
    while True:
        print("\nEnter Race Situation Details")
        print("=" * 30)
        
        if not analyze_track_strategy():
            print("Please try again with different parameters")
            
        choice = input("\nAnalyze another situation? (y/n): ")
        if choice.lower() != 'y':
            break

if __name__ == "__main__":
    main()
