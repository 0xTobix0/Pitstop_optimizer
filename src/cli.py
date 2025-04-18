"""
Command line interface for F1 pitstop optimization.
"""
import argparse
import logging
from data_processor import F1DataProcessor
from ml_model import PitstopPredictor
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='F1 Pitstop Optimization')
    
    # Required arguments
    parser.add_argument('--track', required=True, help='Track name (e.g. Monza, Suzuka)')
    
    # Mode selection
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--predict', action='store_true', help='Make prediction')
    
    # Training arguments
    parser.add_argument('--years', nargs='+', type=int, help='Years to train on')
    
    # Prediction arguments
    parser.add_argument('--lap', type=int, help='Current lap number')
    parser.add_argument('--tire-life', type=int, help='Current tire life in laps')
    parser.add_argument('--compound', help='Tire compound (SOFT, MEDIUM, HARD)')
    parser.add_argument('--lap-time', type=float, help='Last lap time in seconds')
    parser.add_argument('--track-temp', type=float, help='Track temperature in C')
    parser.add_argument('--air-temp', type=float, help='Air temperature in C')
    parser.add_argument('--humidity', type=float, help='Humidity percentage')
    parser.add_argument('--position', type=int, help='Current position')
    parser.add_argument('--lap-progress', type=float, default=0.0, help='Progress through current lap (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Initialize data processor and predictor
    data_processor = F1DataProcessor()
    predictor = PitstopPredictor(args.track)
    
    if args.train:
        if not args.years:
            parser.error("--years is required for training")
            
        # Calculate track characteristics
        logger.info(f"Calculating track characteristics for {args.track} using {args.years} data...")
        track_params = data_processor.calculate_track_characteristics(args.track, args.years)
        
        # Load and process historical data
        df = data_processor.load_historical_data(args.track, args.years)
        
        # Train model
        mae, rmse = predictor.train(df)
        logger.info(f"Training complete. MAE: {mae:.2f} laps, RMSE: {rmse:.2f} laps")
        
    elif args.predict:
        required = ['lap', 'tire_life', 'compound', 'lap_time', 'track_temp', 'air_temp', 'humidity', 'position']
        missing = [arg for arg in required if getattr(args, arg.replace('-', '_')) is None]
        if missing:
            parser.error(f"Missing required arguments for prediction: {', '.join(missing)}")
            
        # Create conditions DataFrame
        conditions = pd.DataFrame({
            'LapNumber': [args.lap],
            'TyreLife': [args.tire_life],
            'Compound': [args.compound],
            'LapTime': [args.lap_time],
            'TrackTemp': [args.track_temp],
            'AirTemp': [args.air_temp],
            'Humidity': [args.humidity],
            'Position': [args.position],
            'LapProgress': [args.lap_progress]
        })
        
        # Get prediction
        result = predictor.predict(conditions)
        
        # Print results
        print("\nPrediction Results:")
        print(f"INFO:__main__:Optimal pit lap: {result['optimal_pit_lap']}")
        print(f"INFO:__main__:Laps until pit: {result['laps_until_pit']:.1f}")
        print(f"INFO:__main__:Pit window: Lap {result['pit_window_start']} to {result['pit_window_end']}")
        print(f"INFO:__main__:Confidence: {result['confidence']*100:.1f}%")
        print(f"INFO:__main__:Uncertainty: ±{result['uncertainty']:.1f} laps\n")
        
    else:
        parser.error("Must specify either --train or --predict")
        
if __name__ == '__main__':
    main()
