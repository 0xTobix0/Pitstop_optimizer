"""
Train the pit stop prediction model.
"""
import os
import pandas as pd
from ml_model import PitstopPredictor
from data_processor import F1DataProcessor

def train_model(track_name: str):
    """Train model for a specific track."""
    print(f"\nTraining model for {track_name}...")
    
    # Initialize components
    data_processor = F1DataProcessor()
    predictor = PitstopPredictor(track_name)
    
    # Load and process data
    print("Loading historical data...")
    df = data_processor.load_historical_data([2022, 2023])
    
    # Train model
    print("Training model...")
    mae, rmse = predictor.train(df)
    
    print(f"\nTraining complete!")
    print(f"Mean Absolute Error: {mae:.2f} laps")
    print(f"Root Mean Square Error: {rmse:.2f} laps")

if __name__ == '__main__':
    # Train for multiple tracks
    tracks = ['China', 'Japan', 'Bahrain']
    for track in tracks:
        train_model(track)
