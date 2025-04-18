# F1 Pitstop Optimizer

An advanced machine learning model for optimizing Formula 1 pit stop strategies. Uses real-time race data to predict optimal pit windows based on tire degradation, track conditions, and race position.

## Program Structure

```
f1_pitstop_optimizer/
├── src/
│   ├── cli.py              # Command-line interface for predictions
│   ├── config.py           # Track parameters and model configuration
│   ├── data_processor.py   # Historical data loading and processing
│   ├── ml_model.py         # Core ML model implementation
│   ├── model_analysis.py   # Model evaluation and feature analysis
│   └── train.py            # Model training pipeline
├── models/                  # Trained model files
│   └── pitstop_model_*.txt # Track-specific models
├── data/                   # Historical race data (cached)
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

### Core Components

1. **Command Line Interface (`cli.py`)**:
   - Handles user input and command parsing
   - Provides real-time pit stop predictions
   - Displays results in a clear format
   - Supports multiple tracks and compounds

2. **Configuration (`config.py`)**:
   - Track-specific parameters:
     * Length, sectors, lap count
     * Tire degradation factors
     * Safety car probabilities
     * Overtaking difficulty
   - Model hyperparameters
   - Tire compound characteristics
   - Race configuration settings

3. **Data Processing (`data_processor.py`)**:
   - FastF1 data integration
   - Historical race data loading
   - Feature engineering:
     * Tire degradation metrics
     * Track evolution factors
     * Weather impact calculations
   - Real-time data processing

4. **ML Model (`ml_model.py`)**:
   - LightGBM model implementation
   - Prediction pipeline:
     * Feature preparation
     * Model inference
     * Confidence calculation
   - Track-specific adjustments
   - Window size optimization

5. **Model Analysis (`model_analysis.py`)**:
   - Feature importance analysis
   - Performance metrics calculation
   - Model validation tools
   - Prediction visualization

6. **Training Pipeline (`train.py`)**:
   - Data preparation
   - Model training workflow
   - Cross-validation
   - Model persistence

### Data Flow

1. **Input Processing**:
   ```
   CLI Input → Parameter Validation → Data Processor
   ```

2. **Feature Engineering**:
   ```
   Raw Data → Data Processor → Feature Matrix → ML Model
   ```

3. **Prediction Pipeline**:
   ```
   Features → ML Model → Raw Prediction → Strategy Adjustment → Final Output
   ```

4. **Model Training**:
   ```
   Historical Data → Feature Engineering → Model Training → Evaluation → Storage
   ```

### Key Design Patterns

1. **Factory Pattern**:
   - Track-specific model creation
   - Feature engineering pipelines
   - Configuration management

2. **Strategy Pattern**:
   - Compound-specific degradation models
   - Track evolution calculations
   - Window size optimization

3. **Observer Pattern**:
   - Real-time data updates
   - Model prediction triggers
   - Performance monitoring

4. **Singleton Pattern**:
   - Configuration management
   - Model caching
   - Data preprocessing pipelines

## Features

- **Dynamic Pit Window Prediction**: Calculates optimal pit windows considering:
  - Tire compound performance and degradation
  - Track-specific characteristics
  - Position and race situation
  - Weather conditions
  - Safety car probability

- **Track-Specific Modeling**: Supports multiple F1 circuits with unique characteristics:
  - Bahrain: High tire degradation, strong track evolution
  - Saudi Arabia: Low degradation, high safety car probability
  - Australia: Medium degradation, challenging overtaking
  - Japan: High-speed corners, weather sensitivity
  - China: Very high degradation, easy overtaking

- **Advanced Features**:
  - Compound-specific tire degradation models
  - Track evolution benefits
  - Position-based strategy adjustments
  - Dynamic confidence calculations
  - Sector-specific tire wear analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/f1-pitstop-optimizer.git
cd f1-pitstop-optimizer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
# Train model for a specific track
python src/train.py --track Japan

# Analyze model performance
python src/model_analysis.py --track Japan
```

### Making Predictions

```bash
# Get pit window prediction
python src/cli.py --track Japan \
    --predict \
    --lap 10 \
    --tire-life 10 \
    --compound SOFT \
    --lap-time 92.8 \
    --track-temp 28 \
    --air-temp 25 \
    --humidity 55 \
    --position 3
```

### Example Output

```
Prediction Results:
Optimal pit lap: 15
Laps until pit: 5
Pit window: Lap 11 to 19
Confidence: 70.0%
Uncertainty: ±0.4 laps
```

## Model Performance

- Mean Absolute Error: 0.26-0.27 laps
- Root Mean Square Error: 0.46-0.47 laps
- Consistent predictions across different tracks and conditions
- Validated against historical race data

## Key Components

1. **Data Processing (`data_processor.py`)**:
   - Historical race data loading
   - Feature engineering
   - Real-time data processing

2. **ML Model (`ml_model.py`)**:
   - LightGBM-based prediction
   - Track-specific parameter tuning
   - Confidence estimation

3. **Configuration (`config.py`)**:
   - Track parameters
   - Tire compound characteristics
   - Model hyperparameters

4. **Analysis Tools (`model_analysis.py`)**:
   - Feature importance analysis
   - Model performance metrics
   - Prediction validation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FastF1 for providing F1 timing data
- Formula 1 for the sport we love
- Contributors and maintainers of key dependencies
