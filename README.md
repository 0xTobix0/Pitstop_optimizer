# F1 Pit Stop Strategy Optimizer

An advanced Formula 1 pit stop strategy analyzer that uses machine learning and real-time data to optimize race strategies based on track characteristics, tire compounds, and fuel effects.

## Features

- **Track-Specific Analysis**: Detailed analysis for high-speed circuits (Monza, Spa, Silverstone)
  - Tire degradation factors
  - Track evolution rates with dynamic bonuses
  - Safety car probabilities
  - Overtaking difficulty assessment
  - ML-based pit window predictions

- **Enhanced Tire Modeling**:
  - Soft, Medium, and Hard compounds
  - Dynamic maximum life calculations
  - Grip level assessment
  - Compound-specific degradation rates
  - Fuel load impact visualization
  - Track-specific tire wear modeling

- **Real-Time Strategy Insights**:
  - ML-optimized pit windows
  - Multi-factor risk assessment
  - Adaptive strategy recommendations
  - Fuel effect calculations
  - Position-based tactical advice
  - Track evolution bonuses

- **Advanced Visualizations**:
  - Fuel effect curves for each compound
  - High-resolution degradation graphs
  - Track-specific strategy visualizations
  - Clear pit window indicators

## Installation

1. Clone the repository:
```bash
git clone https://github.com/0xTobix0/F1-pitstop-analysis.git
cd F1-pitstop-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the example script to analyze pit stop strategies:
```bash
python src/example.py
```

For ML model testing and visualization:
```bash
python src/test_ml_predictions.py [track_name]
```

### Real-Time Data Mode

To use live session data instead of cached characteristics:
```bash
python src/example.py --live-data
```

This will:
1. Connect to F1's timing service via FastF1
2. Fetch real-time track conditions
3. Adjust strategy based on current session data

The analyzer will guide you through:
1. Track selection (Monza/Spa/Silverstone)
2. Current race situation input
3. Comprehensive strategy analysis
4. ML-based recommendations

### Strategy Analysis

The system provides:
1. **Current Situation**:
   - Lap number and remaining laps
   - Tire compound and age
   - Degradation level (0-1 scale)
   - Risk assessment (low/medium/high/critical)

2. **Track Insights**:
   - Degradation factors
   - Evolution rates with bonus laps
   - Safety car probabilities
   - Overtaking difficulty

3. **Recommendations**:
   - Early Race (<10 laps): Compound based on evolution
   - Mid Race: Based on degradation and SC risk
   - Late Race (≤20 laps): Based on overtaking ease

## Track Characteristics

### Monza (Temple of Speed)
- Highest tire degradation (1.3)
- Medium track evolution (0.08, +2 lap bonus)
- 35% safety car probability
- Easiest overtaking (0.25)
- Best for aggressive strategies

### Spa-Francorchamps
- High tire degradation (1.25)
- Highest track evolution (0.09, +3 lap bonus)
- 40% safety car probability
- Good overtaking opportunities (0.30)
- Balanced strategy options

### Silverstone
- Moderate tire degradation (1.2)
- Balanced track evolution (0.07, +1 lap bonus)
- 30% safety car probability
- Technical overtaking (0.35)
- Track position priority

## Machine Learning Model

The optimizer uses LightGBM for pit window predictions, considering:
- Track-specific characteristics
- Current race situation
- Tire compound performance
- Fuel load effects
- Historical race data

## Dependencies
- LightGBM >= 3.3.5 (ML predictions)
- FastF1 >= 3.0.0 (F1 timing data)
- NumPy >= 1.24.3 (Core calculations)
- Pandas >= 2.0.2 (Data handling)
- Matplotlib >= 3.7.1 (Visualizations)
- Scikit-learn >= 1.2.2 (ML preprocessing)

## Data Sources

The optimizer uses two data sources:
1. **Cached Track Data**: Pre-computed characteristics for Monza, Spa, and Silverstone
2. **Live F1 Data**: Real-time session data via FastF1 when cache isn't available

Track characteristics are based on 2022-2023 race data and include:
- Tire degradation factors (1.2-1.3 range)
- Track evolution rates (0.07-0.09)
- Safety car probabilities (30-40%)
- Overtaking difficulty metrics (0.25-0.35)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
