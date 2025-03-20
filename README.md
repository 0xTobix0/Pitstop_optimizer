# F1 Pit Stop Strategy Optimizer

An advanced Formula 1 pit stop strategy analyzer that helps optimize race strategies based on track characteristics, tire compounds, and fuel effects.

## Features

- **Track-Specific Analysis**: Detailed analysis for high-speed circuits (Monza, Spa, Silverstone)
  - Tire degradation factors
  - Track evolution rates
  - Safety car probabilities
  - Overtaking difficulty assessment

- **Tire Compound Analysis**:
  - Soft, Medium, and Hard compounds
  - Maximum life calculations
  - Grip level assessment
  - Degradation rate modeling
  - Fuel load impact on performance

- **Real-Time Strategy Insights**:
  - Optimal pit windows
  - Risk assessment
  - Strategy recommendations
  - Fuel effect calculations

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

The analyzer will guide you through:
1. Track selection
2. Current race situation input
3. Strategy analysis and recommendations

## Track Characteristics

### Monza (Temple of Speed)
- Highest tire degradation (1.3)
- Medium track evolution (0.08)
- 35% safety car probability
- Easiest overtaking (0.25)

### Spa-Francorchamps
- High tire degradation (1.25)
- Highest track evolution (0.09)
- 40% safety car probability
- Good overtaking opportunities (0.30)

### Silverstone
- Moderate tire degradation (1.2)
- Balanced track evolution (0.07)
- 30% safety car probability
- Technical overtaking (0.35)

## Dependencies
- FastF1 >= 3.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
