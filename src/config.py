"""
Configuration settings for F1 pitstop optimization.
"""

# Model training parameters
MODEL_CONFIG = {
    'objective': 'regression',
    'metric': ['mae', 'rmse'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'num_boost_round': 100,
    'early_stopping_rounds': 10,
    'verbose': -1
}

# Race configuration
RACE_CONFIG = {
    'initial_fuel': 110,  # kg
    'fuel_penalty_per_kg': 0.03,  # seconds per kg
    'base_tire_deg_rate': 0.1,  # base degradation per lap
    'prediction_window_size': 3,  # base window size in laps
    'fuel_per_km': 0.38,  # kg per km, average F1 fuel consumption
    'base_tire_deg_rate': 0.05,  # Base tire degradation per lap
    'base_fuel_penalty': 0.02,   # Base fuel weight penalty per lap
}

# Track-specific parameters
TRACK_PARAMS = {
    'Bahrain': {
        'length': 5.412,  # km
        'tire_deg_factor': 1.3,  # High deg due to abrasive surface
        'overtaking_difficulty': 0.5,  # Medium difficulty
        'safety_car_probability': 0.45,  # 45% based on historical data
        'virtual_safety_car_probability': 0.60,  # 60% based on historical data
        'track_evolution': 0.09,  # High evolution
        'total_laps': 57,
        'max_stint': {
            'SOFT': {
                'optimal': 15,
                'maximum': 18
            },
            'MEDIUM': {
                'optimal': 25,
                'maximum': 28
            },
            'HARD': {
                'optimal': 35,
                'maximum': 40
            }
        }
    },
    'Saudi Arabia': {
        'length': 6.174,  # km
        'tire_deg_factor': 1.1,  # Low deg due to smooth surface
        'overtaking_difficulty': 0.7,  # High difficulty
        'safety_car_probability': 0.65,  # 65% based on historical data
        'virtual_safety_car_probability': 0.75,  # 75% based on historical data
        'track_evolution': 0.08,  # Medium evolution
        'total_laps': 50,
        'max_stint': {
            'SOFT': {
                'optimal': 18,
                'maximum': 20
            },
            'MEDIUM': {
                'optimal': 28,
                'maximum': 30
            },
            'HARD': {
                'optimal': 38,
                'maximum': 42
            }
        }
    },
    'Australia': {
        'length': 5.278,  # km
        'tire_deg_factor': 1.2,  # Medium-high deg due to fast corners
        'overtaking_difficulty': 0.7,  # High difficulty based on track layout
        'safety_car_probability': 0.50,  # 50% based on historical data
        'virtual_safety_car_probability': 0.67,  # 67% based on historical data
        'track_evolution': 0.08,  # Medium evolution
        'total_laps': 58,
        'max_stint': {
            'SOFT': {
                'optimal': 15,
                'maximum': 18
            },
            'MEDIUM': {
                'optimal': 22,
                'maximum': 25
            },
            'HARD': {
                'optimal': 30,
                'maximum': 35
            }
        }
    },
    'Japan': {
        'length': 5.807,  # km
        'tire_deg_factor': 1.25,  # High deg due to high-speed corners
        'overtaking_difficulty': 0.6,  # Medium difficulty
        'safety_car_probability': 0.40,  # 40% based on historical data
        'virtual_safety_car_probability': 0.55,  # 55% based on historical data
        'track_evolution': 0.07,  # Low evolution due to frequent rain
        'total_laps': 53,
        'max_stint': {
            'SOFT': {
                'optimal': 14,
                'maximum': 16
            },
            'MEDIUM': {
                'optimal': 23,
                'maximum': 26
            },
            'HARD': {
                'optimal': 32,
                'maximum': 36
            }
        }
    },
    'China': {
        'length': 5.451,  # km
        'tire_deg_factor': 1.35,  # Very high deg due to long corners
        'overtaking_difficulty': 0.4,  # Easy with long straights
        'safety_car_probability': 0.35,  # 35% based on historical data
        'virtual_safety_car_probability': 0.50,  # 50% based on historical data
        'track_evolution': 0.09,  # High evolution
        'total_laps': 56,
        'max_stint': {
            'SOFT': {
                'optimal': 13,
                'maximum': 15
            },
            'MEDIUM': {
                'optimal': 21,
                'maximum': 24
            },
            'HARD': {
                'optimal': 30,
                'maximum': 34
            }
        }
    }
}

# Tire compound characteristics
TIRE_COMPOUNDS = {
    'SOFT': {
        'max_life': 20,
        'grip_level': 1.0,
        'deg_rate': 1.3,
        'optimal_temp': 90
    },
    'MEDIUM': {
        'max_life': 30,
        'grip_level': 0.85,
        'deg_rate': 1.0,
        'optimal_temp': 85
    },
    'HARD': {
        'max_life': 40,
        'grip_level': 0.7,
        'deg_rate': 0.8,
        'optimal_temp': 80
    }
}

# Track type characteristics
TRACK_TYPES = {
    'Street': {
        'evolution_range': (0.14, 0.18),
        'pit_stops': (1, 2),
        'strategy': 'Prioritize track position'
    },
    'High-speed': {
        'evolution_range': (0.07, 0.10),
        'pit_stops': (2, 3),
        'strategy': 'Allow aggressive strategies'
    },
    'Technical': {
        'evolution_range': (0.06, 0.09),
        'pit_stops': (2, 2),
        'strategy': 'Balance position and pace'
    }
}
