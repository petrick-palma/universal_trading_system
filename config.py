import os
from dataclasses import dataclass
from typing import Dict, List
from dotenv import load_dotenv

# Carrega as variáveis do ficheiro .env para o ambiente do sistema
# Isto deve ser feito logo no início.
load_dotenv()

@dataclass
class TradingConfig:
    # API Configuration
    # os.getenv() vai ler automaticamente as variáveis carregadas pelo load_dotenv()
    API_KEY: str = os.getenv('BINANCE_API_KEY', '')
    API_SECRET: str = os.getenv('BINANCE_API_SECRET', '')
    TESTNET: bool = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    
    # Trading Parameters
    INITIAL_CAPITAL: float = 40.0
    MAX_CONCURRENT_TRADES: int = 4
    MIN_POSITION_SIZE: float = 8.0
    MAX_POSITION_SIZE: float = 20.0
    
    # Fees
    MAKER_FEE: float = 0.0010
    TAKER_FEE: float = 0.0010
    
    # Risk Management
    MAX_DRAWDOWN: float = 0.10  # 10%
    DAILY_LOSS_LIMIT: float = 0.05  # 5%
    MAX_LEVERAGE: int = 1
    
    # Strategy Weights
    STRATEGY_WEIGHTS: Dict[str, float] = None
    
    # Technical Analysis
    MIN_VOLUME_USDT: float = 200000
    MAX_SPREAD_PERCENT: float = 0.05
    VOLATILITY_THRESHOLD: float = 1.0
    
    # Bot Configuration
    SCAN_INTERVAL: int = 10  # seconds
    FULL_SCAN_INTERVAL: int = 300  # 30 minutes
    MONITOR_INTERVAL: int = 10  # seconds
    
    # Web Interface Configuration (Movido para WebConfig)
    
    # Order Configuration
    DEFAULT_TAKE_PROFIT: float = 0.003  # 0.3%
    DEFAULT_STOP_LOSS: float = 0.004    # 0.4%
    REBALANCE_THRESHOLD: float = -0.0015  # -0.15%

    def __post_init__(self):
        if self.STRATEGY_WEIGHTS is None:
            self.STRATEGY_WEIGHTS = {
                'rsi_momentum': 1.6,
                'macd_crossover': 1.5,
                'bollinger_breakout': 1.7,
                'volume_breakout': 1.9,
                'stochastic_reversal': 1.4,
                'ema_crossover': 1.8,
                'ichimoku_cloud': 1.3,
                'vwap_strategy': 1.5
            }

@dataclass
class DQLConfig:
    # Configurações para o seu modelo de Deep Q-Learning (IA)
    LEARNING_RATE: float = 0.001
    GAMMA: float = 0.95
    EPSILON: float = 1.0
    EPSILON_MIN: float = 0.01
    EPSILON_DECAY: float = 0.995
    MEMORY_SIZE: int = 2000
    BATCH_SIZE: int = 32
    TARGET_UPDATE_FREQ: int = 100

@dataclass
class WebConfig:
    # Configurações específicas da Interface Web
    HOST: str = os.getenv('WEB_HOST', '0.0.0.0')
    PORT: int = int(os.getenv('WEB_PORT', 5000))
    DEBUG: bool = os.getenv('WEB_DEBUG', 'false').lower() == 'true'
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'your-secret-key-here')