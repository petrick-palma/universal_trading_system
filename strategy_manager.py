import importlib
import inspect
from typing import Dict, List, Optional
import pandas as pd

from config import TradingConfig
from strategies.base_strategy import BaseStrategy

class StrategyManager:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.strategies: Dict[str, BaseStrategy] = {}
        self.load_strategies()
        
    def load_strategies(self):
        """Load all available strategies"""
        strategy_classes = {
            'rsi_momentum': 'RSIMomentumStrategy',
            'macd_crossover': 'MACDCrossoverStrategy', 
            'bollinger_breakout': 'BollingerBreakoutStrategy',
            'volume_breakout': 'VolumeBreakoutStrategy',
            'stochastic_reversal': 'StochasticReversalStrategy',
            'ema_crossover': 'EMACrossoverStrategy',
            'ichimoku_cloud': 'IchimokuCloudStrategy',
            'vwap_strategy': 'VWAPStrategy'
        }
        
        for strategy_name, class_name in strategy_classes.items():
            try:
                module = importlib.import_module(f'strategies.{strategy_name}')
                strategy_class = getattr(module, class_name)
                
                if inspect.isclass(strategy_class) and issubclass(strategy_class, BaseStrategy):
                    strategy_instance = strategy_class()
                    self.strategies[strategy_name] = strategy_instance
                    
            except Exception as e:
                print(f"Failed to load strategy {strategy_name}: {e}")
                
        print(f"Loaded {len(self.strategies)} strategies")
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """Analyze symbol with all strategies"""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                # Calculate indicators
                df_with_indicators = strategy.calculate_indicators(df)
                
                # Generate signal
                signal = strategy.generate_signal(df_with_indicators)
                
                if signal['signal'] != 'HOLD':
                    # Calculate quality score
                    quality_score = strategy.calculate_quality_score(df_with_indicators, signal)
                    
                    # Calculate TP/SL
                    current_price = df['close'].iloc[-1]
                    tp_sl = strategy.calculate_tp_sl(current_price, signal['signal'])
                    
                    signals.append({
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'signal': signal['signal'],
                        'strength': signal['strength'],
                        'quality_score': quality_score,
                        'current_price': current_price,
                        'take_profit': tp_sl['take_profit'],
                        'stop_loss': tp_sl['stop_loss'],
                        'weight': strategy.weight,
                        'timeframe': strategy.timeframe
                    })
                    
            except Exception as e:
                print(f"Error analyzing {symbol} with {strategy_name}: {e}")
                continue
                
        return signals
    
    def get_best_strategy(self, signals: List[Dict]) -> Optional[Dict]:
        """Get the best strategy from multiple signals"""
        if not signals:
            return None
            
        # Calculate weighted score
        for signal in signals:
            weighted_score = signal['strength'] * signal['quality_score'] * signal['weight']
            signal['weighted_score'] = weighted_score
            
        # Return signal with highest weighted score
        best_signal = max(signals, key=lambda x: x['weighted_score'])
        
        # Only return if score meets minimum threshold
        if best_signal['weighted_score'] > 50:
            return best_signal
            
        return None