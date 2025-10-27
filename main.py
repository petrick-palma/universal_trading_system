import time
import signal
import sys
import logging
from threading import Thread
from typing import Dict, List
# A linha "import schedule" foi removida daqui
import numpy as np

from config import TradingConfig, WebConfig
from binance_client import BinanceClient
from strategy_manager import StrategyManager
from bots.scanner_bot import ScannerBot
from bots.executor_bot import ExecutorBot
from bots.monitor_bot import MonitorBot
from bots.training_bot import TrainingBot
from bots.learning_bot import LearningBot
from web_interface.app import create_app
from utils.logger import setup_logger

class UniversalTradingSystem:
    def __init__(self):
        self.config = TradingConfig()
        self.web_config = WebConfig()
        self.logger = setup_logger('main')
        
        # Initialize components
        self.binance_client = BinanceClient(self.config)
        self.strategy_manager = StrategyManager(self.config)
        
        # Initialize bots
        self.scanner_bot = ScannerBot(self.config, self.binance_client, self.strategy_manager)
        self.executor_bot = ExecutorBot(self.config, self.binance_client, self.strategy_manager)
        self.monitor_bot = MonitorBot(self.config, self.binance_client)
        self.training_bot = TrainingBot(self.config, self.binance_client, self.strategy_manager)
        self.learning_bot = LearningBot(self.config, self.binance_client, self.strategy_manager)
        
        # Connect monitor bot to trading system for learning
        self.monitor_bot.trading_system = self
        
        # Web interface
        self.web_app = create_app(self)
        
        # State
        self.running = False
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'current_balance': 0.0,
            'active_positions': 0,
            'today_trades': 0,
            'win_rate': 0.0,
            'available_opportunities': 0,
            'system_status': 'Initializing',
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'learning_cycles': 0,
            'training_sessions': 0,
            'knowledge_base_loaded': False
        }
        
        # Real-time data for web interface
        self.realtime_data = {
            'opportunities': [],
            'positions': {},
            'performance_history': [],
            'last_update': None,
            'learning_insights': [],
            'training_progress': {},
            'market_regimes': {}
        }
        
        self.logger.info("Universal Trading System initialized with enhanced learning capabilities")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Received shutdown signal...")
        self.running = False
        sys.exit(0)
    
    def update_stats(self):
        """Update system statistics with enhanced metrics"""
        try:
            self.stats['current_balance'] = self.binance_client.get_account_balance()
            self.stats['active_positions'] = len(self.monitor_bot.active_positions)
            self.stats['available_opportunities'] = len(self.scanner_bot.highlight_opportunities)
            
            # Calculate win rate
            total_trades = self.stats['total_trades']
            if total_trades > 0:
                self.stats['win_rate'] = (self.stats['winning_trades'] / total_trades) * 100
            
            # Update learning and training stats
            if hasattr(self.learning_bot, 'performance_data'):
                self.stats['learning_cycles'] = len(self.learning_bot.performance_data) // 30
            if hasattr(self.training_bot, 'training_history'):
                self.stats['training_sessions'] = len(self.training_bot.training_history)
            
            # Update knowledge base status
            if hasattr(self.learning_bot, 'market_regime_knowledge'):
                self.stats['knowledge_base_loaded'] = len(self.learning_bot.market_regime_knowledge) > 0
            
            # Update real-time data for web interface
            self.realtime_data['opportunities'] = self.scanner_bot.highlight_opportunities
            self.realtime_data['positions'] = self.monitor_bot.active_positions
            self.realtime_data['last_update'] = time.time()
            
            # Add performance data point
            if len(self.realtime_data['performance_history']) < 100:  # Increased buffer
                self.realtime_data['performance_history'].append(self.stats['current_balance'])
            else:
                self.realtime_data['performance_history'].pop(0)
                self.realtime_data['performance_history'].append(self.stats['current_balance'])
            
            # Update learning insights and market regimes
            self._update_learning_insights()
                
        except Exception as e:
            self.logger.error(f"Error updating stats: {e}")
    
    def _update_learning_insights(self):
        """Update learning insights for web interface"""
        try:
            # Get recent learning insights
            if hasattr(self.learning_bot, 'generate_performance_insights'):
                insights = self.learning_bot.generate_performance_insights()
                self.realtime_data['learning_insights'] = insights[:5]  # Top 5 insights
            
            # Update training progress
            if hasattr(self.training_bot, 'training_data'):
                self.realtime_data['training_progress'] = {
                    'total_samples': len(self.training_bot.training_data),
                    'recent_quality': np.mean([d.get('quality_score', 0) for d in self.training_bot.training_data[-50:]]) if self.training_bot.training_data else 0,
                    'model_versions': len(self.training_bot.model_versions) if hasattr(self.training_bot, 'model_versions') else 0,
                    'last_training': self.training_bot.last_training.isoformat() if hasattr(self.training_bot, 'last_training') and self.training_bot.last_training else None
                }
            
            # Update market regimes knowledge
            if hasattr(self.learning_bot, 'market_regime_knowledge'):
                self.realtime_data['market_regimes'] = {
                    'total_regimes': len(self.learning_bot.market_regime_knowledge),
                    'recent_optimizations': self.learning_bot.optimization_history[-3:] if hasattr(self.learning_bot, 'optimization_history') else []
                }
                
        except Exception as e:
            self.logger.debug(f"Error updating learning insights: {e}")
    
    def record_trade_result(self, trade_data: Dict):
        """Registrar resultado de trade para aprendizado cumulativo"""
        try:
            # Record in learning bot
            self.learning_bot.record_trade_result(trade_data)
            
            # Update stats based on trade result
            self.stats['total_trades'] += 1
            pnl_absolute = trade_data.get('pnl_absolute', 0)
            
            if pnl_absolute > 0:
                self.stats['winning_trades'] += 1
                self.stats['consecutive_wins'] = self.stats.get('consecutive_wins', 0) + 1
                self.stats['consecutive_losses'] = 0
            else:
                self.stats['losing_trades'] += 1
                self.stats['consecutive_losses'] = self.stats.get('consecutive_losses', 0) + 1
                self.stats['consecutive_wins'] = 0
            
            self.stats['total_profit'] += pnl_absolute
            
            # Trigger learning cycle after significant number of trades
            if self.stats['total_trades'] % 25 == 0:  # Every 25 trades
                self.logger.info("🔄 Triggering learning cycle based on trade count")
                self.learning_bot.run_learning_cycle()
            
            # Save knowledge base after important trades
            if abs(pnl_absolute) > 10:  # Significant trade
                if hasattr(self.learning_bot, 'save_knowledge_base'):
                    self.learning_bot.save_knowledge_base()
                    self.logger.info("💾 Knowledge base saved after significant trade")
            
        except Exception as e:
            self.logger.error(f"Error recording trade result: {e}")
    
    def recover_open_positions(self):
        """Recover open positions on startup with enhanced tracking"""
        try:
            self.logger.info("🔍 Recovering open positions...")
            open_positions = self.binance_client.get_open_positions()
            
            recovered_count = 0
            for position in open_positions:
                symbol = position['symbol']
                quantity = position['quantity']
                entry_price = position['estimated_entry_price']
                
                # Check if position is already being monitored
                if not self.monitor_bot.risk_manager.has_position(symbol):
                    # Register recovered position with risk manager
                    position_data = {
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'strategy': 'recovered',
                        'signal_type': 'LONG',
                        'take_profit': entry_price * 1.0005,  # 0.05% TP
                        'stop_loss': entry_price * 0.996,    # 0.4% SL
                        'entry_fee': 0.0,
                        'entry_notional': entry_price * quantity,
                        'quality_score': 80,
                        'recovered': True,
                        'entry_time': time.time() - 3600,  # Assume entered 1 hour ago
                        'current_price': entry_price,
                        'unrealized_pnl': 0.0
                    }
                    
                    position_id = f"recovered_{symbol}_{int(time.time())}"
                    self.monitor_bot.risk_manager.register_position(position_id, position_data)
                    
                    self.logger.info(f"✅ Recovered position: {symbol} - {quantity:.6f} @ ${entry_price:.6f}")
                    recovered_count += 1
                else:
                    self.logger.info(f"ℹ️ Position already monitored: {symbol}")
            
            self.logger.info(f"🎯 {recovered_count} positions recovered")
            
            # Update learning bot about recovered positions
            if recovered_count > 0 and hasattr(self.learning_bot, 'logger'):
                self.learning_bot.logger.info(f"📝 {recovered_count} positions recovered on startup")
            
        except Exception as e:
            self.logger.error(f"Error recovering open positions: {e}")
    
    def run_scanner_bot(self):
        """Run scanner bot in separate thread"""
        while self.running:
            try:
                opportunities = self.scanner_bot.run_scan_cycle()
                self.executor_bot.update_opportunities(opportunities)
                self.stats['system_status'] = 'Scanning'
                time.sleep(self.config.SCAN_INTERVAL)
            except Exception as e:
                self.logger.error(f"Scanner bot error: {e}")
                self.stats['system_status'] = 'Scanner Error'
                time.sleep(30)
    
    def run_executor_bot(self):
        """Run executor bot in separate thread"""
        while self.running:
            try:
                self.executor_bot.run_execution_cycle()
                self.stats['system_status'] = 'Monitoring'
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Executor bot error: {e}")
                self.stats['system_status'] = 'Executor Error'
                time.sleep(30)
    
    def run_monitor_bot(self):
        """Run monitor bot in separate thread"""
        while self.running:
            try:
                self.monitor_bot.run_monitoring_cycle()
                self.update_stats()
                self.stats['system_status'] = 'Monitoring'
                time.sleep(self.config.MONITOR_INTERVAL)
            except Exception as e:
                self.logger.error(f"Monitor bot error: {e}")
                self.stats['system_status'] = 'Monitor Error'
                time.sleep(30)
    
    def run_training_bot(self):
        """Run training bot in separate thread with enhanced scheduling"""
        while self.running:
            try:
                self.training_bot.run_training_cycle()
                
                # Generate training insights periodically
                if (hasattr(self.training_bot, 'training_data') and 
                    len(self.training_bot.training_data) > 0 and 
                    len(self.training_bot.training_data) % 100 == 0):
                    if hasattr(self.training_bot, 'generate_training_insights'):
                        self.training_bot.generate_training_insights()
                
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Training bot error: {e}")
                time.sleep(60)
    
    def run_learning_bot(self):
        """Run learning bot in separate thread with cumulative learning"""
        while self.running:
            try:
                self.learning_bot.run_learning_cycle()
                
                # Save knowledge base periodically
                if (hasattr(self.learning_bot, 'performance_data') and 
                    len(self.learning_bot.performance_data) > 0 and
                    len(self.learning_bot.performance_data) % 20 == 0):
                    if hasattr(self.learning_bot, 'save_knowledge_base'):
                        self.learning_bot.save_knowledge_base()
                        self.logger.debug("💾 Periodic knowledge base save completed")
                
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Learning bot error: {e}")
                time.sleep(300)
    
    def run_performance_report(self):
        """Run periodic performance reporting with learning insights"""
        while self.running:
            try:
                self.generate_performance_report()
                time.sleep(300)  # Report every 5 minutes
            except Exception as e:
                self.logger.error(f"Performance report error: {e}")
                time.sleep(300)
    
    def generate_performance_report(self):
        """Generate comprehensive performance report with learning metrics"""
        try:
            current_balance = self.binance_client.get_account_balance()
            active_positions = len(self.monitor_bot.active_positions)
            total_trades = self.stats['total_trades']
            win_rate = self.stats['win_rate']
            
            # Learning metrics
            learning_data_count = len(self.learning_bot.performance_data) if hasattr(self.learning_bot, 'performance_data') else 0
            training_samples = len(self.training_bot.training_data) if hasattr(self.training_bot, 'training_data') else 0
            market_regimes_learned = len(self.learning_bot.market_regime_knowledge) if hasattr(self.learning_bot, 'market_regime_knowledge') else 0
            strategy_patterns = len(self.learning_bot.strategy_patterns) if hasattr(self.learning_bot, 'strategy_patterns') else 0
            
            self.logger.info("📊 ENHANCED PERFORMANCE REPORT")
            self.logger.info("=" * 60)
            self.logger.info(f"💰 Current Balance: ${current_balance:.2f}")
            self.logger.info(f"📈 Active Positions: {active_positions}")
            self.logger.info(f"🎯 Total Trades: {total_trades}")
            self.logger.info(f"📊 Win Rate: {win_rate:.1f}%")
            self.logger.info(f"💰 Total Profit: ${self.stats['total_profit']:.2f}")
            self.logger.info(f"🔍 Available Opportunities: {self.stats['available_opportunities']}")
            self.logger.info(f"🔄 System Status: {self.stats['system_status']}")
            self.logger.info(f"🧠 Learning Data: {learning_data_count} trades")
            self.logger.info(f"🎓 Training Samples: {training_samples}")
            self.logger.info(f"🌡️ Market Regimes Learned: {market_regimes_learned}")
            self.logger.info(f"📊 Strategy Patterns: {strategy_patterns}")
            self.logger.info(f"📈 Consecutive Wins: {self.stats.get('consecutive_wins', 0)}")
            self.logger.info(f"📉 Consecutive Losses: {self.stats.get('consecutive_losses', 0)}")
            self.logger.info(f"💾 Knowledge Base: {'✅ Loaded' if self.stats['knowledge_base_loaded'] else '❌ Empty'}")
            self.logger.info("=" * 60)
            
            # Generate learning report periodically
            if total_trades > 0 and total_trades % 50 == 0 and hasattr(self.learning_bot, 'generate_learning_report'):
                self.learning_bot.generate_learning_report()
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
    
    def run_web_interface(self):
        """Run web interface"""
        try:
            self.web_app.run(
                host=self.web_config.HOST,
                port=self.web_config.PORT,
                debug=self.web_config.DEBUG,
                use_reloader=False  # Important for multi-threading
            )
        except Exception as e:
            self.logger.error(f"Web interface error: {e}")
    
    def initialize_system(self):
        """Initialize the trading system with enhanced learning"""
        try:
            self.logger.info("🔧 Initializing Universal Trading System with Cumulative Learning...")
            self.stats['system_status'] = 'Initializing'
            
            # Test API connectivity
            if not self.binance_client.test_connectivity():
                self.logger.error("❌ API connectivity test failed")
                self.stats['system_status'] = 'API Error'
                return False
            
            # Get initial balance
            initial_balance = self.binance_client.get_account_balance()
            self.logger.info(f"💰 Initial Balance: ${initial_balance:.2f}")
            
            if initial_balance < self.config.MIN_POSITION_SIZE:
                self.logger.warning(f"⚠️ Low balance: ${initial_balance:.2f} (Minimum: ${self.config.MIN_POSITION_SIZE:.2f})")
            
            # Initialize learning system first
            self._initialize_learning_system()
            
            # Recover open positions
            self.recover_open_positions()
            
            # Update initial stats
            self.update_stats()
            self.stats['system_status'] = 'Running'
            
            self.logger.info("✅ System initialization completed with cumulative learning")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            self.stats['system_status'] = 'Initialization Failed'
            return False

    def _initialize_learning_system(self):
        """Initialize learning and training systems"""
        try:
            # Importar aqui para evitar dependência circular
            from bots.learning_initializer import LearningInitializer
            
            self.logger.info("🧠 Initializing learning system...")
            
            # Verificar e inicializar dados de aprendizado se necessário
            initializer = LearningInitializer()
            initialization_result = initializer.initialize_learning_data()
            
            if initialization_result:
                self.logger.info("✅ Learning data initialized successfully")
            else:
                self.logger.warning("⚠️ Learning data initialization had issues, but continuing...")
            
            # Load learning knowledge base
            if hasattr(self.learning_bot, 'load_knowledge_base'):
                self.learning_bot.load_knowledge_base()
                self.logger.info("✅ Learning knowledge base loaded")
            
            # Load training history
            if hasattr(self.training_bot, 'load_training_history'):
                self.training_bot.load_training_history()
                self.logger.info("✅ Training history loaded")
            
            # Generate initial learning report if we have data
            if (hasattr(self.learning_bot, 'performance_data') and 
                len(self.learning_bot.performance_data) > 0 and
                hasattr(self.learning_bot, 'generate_learning_report')):
                self.learning_bot.generate_learning_report()
                self.logger.info("✅ Initial learning report generated")
            
            # Run initial training if no training data exists
            if (hasattr(self.training_bot, 'training_data') and 
                len(self.training_bot.training_data) == 0 and
                hasattr(self.training_bot, 'run_quick_training')):
                self.logger.info("🚀 Running initial training session...")
                self.training_bot.run_quick_training()
            
        except ImportError:
            self.logger.warning("⚠️ Learning initializer not available, starting with empty knowledge base")
        except Exception as e:
            self.logger.error(f"Error initializing learning system: {e}")
    
    def start(self):
        """Start the trading system with enhanced bots"""
        self.logger.info("🚀 Starting Universal Trading System with Enhanced Learning...")
        
        # Initialize system
        if not self.initialize_system():
            self.logger.error("❌ Failed to initialize system")
            return
        
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Start bots in separate threads
            scanner_thread = Thread(target=self.run_scanner_bot, daemon=True)
            executor_thread = Thread(target=self.run_executor_bot, daemon=True)
            monitor_thread = Thread(target=self.run_monitor_bot, daemon=True)
            training_thread = Thread(target=self.run_training_bot, daemon=True)
            learning_thread = Thread(target=self.run_learning_bot, daemon=True)
            report_thread = Thread(target=self.run_performance_report, daemon=True)
            
            scanner_thread.start()
            executor_thread.start()
            monitor_thread.start()
            training_thread.start()
            learning_thread.start()
            report_thread.start()
            
            self.logger.info("✅ All trading bots started")
            self.logger.info(f"🌐 Web interface available at: http://{self.web_config.HOST}:{self.web_config.PORT}")
            self.logger.info("📊 Performance reports every 5 minutes")
            self.logger.info("🤖 Training bot running scheduled trainings")
            self.logger.info("🧠 Learning bot optimizing strategies cumulatively")
            self.logger.info("💾 Knowledge base persistence enabled")
            self.logger.info("🎯 Learning system initialized with sample data")
            
            # Run web interface in main thread
            self.run_web_interface()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 System stopped by user")
        except Exception as e:
            self.logger.error(f"💥 System error: {e}")
        finally:
            self.running = False
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources before shutdown with knowledge preservation"""
        try:
            self.logger.info("🧹 Cleaning up resources and saving knowledge...")
            self.stats['system_status'] = 'Shutting Down'
            
            # Save learning knowledge
            if hasattr(self.learning_bot, 'save_knowledge_base'):
                self.learning_bot.save_knowledge_base()
                self.logger.info("✅ Learning knowledge saved")
            
            # Save training history
            if hasattr(self.training_bot, 'save_training_history'):
                self.training_bot.save_training_history()
                self.logger.info("✅ Training history saved")
            
            # Close all active positions if emergency stop is needed
            # Uncomment the following line if you want to close all positions on shutdown
            # self.monitor_bot.close_all_positions("SYSTEM_SHUTDOWN")
            
            self.logger.info("✅ Cleanup completed with knowledge preservation")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def emergency_stop(self):
        """Emergency stop - close all positions with learning record"""
        try:
            self.logger.warning("🛑 EMERGENCY STOP ACTIVATED")
            self.stats['system_status'] = 'Emergency Stop'
            
            # Record emergency stop in learning system
            emergency_record = {
                'timestamp': time.time(),
                'type': 'EMERGENCY_STOP',
                'active_positions': len(self.monitor_bot.active_positions),
                'reason': 'Manual activation'
            }
            
            # Close all positions through monitor bot
            positions = self.monitor_bot.active_positions.copy()
            closed_count = 0
            
            for position_id, position in positions.items():
                symbol = position['symbol']
                quantity = position['quantity']
                
                success = self.monitor_bot.close_position(position_id, position, "EMERGENCY_STOP")
                if success:
                    closed_count += 1
                    time.sleep(1)  # Small delay between closes
            
            self.logger.info(f"✅ Emergency stop completed: {closed_count} positions closed")
            
            # Update learning system about emergency stop
            if hasattr(self.learning_bot, 'performance_data'):
                self.learning_bot.performance_data.append(emergency_record)
            
            return closed_count
            
        except Exception as e:
            self.logger.error(f"❌ Emergency stop failed: {e}")
            return 0
    
    def get_learning_insights(self) -> Dict:
        """Get learning insights for web interface"""
        try:
            insights = {
                'performance_metrics': {},
                'strategy_effectiveness': {},
                'market_insights': [],
                'training_progress': {},
                'knowledge_base_status': {
                    'loaded': self.stats['knowledge_base_loaded'],
                    'performance_records': len(self.learning_bot.performance_data) if hasattr(self.learning_bot, 'performance_data') else 0,
                    'market_regimes': len(self.learning_bot.market_regime_knowledge) if hasattr(self.learning_bot, 'market_regime_knowledge') else 0,
                    'strategy_patterns': len(self.learning_bot.strategy_patterns) if hasattr(self.learning_bot, 'strategy_patterns') else 0
                }
            }
            
            # Get performance metrics from learning bot
            if hasattr(self.learning_bot, 'strategy_metrics'):
                insights['performance_metrics'] = self.learning_bot.strategy_metrics
            
            # Get training progress
            if hasattr(self.training_bot, 'training_data'):
                insights['training_progress'] = {
                    'total_samples': len(self.training_bot.training_data),
                    'recent_quality': np.mean([d.get('quality_score', 0) for d in self.training_bot.training_data[-20:]]) if self.training_bot.training_data else 0,
                    'training_sessions': len(self.training_bot.training_history) if hasattr(self.training_bot, 'training_history') else 0
                }
            
            # Get market insights
            if hasattr(self.learning_bot, 'generate_performance_insights'):
                insights['market_insights'] = self.learning_bot.generate_performance_insights()
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting learning insights: {e}")
            return {
                'performance_metrics': {},
                'strategy_effectiveness': {},
                'market_insights': [],
                'training_progress': {},
                'knowledge_base_status': {'loaded': False, 'performance_records': 0, 'market_regimes': 0, 'strategy_patterns': 0}
            }

if __name__ == "__main__":
    try:
        system = UniversalTradingSystem()
        system.start()
    except Exception as e:
        logging.error(f"💥 Failed to start Universal Trading System: {e}")
        sys.exit(1)