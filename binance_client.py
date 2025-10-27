import os
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from binance.spot import Spot
from binance.error import ClientError
from decimal import Decimal, ROUND_DOWN
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import TradingConfig
from utils.logger import setup_logger

class BinanceClient:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = setup_logger('binance_client')
        
        # Initialize Binance client
        base_url = 'https://testnet.binance.vision' if config.TESTNET else 'https://api.binance.com'
        self.client = Spot(
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            base_url=base_url
        )
        
        # Setup session with retry strategy
        self.session = self._create_session()
        
        # Caches
        self.symbol_info_cache = {}
        self.price_cache = {}
        self.volume_cache = {}
        self.cache_timeout = 5
        self.valid_symbols_cache = None
        self.valid_symbols_cache_time = 0
        
        self.logger.info("Binance client initialized")

    def _create_session(self):
        """Create session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=50, pool_maxsize=100)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def safe_api_call(self, api_func, *args, **kwargs):
        """Safe API call with retry mechanism and enhanced error handling"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                result = api_func(*args, **kwargs)
                return result
                
            except ClientError as e:
                error_code = getattr(e, 'code', None)
                error_message = getattr(e, 'message', str(e))
                
                self.logger.error(f"Binance API error in {api_func.__name__}: {error_message} (Code: {error_code})")
                
                # Don't retry on authentication errors
                if error_code in [-2015, -2014, -2010]:  # Invalid API key, format, etc.
                    self.logger.error("Authentication error - check API keys and permissions")
                    raise e
                
                # Rate limiting - wait and retry
                if error_code == -1003 or error_code == -1008:  # Too many requests
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                    
                # Timestamp errors - check system time
                if error_code == -1021:  # Timestamp outside recvWindow
                    self.logger.error("System time synchronization issue detected")
                    self._check_time_sync()
                    raise e
                
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = 1 * (attempt + 1)
                self.logger.debug(f"Retry {attempt + 1} for {api_func.__name__}, waiting {wait_time}s")
                time.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Unexpected error in {api_func.__name__}: {e}")
                if attempt == max_retries - 1:
                    raise e
                wait_time = 1 * (attempt + 1)
                self.logger.debug(f"Retry {attempt + 1} for {api_func.__name__}, waiting {wait_time}s")
                time.sleep(wait_time)

    def _check_time_sync(self):
        """Check and log time synchronization status"""
        try:
            server_time = self.client.time()
            local_time = int(time.time() * 1000)
            time_diff = abs(server_time['serverTime'] - local_time)
            
            self.logger.info(f"Time sync check - Server: {server_time['serverTime']}, Local: {local_time}, Diff: {time_diff}ms")
            
            if time_diff > 1000:  # More than 1 second difference
                self.logger.warning(f"Significant time difference detected: {time_diff}ms. Consider syncing system time.")
            else:
                self.logger.info("Time synchronization OK")
                
        except Exception as e:
            self.logger.error(f"Error checking time sync: {e}")

    def test_connectivity(self) -> bool:
        """Test API connectivity - CORRIGIDO"""
        try:
            # Use a simple API call to test connectivity
            self.safe_api_call(self.client.time)
            self.logger.info("✅ API connectivity test passed")
            return True
        except ClientError as e:
            error_code = getattr(e, 'code', None)
            if error_code == -2015:
                self.logger.error("❌ API connectivity failed: Invalid API key")
            elif error_code == -2014:
                self.logger.error("❌ API connectivity failed: API key format invalid")
            else:
                self.logger.error(f"❌ API connectivity failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ API connectivity test failed: {e}")
            return False

    def get_valid_trading_symbols(self) -> List[str]:
        """Get valid trading symbols with cache"""
        current_time = time.time()
        if (self.valid_symbols_cache and 
            current_time - self.valid_symbols_cache_time < 300):  # 5 minutes cache
            return self.valid_symbols_cache
        
        try:
            exchange_info = self.safe_api_call(self.client.exchange_info)
            valid_symbols = []
            
            for symbol_info in exchange_info['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['quoteAsset'] == 'USDT' and
                    symbol_info['isSpotTradingAllowed']):
                    
                    symbol = symbol_info['symbol']
                    
                    # Check if symbol has necessary filters
                    lot_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                    notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'NOTIONAL'), None)
                    
                    if lot_filter and notional_filter:
                        min_notional = float(notional_filter.get('minNotional', 10.0))
                        
                        # Only include symbols with reasonable minimum notional
                        if min_notional <= 50.0:
                            valid_symbols.append(symbol)
            
            self.valid_symbols_cache = valid_symbols
            self.valid_symbols_cache_time = current_time
            self.logger.info(f"Loaded {len(valid_symbols)} valid trading symbols")
            return valid_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting valid symbols: {e}")
            # Fallback to major symbols
            return ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'BNBUSDT', 
                   'XRPUSDT', 'SOLUSDT', 'MATICUSDT', 'LTCUSDT']

    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid for trading"""
        valid_symbols = self.get_valid_trading_symbols()
        return symbol in valid_symbols

    def get_account_balance(self) -> float:
        """Get USDT balance"""
        try:
            account = self.safe_api_call(self.client.account)
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    self.logger.debug(f"USDT balance: {usdt_balance}")
                    return usdt_balance
            return 0.0
        except ClientError as e:
            error_code = getattr(e, 'code', None)
            if error_code == -2015:
                self.logger.error("Failed to get account balance: Invalid API key or permissions")
            else:
                self.logger.error(f"Failed to get account balance: {e}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Failed to get account balance: {e}")
            return 0.0

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information with cache - COM VALIDAÇÃO"""
        if not self.is_valid_symbol(symbol):
            self.logger.debug(f"Symbol {symbol} is not valid for trading")
            return None
            
        if symbol in self.symbol_info_cache:
            cached_info = self.symbol_info_cache[symbol]
            if time.time() - cached_info['timestamp'] < 300:
                return cached_info['info']
        
        try:
            info = self.safe_api_call(self.client.exchange_info, symbol=symbol)
            if info and 'symbols' in info and len(info['symbols']) > 0:
                symbol_info = info['symbols'][0]
                self.symbol_info_cache[symbol] = {
                    'info': symbol_info,
                    'timestamp': time.time()
                }
                return symbol_info
        except ClientError as e:
            error_code = getattr(e, 'code', None)
            if error_code == -1121:  # Invalid symbol
                self.logger.debug(f"Invalid symbol: {symbol}")
            else:
                self.logger.debug(f"Error getting symbol info {symbol}: {e}")
        except Exception as e:
            self.logger.debug(f"Error getting symbol info {symbol}: {e}")
        
        return None

    def get_current_price(self, symbol: str) -> float:
        """Get current price with cache - COM VALIDAÇÃO"""
        if not self.is_valid_symbol(symbol):
            self.logger.debug(f"Invalid symbol for price check: {symbol}")
            return 0.0
            
        if symbol in self.price_cache:
            cached_price = self.price_cache[symbol]
            if time.time() - cached_price['timestamp'] < 2:
                return cached_price['price']
        
        try:
            ticker = self.safe_api_call(self.client.ticker_price, symbol)
            price = float(ticker['price'])
            self.price_cache[symbol] = {'price': price, 'timestamp': time.time()}
            return price
        except ClientError as e:
            error_code = getattr(e, 'code', None)
            if error_code == -1121:  # Invalid symbol
                self.logger.debug(f"Invalid symbol for price: {symbol}")
            else:
                self.logger.debug(f"Error getting price for {symbol}: {e}")
            return 0.0
        except Exception as e:
            self.logger.debug(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_klines(self, symbol: str, interval: str = '3m', limit: int = 50) -> Optional[pd.DataFrame]:
        """Get klines data - COM VALIDAÇÃO"""
        if not self.is_valid_symbol(symbol):
            self.logger.debug(f"Invalid symbol for klines: {symbol}")
            return None
            
        try:
            klines = self.safe_api_call(self.client.klines, symbol, interval, limit=limit)
            
            if not klines or len(klines) < 25:
                self.logger.debug(f"Insufficient klines data for {symbol}")
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df
        except ClientError as e:
            error_code = getattr(e, 'code', None)
            if error_code == -1121:  # Invalid symbol
                self.logger.debug(f"Invalid symbol for klines: {symbol}")
            else:
                self.logger.debug(f"Error getting klines for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.debug(f"Error getting klines for {symbol}: {e}")
            return None

    def get_lot_size_info(self, symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get LOT_SIZE information for a symbol - COM VALIDAÇÃO"""
        if not self.is_valid_symbol(symbol):
            return None, None, None
            
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return None, None, None
        
        lot_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_filter:
            return None, None, None
        
        min_qty = float(lot_filter['minQty'])
        max_qty = float(lot_filter.get('maxQty', 10000000.0))
        step_size = float(lot_filter.get('stepSize', 0.00001))
        
        return min_qty, max_qty, step_size

    def adjust_quantity_to_precision(self, quantity: float, step_size: float) -> float:
        """Adjust quantity to step_size precision"""
        if step_size <= 0:
            return quantity
        
        quantity_dec = Decimal(str(quantity))
        step_dec = Decimal(str(step_size))
        
        adjusted_quantity = (quantity_dec // step_dec) * step_dec
        
        if adjusted_quantity <= 0 and quantity > 0:
            adjusted_quantity = step_dec
        
        return float(adjusted_quantity)

    def format_quantity_precision(self, quantity: float, step_size: float) -> str:
        """Format quantity with correct precision"""
        if step_size <= 0:
            return str(quantity)
        
        step_str = str(step_size).rstrip('0')
        if 'e-' in step_str:
            precision = int(step_str.split('e-')[1])
        elif '.' in step_str:
            precision = len(step_str.split('.')[1])
        else:
            precision = 0
        
        if precision > 0:
            formatted = f"{quantity:.{precision}f}"
        else:
            formatted = f"{int(quantity)}"
        
        return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

    def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        """Place market order - COM VALIDAÇÃO"""
        if not self.is_valid_symbol(symbol):
            self.logger.error(f"Invalid symbol for trading: {symbol}")
            return None
            
        try:
            min_qty, max_qty, step_size = self.get_lot_size_info(symbol)
            if min_qty is None:
                self.logger.error(f"Could not get lot size info for {symbol}")
                return None
            
            # Adjust quantity to precision
            adjusted_quantity = self.adjust_quantity_to_precision(quantity, step_size)
            
            # Check minimum quantity
            if adjusted_quantity < min_qty:
                self.logger.error(f"Quantity {adjusted_quantity} below minimum {min_qty} for {symbol}")
                return None
            
            # Check maximum quantity
            if adjusted_quantity > max_qty:
                self.logger.error(f"Quantity {adjusted_quantity} above maximum {max_qty} for {symbol}")
                return None
            
            # Format quantity with correct precision
            formatted_quantity = self.format_quantity_precision(adjusted_quantity, step_size)
            
            self.logger.info(f"Placing {side} market order for {symbol}: {formatted_quantity}")
            
            # Place the order
            order = self.safe_api_call(
                self.client.new_order,
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=formatted_quantity
            )
            
            if order and 'orderId' in order:
                self.logger.info(f"✅ Order executed successfully: {order['orderId']}")
                
                # Log order details
                if order.get('fills'):
                    total_qty = sum(float(fill['qty']) for fill in order['fills'])
                    avg_price = sum(float(fill['price']) * float(fill['qty']) for fill in order['fills']) / total_qty
                    self.logger.info(f"  Executed: {total_qty} at average price: {avg_price}")
                
                return order
            else:
                self.logger.error("Order placement failed - no order ID returned")
                return None
                
        except ClientError as e:
            error_code = getattr(e, 'code', None)
            error_message = getattr(e, 'message', str(e))
            
            if error_code == -1013:  # Filter failure
                self.logger.error(f"Order quantity filter failure: {error_message}")
            elif error_code == -2010:  # Insufficient balance
                self.logger.error(f"Insufficient balance for order: {error_message}")
            elif error_code == -1111:  # Lot size
                self.logger.error(f"Lot size validation failed: {error_message}")
            else:
                self.logger.error(f"Binance client error placing order: {error_message}")
                
            return None
        except Exception as e:
            self.logger.error(f"Error placing {side} order for {symbol}: {e}")
            return None

    def get_open_positions(self) -> List[Dict]:
        """Recover open positions from account - CORRIGIDO"""
        try:
            account_info = self.safe_api_call(self.client.account)
            open_positions = []
            valid_symbols = self.get_valid_trading_symbols()
            
            for balance in account_info['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                # Skip USDT and zero balances
                if asset == 'USDT' or total <= 0:
                    continue
                
                # Find matching trading pairs - only create symbol if asset is valid
                symbol = f"{asset}USDT"
                
                # Check if this is a valid trading symbol
                if symbol not in valid_symbols:
                    self.logger.debug(f"Skipping invalid symbol: {symbol}")
                    continue
                
                try:
                    current_price = self.get_current_price(symbol)
                    if current_price > 0:
                        position_value = total * current_price
                        
                        # Only consider positions with significant value
                        if position_value >= 5.0:
                            # Estimate entry price from recent trades
                            entry_price = self.estimate_entry_price(symbol, asset)
                            
                            open_positions.append({
                                'symbol': symbol,
                                'asset': asset,
                                'quantity': total,
                                'current_price': current_price,
                                'position_value': position_value,
                                'estimated_entry_price': entry_price,
                                'unrealized_pnl': (current_price - entry_price) * total,
                                'unrealized_pnl_percent': ((current_price - entry_price) / entry_price) * 100
                            })
                except Exception as e:
                    self.logger.debug(f"Error processing position for {symbol}: {e}")
                    continue
            
            self.logger.info(f"Recovered {len(open_positions)} valid open positions")
            return open_positions
            
        except ClientError as e:
            error_code = getattr(e, 'code', None)
            if error_code == -2015:
                self.logger.error("Cannot get account info: Invalid API key or permissions")
            else:
                self.logger.error(f"Error recovering open positions: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error recovering open positions: {e}")
            return []

    def estimate_entry_price(self, symbol: str, asset: str) -> float:
        """Estimate entry price from recent trades - COM VALIDAÇÃO"""
        if not self.is_valid_symbol(symbol):
            current_price = self.get_current_price(symbol)
            return current_price * 0.995
            
        try:
            # Get recent trades for this symbol
            trades = self.safe_api_call(self.client.my_trades, symbol=symbol, limit=10)
            
            if trades:
                buy_trades = [t for t in trades if t['isBuyer'] == True]
                if buy_trades:
                    total_quantity = 0
                    total_value = 0
                    
                    for trade in buy_trades:
                        qty = float(trade['qty'])
                        price = float(trade['price'])
                        total_quantity += qty
                        total_value += qty * price
                    
                    if total_quantity > 0:
                        return total_value / total_quantity
            
            # Fallback: use current price with a small discount
            current_price = self.get_current_price(symbol)
            return current_price * 0.995
            
        except ClientError as e:
            error_code = getattr(e, 'code', None)
            if error_code == -2015:
                self.logger.debug(f"Cannot get trades for {symbol}: API permission issue")
            else:
                self.logger.debug(f"Error estimating entry price for {symbol}: {e}")
            current_price = self.get_current_price(symbol)
            return current_price
        except Exception as e:
            self.logger.debug(f"Error estimating entry price for {symbol}: {e}")
            current_price = self.get_current_price(symbol)
            return current_price

    def get_all_symbols(self) -> List[str]:
        """Get all available trading symbols"""
        return self.get_valid_trading_symbols()

    def get_server_time(self) -> int:
        """Get server time for synchronization"""
        try:
            server_time = self.safe_api_call(self.client.time)
            return server_time['serverTime']
        except Exception as e:
            self.logger.error(f"Error getting server time: {e}")
            return int(time.time() * 1000)

    def get_account_status(self) -> Dict:
        """Get account status"""
        try:
            return self.safe_api_call(self.client.account_status)
        except Exception as e:
            self.logger.error(f"Error getting account status: {e}")
            return {}

    # Métodos auxiliares para cálculo de fees
    def calculate_fee_cost(self, notional_value: float, side: str = 'BUY') -> float:
        """Calculate fee cost for a trade"""
        fee_rate = self.config.TAKER_FEE  # Market orders use taker fee
        return notional_value * fee_rate

    def calculate_net_profit(self, entry_notional: float, exit_notional: float, 
                           entry_fee: float, exit_fee: float) -> Tuple[float, float]:
        """Calculate net profit including all fees"""
        gross_profit = exit_notional - entry_notional
        total_fees = entry_fee + exit_fee
        net_profit = gross_profit - total_fees
        return net_profit, total_fees

    def get_24h_volume(self, symbol: str) -> float:
        """Get 24h volume for symbol - COM VALIDAÇÃO"""
        if not self.is_valid_symbol(symbol):
            return 0.0
            
        try:
            if symbol in self.volume_cache:
                cached_volume = self.volume_cache[symbol]
                if time.time() - cached_volume['timestamp'] < 60:
                    return cached_volume['volume']
            
            ticker = self.safe_api_call(self.client.ticker_24hr, symbol)
            volume = float(ticker.get('quoteVolume', 0))
            self.volume_cache[symbol] = {'volume': volume, 'timestamp': time.time()}
            return volume
        except ClientError as e:
            error_code = getattr(e, 'code', None)
            if error_code == -1121:  # Invalid symbol
                self.logger.debug(f"Invalid symbol for volume: {symbol}")
            else:
                self.logger.debug(f"Error getting volume for {symbol}: {e}")
            return 0.0
        except Exception as e:
            self.logger.debug(f"Error getting volume for {symbol}: {e}")
            return 0.0

    def get_current_spread(self, symbol: str) -> float:
        """Calculate current spread percentage - COM VALIDAÇÃO"""
        if not self.is_valid_symbol(symbol):
            return 0.1
            
        try:
            ticker = self.safe_api_call(self.client.book_ticker, symbol)
            if ticker:
                best_bid = float(ticker['bidPrice'])
                best_ask = float(ticker['askPrice'])
                spread_percent = ((best_ask - best_bid) / best_bid) * 100
                return spread_percent
        except ClientError as e:
            error_code = getattr(e, 'code', None)
            if error_code == -1121:  # Invalid symbol
                self.logger.debug(f"Invalid symbol for spread: {symbol}")
            else:
                self.logger.debug(f"Spread calculation failed for {symbol}: {e}")
        except Exception as e:
            self.logger.debug(f"Spread calculation failed for {symbol}: {e}")
        return 0.1