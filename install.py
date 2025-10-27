#!/usr/bin/env python3
"""
Installation script for Universal Trading System
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python 3.11 or higher is required")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'data',
        'models',
        'strategies',
        'bots',
        'dql_agent',
        'web_interface/templates',
        'web_interface/static',
        'utils'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def setup_environment():
    """Setup environment variables"""
    env_file = '.env'
    if not os.path.exists(env_file):
        with open(env_file, 'w') as f:
            f.write("""# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
BINANCE_TESTNET=true

# Trading Configuration
INITIAL_CAPITAL=40.0
MAX_CONCURRENT_TRADES=4
MIN_POSITION_SIZE=8.0
MAX_POSITION_SIZE=20.0

# Risk Management
MAX_DRAWDOWN=0.10
DAILY_LOSS_LIMIT=0.05

# Web Interface
SECRET_KEY=your_secret_key_here
WEB_HOST=0.0.0.0
WEB_PORT=5000
DEBUG=false
""")
        print("✅ Created .env file - Please configure your API keys")
    else:
        print("✅ .env file already exists")

def main():
    """Main installation function"""
    print("🚀 Universal Trading System - Installation")
    print("=" * 50)
    
    try:
        check_python_version()
        install_requirements()
        create_directories()
        setup_environment()
        
        print("\n🎉 Installation completed successfully!")
        print("\n📝 Next steps:")
        print("1. Edit the .env file with your Binance API keys")
        print("2. Set BINANCE_TESTNET=true for testing")
        print("3. Run: python main.py")
        print("4. Open http://localhost:5000 in your browser")
        
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()