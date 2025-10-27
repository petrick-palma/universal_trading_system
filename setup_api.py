import os
import sys
from config import TradingConfig

def setup_api_config():
    """Interactive API configuration setup"""
    print("🔧 Binance API Configuration Setup")
    print("=" * 50)
    
    # Check current configuration
    config = TradingConfig()
    print(f"Current TESTNET mode: {config.TESTNET}")
    print(f"Current API Key: {config.API_KEY[:10]}...{config.API_KEY[-4:] if config.API_KEY else 'None'}")
    
    print("\n📝 Let's configure your API credentials:")
    
    # Testnet or Mainnet
    while True:
        use_testnet = input("Use TESTNET? (recommended for development) [y/N]: ").lower().strip()
        if use_testnet in ['y', 'yes', '']:
            testnet = True
            print("✅ Using TESTNET (https://testnet.binance.vision)")
            break
        elif use_testnet in ['n', 'no']:
            testnet = False
            print("⚠️  Using MAINNET - Be careful with real funds!")
            break
        else:
            print("Please enter 'y' for testnet or 'n' for mainnet")
    
    # Get API Key
    api_key = input("\nEnter your Binance API Key: ").strip()
    if not api_key:
        print("❌ API Key is required!")
        return False
    
    # Get API Secret
    api_secret = input("Enter your Binance API Secret: ").strip()
    if not api_secret:
        print("❌ API Secret is required!")
        return False
    
    # Test the credentials
    print("\n🔐 Testing API credentials...")
    
    try:
        from binance.spot import Spot
        from binance.error import ClientError
        
        base_url = 'https://testnet.binance.vision' if testnet else 'https://api.binance.com'
        client = Spot(api_key=api_key, api_secret=api_secret, base_url=base_url)
        
        # Test authentication
        account_info = client.account()
        print("✅ API credentials are valid!")
        print(f"✅ Account can make trades: {account_info.get('canTrade', False)}")
        print(f"✅ Account can withdraw: {account_info.get('canWithdraw', False)}")
        print(f"✅ Account can deposit: {account_info.get('canDeposit', False)}")
        
        # Show balances
        print("\n💰 Account Balances (top 5):")
        balances = [b for b in account_info['balances'] if float(b['free']) > 0 or float(b['locked']) > 0]
        for balance in balances[:5]:
            asset = balance['asset']
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                print(f"   {asset}: Free={free:.8f}, Locked={locked:.8f}")
        
    except ClientError as e:
        print(f"❌ API Error: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Go to Binance website → API Management")
        if testnet:
            print("2. For TESTNET: go to https://testnet.binance.vision")
            print("3. Create API key with 'Enable Trading' permission")
        else:
            print("2. For MAINNET: go to Binance.com → API Management")
            print("3. Create API key with 'Enable Reading' and 'Enable Spot & Margin Trading'")
        print("4. If using IP restrictions, add your current IP address")
        print("5. Make sure you're using the correct environment (Testnet/Mainnet)")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Save to environment file
    print("\n💾 Saving configuration...")
    env_content = f"""# Binance API Configuration
BINANCE_API_KEY={api_key}
BINANCE_API_SECRET={api_secret}
BINANCE_TESTNET={str(testnet).lower()}

# Trading Configuration
MIN_POSITION_SIZE=10
MAX_POSITION_SIZE=100
MAX_OPEN_POSITIONS=5
RISK_PER_TRADE=0.02

# Bot Configuration
SCAN_INTERVAL=60
MONITOR_INTERVAL=30
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Configuration saved to .env file")
    print("🔒 Make sure to keep your .env file secure and don't commit it to version control!")
    
    return True

def show_help_links():
    """Show helpful links for API setup"""
    print("\n🔗 Helpful Links:")
    print("• Binance Testnet: https://testnet.binance.vision/")
    print("• Binance API Documentation: https://binance-docs.github.io/apidocs/spot/en/")
    print("• Create Testnet API Key: https://testnet.binance.vision/")
    print("• Create Mainnet API Key: https://www.binance.com/en/my/settings/api-management")

if __name__ == "__main__":
    print("Universal Trading System - API Setup")
    print("=" * 50)
    
    if setup_api_config():
        print("\n🎉 Setup completed successfully!")
        print("You can now run the trading system with: python main.py")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        show_help_links()
        sys.exit(1)