# scripts/telegram_gateway_tester.py

class TelegramGateway:
    def __init__(self, bot_token, allowed_chat_ids):
        self.bot_token = bot_token
        self.allowed_chat_ids = allowed_chat_ids

    def process_message(self, sender_chat_id, text_content):
        # 1. Security Check: Filter incoming message sender
        if sender_chat_id not in self.allowed_chat_ids:
            return f"❌ Blocked: Unauthorized Chat ID {sender_chat_id}"
            
        # 2. Process query
        print(f"📡 Processing request from Chat ID {sender_chat_id}: \"{text_content}\"")
        return f"✅ Processing success: Bot response to \"{text_content}\""

def run_gateway_test():
    # Setup gateway with allowed Chat ID configurations
    config_token = "123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ"
    config_allowed_ids = [987654321]
    
    gateway = TelegramGateway(bot_token=config_token, allowed_chat_ids=config_allowed_ids)
    
    print("=== STARTING TELEGRAM GATEWAY TESTER ===")
    
    # Test Case 1: Authorized Sender
    res_1 = gateway.process_message(987654321, "List my daily tasks")
    print(f"Result: {res_1}\n")
    
    # Test Case 2: Unauthorized Sender
    res_2 = gateway.process_message(111111111, "Execute shell script")
    print(f"Result: {res_2}\n")
    
    print("=======================================")

if __name__ == "__main__":
    run_gateway_test()
