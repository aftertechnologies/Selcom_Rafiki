import os
import asyncio
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
from datetime import datetime

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Load environment variables
load_dotenv()

# Data Models
class UserProfile(BaseModel):
    phone_number: str = ""
    name: str = ""
    pin: str = ""
    account_status: str = "not_registered"
    balance: float = 0.0

class ChatContext(BaseModel):
    user_profile: UserProfile = UserProfile()
    current_process: str = ""
    process_step: int = 0
    session_data: Dict[str, Any] = {}
    conversation_history: List[Dict[str, str]] = []
    language: str = "auto"  # auto, en, sw

# Language Detection and Responses
class LanguageManager:
    def __init__(self):
        self.swahili_keywords = [
            'habari', 'shikamoo', 'mambo', 'vipi', 'poa', 'sawa', 'asante', 'karibu',
            'nataka', 'nina', 'nini', 'wapi', 'namna', 'jinsi', 'kuweza', 'kuwa',
            'akaunti', 'salio', 'pesa', 'kutuma', 'kupokea', 'msaada', 'huduma',
            'tengeneza', 'unda', 'angalia', 'ona', 'tuma', 'pokea', 'fungua'
        ]
        
        self.responses = {
            'greeting': {
                'en': "Hello! How can I help you with Selcom Pesa services today?",
                'sw': "Shikamoo! Nisaidie kwa swali lolote kuhusu Selcom Pesa leo."
            },
            'account_creation_start': {
                'en': "Let's create your Selcom Pesa account. Please enter your full name:",
                'sw': "Tuanze kuunda akaunti yako ya Selcom Pesa. Tafadhali andika jina lako la kamili:"
            },
            'ask_phone': {
                'en': "Thank you, {name}. Now enter your phone number (format: +255XXXXXXXXX):",
                'sw': "Asante, {name}. Sasa andika nambari yako ya simu (fomati: +255XXXXXXXXX):"
            },
            'ask_pin': {
                'en': "Great! Please set a 4-digit PIN:",
                'sw': "Sawa! Tafadhali weka PIN ya tarakimu 4:"
            },
            'account_created': {
                'en': "Your account has been created successfully!",
                'sw': "Akaunti yako imeundwa kwa mafanikio!"
            },
            'balance_check_start': {
                'en': "Let's check your Selcom Pesa account balance. Please enter your phone number (format: +255XXXXXXXXX):",
                'sw': "Tuanze kuangalia salio lako la akaunti ya Selcom Pesa. Tafadhali andika nambari yako ya simu (fomati: +255XXXXXXXXX):"
            },
            'enter_pin': {
                'en': "Great! Please enter your 4-digit PIN:",
                'sw': "Sawa! Tafadhali andika PIN yako ya tarakimu 4:"
            },
            'balance_result': {
                'en': "Your balance is TZS {balance:,.2f}",
                'sw': "Salio lako ni TZS {balance:,.2f}"
            },
            'goodbye': {
                'en': "Thank you for using Selcom Pesa services. Goodbye!",
                'sw': "Asante kwa kutumia huduma za Selcom Pesa. Kwaheri!"
            },
            'invalid_phone': {
                'en': "Please enter a valid phone number (format: +255XXXXXXXXX):",
                'sw': "Tafadhali andika nambari sahihi ya simu (fomati: +255XXXXXXXXX):"
            },
            'invalid_pin': {
                'en': "Please enter a valid 4-digit PIN:",
                'sw': "Tafadhali andika PIN sahihi ya tarakimu 4:"
            },
            'account_not_found': {
                'en': "Account not found. Would you like to create a new account?",
                'sw': "Akaunti haijapatikana. Je, ungependa kuunda akaunti mpya?"
            },
            'wrong_pin': {
                'en': "Incorrect PIN. Please try again:",
                'sw': "PIN si sahihi. Tafadhali jaribu tena:"
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Swahili or English"""
        text_lower = text.lower()
        swahili_count = sum(1 for word in self.swahili_keywords if word in text_lower)
        
        if swahili_count > 0:
            return 'sw'
        return 'en'
    
    def get_response(self, key: str, language: str, **kwargs) -> str:
        """Get response in specified language"""
        if key in self.responses and language in self.responses[key]:
            return self.responses[key][language].format(**kwargs)
        return self.responses[key]['en'].format(**kwargs)

# Knowledge Base Manager with ChromaDB
class KnowledgeBaseManager:
    def __init__(self, knowledge_file="selcom_knowledge_base.txt", persist_dir="./chroma_db"):
        self.knowledge_file = knowledge_file
        self.persist_dir = persist_dir
        self.index = None
        self.query_engine = None
        
    def initialize(self):
        """Initialize or load existing vector store"""
        # Set up embeddings
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        chroma_collection = chroma_client.get_or_create_collection("selcom_knowledge")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Check if index already exists
        try:
            self.index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            print("Loaded existing knowledge base from ChromaDB")
        except:
            # Create new index if it doesn't exist
            if not os.path.exists(self.knowledge_file):
                raise FileNotFoundError(f"Knowledge base file '{self.knowledge_file}' not found.")
            
            print("Creating new knowledge base index...")
            documents = SimpleDirectoryReader(input_files=[self.knowledge_file]).load_data()
            self.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            print("Knowledge base indexed and saved to ChromaDB")
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(similarity_top_k=3)
    
    def query(self, question: str) -> str:
        """Query the knowledge base"""
        if not self.query_engine:
            return "Knowledge base not initialized"
        
        try:
            response = self.query_engine.query(question)
            return str(response)
        except Exception as e:
            return f"Error querying knowledge base: {str(e)}"

# Mock Selcom API Client
class MockSelcomAPI:
    def __init__(self):
        self.accounts = {}  # phone_number -> UserProfile
    
    async def create_account(self, name: str, phone: str, pin: str) -> Dict:
        """Create a new account"""
        await asyncio.sleep(0.5)  # Simulate API delay
        
        if phone in self.accounts:
            return {"success": False, "message": "Account already exists"}
        
        # Create account with initial balance
        profile = UserProfile(
            name=name,
            phone_number=phone,
            pin=pin,
            account_status="active",
            balance=10000.00  # Initial balance
        )
        
        self.accounts[phone] = profile
        return {"success": True, "message": "Account created successfully"}
    
    async def check_balance(self, phone: str, pin: str) -> Dict:
        """Check account balance"""
        await asyncio.sleep(0.3)
        
        if phone not in self.accounts:
            return {"success": False, "message": "Account not found"}
        
        account = self.accounts[phone]
        if account.pin != pin:
            return {"success": False, "message": "Invalid PIN"}
        
        return {
            "success": True, 
            "balance": account.balance,
            "name": account.name
        }
    
    async def add_money(self, phone: str, pin: str, amount: float) -> Dict:
        """Add money to account"""
        await asyncio.sleep(0.5)
        
        if phone not in self.accounts:
            return {"success": False, "message": "Account not found"}
        
        account = self.accounts[phone]
        if account.pin != pin:
            return {"success": False, "message": "Invalid PIN"}
        
        account.balance += amount
        return {
            "success": True, 
            "message": f"Added TZS {amount:,.2f} to your account",
            "new_balance": account.balance
        }

# Pydantic AI Agent Tools
async def get_knowledge_info(ctx: RunContext[ChatContext], query: str) -> str:
    """Tool to query knowledge base via LlamaIndex"""
    kb_manager = ctx.deps.get('kb_manager')
    if kb_manager:
        return kb_manager.query(query)
    return "Knowledge base not available"

async def start_account_creation(ctx: RunContext[ChatContext]) -> str:
    """Tool to start account creation process"""
    context = ctx.data
    lang_manager = ctx.deps.get('lang_manager')
    
    context.current_process = "create_account"
    context.process_step = 1
    context.session_data = {}
    
    return lang_manager.get_response('account_creation_start', context.language)

async def start_balance_check(ctx: RunContext[ChatContext]) -> str:
    """Tool to start balance check process"""
    context = ctx.data
    lang_manager = ctx.deps.get('lang_manager')
    
    context.current_process = "check_balance"
    context.process_step = 1
    context.session_data = {}
    
    return lang_manager.get_response('balance_check_start', context.language)

def create_selcom_agent(kb_manager: KnowledgeBaseManager, lang_manager: LanguageManager):
    """Create the main Selcom assistant agent"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    model = OpenAIModel(
        model_name='gpt-3.5-turbo',
        provider=OpenAIProvider(api_key=api_key)
    )
    
    system_prompt = """You are Selcom Rafiki, an interactive assistant for Selcom Pesa services.

CORE CAPABILITIES:
- Answer questions using the knowledge base
- Help users start account creation process
- Help users start balance checking process
- Provide information in English or Swahili based on user's language

LANGUAGE RULES:
- Always respond in the same language the user uses (English or Swahili)
- Keep responses clear and concise
- Be helpful and professional

WHEN TO USE TOOLS:
- Use get_knowledge_info for general questions about Selcom Pesa services
- Use start_account_creation when users want to create a new account
- Use start_balance_check when users want to check their balance

IMPORTANT: You only help START processes. The actual step-by-step collection of information is handled separately. Just help identify what the user wants and use the appropriate tool."""
    
    agent = Agent(
        model,
        system_prompt=system_prompt,
        tools=[get_knowledge_info, start_account_creation, start_balance_check],
        deps_type=Dict[str, Any],
        retries=2
    )
    
    return agent

class SelcomRafikiAssistant:
    def __init__(self):
        print("Initializing Selcom Rafiki Assistant...")
        
        # Initialize language manager
        self.lang_manager = LanguageManager()
        
        # Initialize knowledge base with ChromaDB
        self.kb_manager = KnowledgeBaseManager()
        self.kb_manager.initialize()
        
        # Initialize API client
        self.api = MockSelcomAPI()
        
        # Initialize agent
        self.agent = create_selcom_agent(self.kb_manager, self.lang_manager)
        
        # Initialize context
        self.context = ChatContext()
        
        print("Assistant ready!")
    
    def is_greeting(self, text: str) -> bool:
        """Check if text is a greeting"""
        greetings = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'habari', 'shikamoo', 'mambo', 'vipi', 'hujambo'
        ]
        return any(greeting in text.lower() for greeting in greetings)
    
    def is_account_creation_request(self, text: str) -> bool:
        """Check if user wants to create account"""
        create_terms_en = ['create account', 'new account', 'register', 'sign up', 'open account']
        create_terms_sw = ['tengeneza akaunti', 'unda akaunti', 'fungua akaunti', 'jiandikishe', 'nataka akaunti']
        
        text_lower = text.lower()
        return (any(term in text_lower for term in create_terms_en) or 
                any(term in text_lower for term in create_terms_sw))
    
    def is_balance_check_request(self, text: str) -> bool:
        """Check if user wants to check balance"""
        balance_terms_en = ['check balance', 'my balance', 'balance', 'how much']
        balance_terms_sw = ['angalia salio', 'salio', 'pesa zangu', 'nina pesa ngapi']
        
        text_lower = text.lower()
        return (any(term in text_lower for term in balance_terms_en) or 
                any(term in text_lower for term in balance_terms_sw))
    
    def is_exit_command(self, text: str) -> bool:
        """Check if user wants to exit"""
        exit_terms = ['exit', 'quit', 'bye', 'goodbye', 'kwaheri', 'tutaonana']
        return text.lower().strip() in exit_terms
    
    def validate_phone(self, phone: str) -> bool:
        """Validate Tanzanian phone number"""
        phone = phone.strip()
        if phone.startswith('+255') and len(phone) == 13 and phone[4:].isdigit():
            return True
        if phone.startswith('0') and len(phone) == 10 and phone[1:].isdigit():
            return True
        return False
    
    def normalize_phone(self, phone: str) -> str:
        """Normalize phone number to +255XXXXXXXXX format"""
        phone = phone.strip()
        if phone.startswith('0'):
            return '+255' + phone[1:]
        return phone
    
    def validate_pin(self, pin: str) -> bool:
        """Validate 4-digit PIN"""
        return pin.strip().isdigit() and len(pin.strip()) == 4
    
    async def process_input(self, user_input: str) -> str:
        """Process user input and return response"""
        user_input = user_input.strip()
        
        # Detect language from input
        detected_lang = self.lang_manager.detect_language(user_input)
        if self.context.language == "auto":
            self.context.language = detected_lang
        
        # Handle exit command
        if self.is_exit_command(user_input):
            return self.lang_manager.get_response('goodbye', self.context.language)
        
        # Handle ongoing processes
        if self.context.current_process:
            return await self.handle_process_step(user_input)
        
        # Handle greetings directly
        if self.is_greeting(user_input):
            return self.lang_manager.get_response('greeting', self.context.language)
        
        # For other inputs, use the AI agent
        try:
            deps = {
                'kb_manager': self.kb_manager,
                'lang_manager': self.lang_manager
            }
            
            result = await self.agent.run(user_input, deps=deps, message_history=self.context.conversation_history)
            response = str(result.output) if hasattr(result, 'output') else str(result)
            
            # Update conversation history
            self.context.conversation_history.append({"role": "user", "content": user_input})
            self.context.conversation_history.append({"role": "assistant", "content": response})
            
            # Keep history manageable
            if len(self.context.conversation_history) > 10:
                self.context.conversation_history = self.context.conversation_history[-8:]
            
            return response
            
        except Exception as e:
            print(f"Agent error: {e}")
            # Fallback to direct processing
            if self.is_account_creation_request(user_input):
                self.context.current_process = "create_account"
                self.context.process_step = 1
                self.context.session_data = {}
                return self.lang_manager.get_response('account_creation_start', self.context.language)
            
            elif self.is_balance_check_request(user_input):
                self.context.current_process = "check_balance"
                self.context.process_step = 1
                self.context.session_data = {}
                return self.lang_manager.get_response('balance_check_start', self.context.language)
            
            else:
                # Try knowledge base directly
                kb_response = self.kb_manager.query(user_input)
                if kb_response and "Error" not in kb_response:
                    return kb_response
                
                if self.context.language == 'sw':
                    return "Samahani, sijaelewa swali lako. Je, ungependa kuunda akaunti au kuangalia salio?"
                else:
                    return "I'm sorry, I didn't understand your question. Would you like to create an account or check your balance?"
    
    async def handle_process_step(self, user_input: str) -> str:
        """Handle steps in ongoing processes"""
        if self.context.current_process == "create_account":
            return await self.handle_account_creation(user_input)
        elif self.context.current_process == "check_balance":
            return await self.handle_balance_check(user_input)
        
        return "Unknown process"
    
    async def handle_account_creation(self, user_input: str) -> str:
        """Handle account creation process"""
        if self.context.process_step == 1:  # Collect name
            name = user_input.strip()
            if len(name) >= 2:
                self.context.session_data['name'] = name
                self.context.process_step = 2
                return self.lang_manager.get_response('ask_phone', self.context.language, name=name)
            else:
                return self.lang_manager.get_response('account_creation_start', self.context.language)
        
        elif self.context.process_step == 2:  # Collect phone
            if self.validate_phone(user_input):
                phone = self.normalize_phone(user_input)
                self.context.session_data['phone'] = phone
                self.context.process_step = 3
                return self.lang_manager.get_response('ask_pin', self.context.language)
            else:
                return self.lang_manager.get_response('invalid_phone', self.context.language)
        
        elif self.context.process_step == 3:  # Collect PIN
            if self.validate_pin(user_input):
                pin = user_input.strip()
                self.context.session_data['pin'] = pin
                
                # Create account
                result = await self.api.create_account(
                    self.context.session_data['name'],
                    self.context.session_data['phone'],
                    pin
                )
                
                # Reset process
                self.context.current_process = ""
                self.context.process_step = 0
                
                if result['success']:
                    return self.lang_manager.get_response('account_created', self.context.language)
                else:
                    if self.context.language == 'sw':
                        return f"Hitilafu: {result['message']}"
                    return f"Error: {result['message']}"
            else:
                return self.lang_manager.get_response('invalid_pin', self.context.language)
        
        return "Unexpected step"
    
    async def handle_balance_check(self, user_input: str) -> str:
        """Handle balance check process"""
        if self.context.process_step == 1:  # Collect phone
            if self.validate_phone(user_input):
                phone = self.normalize_phone(user_input)
                self.context.session_data['phone'] = phone
                self.context.process_step = 2
                return self.lang_manager.get_response('enter_pin', self.context.language)
            else:
                return self.lang_manager.get_response('invalid_phone', self.context.language)
        
        elif self.context.process_step == 2:  # Collect PIN and check balance
            if self.validate_pin(user_input):
                pin = user_input.strip()
                
                result = await self.api.check_balance(
                    self.context.session_data['phone'],
                    pin
                )
                
                # Reset process
                self.context.current_process = ""
                self.context.process_step = 0
                
                if result['success']:
                    return self.lang_manager.get_response('balance_result', self.context.language, balance=result['balance'])
                else:
                    if result['message'] == "Account not found":
                        return self.lang_manager.get_response('account_not_found', self.context.language)
                    elif result['message'] == "Invalid PIN":
                        return self.lang_manager.get_response('wrong_pin', self.context.language)
                    else:
                        if self.context.language == 'sw':
                            return f"Hitilafu: {result['message']}"
                        return f"Error: {result['message']}"
            else:
                return self.lang_manager.get_response('invalid_pin', self.context.language)
        
        return "Unexpected step"
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "="*50)
        print("SELCOM RAFIKI - Interactive Assistant")
        print("="*50)
        print("I can help you with:")
        print("- Create a Selcom Pesa account")
        print("- Check your account balance")
        print("- Answer questions about Selcom Pesa")
        print("\nSupports English and Swahili")
        print("Type 'exit' to quit")
        print("-"*50)
    
    async def run(self):
        """Main conversation loop"""
        self.display_welcome()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                response = await self.process_input(user_input)
                print(f"Selcom Rafiki: {response}")
                
                # Check if user exited
                if self.is_exit_command(user_input):
                    break
                
            except KeyboardInterrupt:
                print(f"\n\nSelcom Rafiki: {self.lang_manager.get_response('goodbye', self.context.language)}")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

# Main execution
async def main():
    try:
        assistant = SelcomRafikiAssistant()
        await assistant.run()
    except Exception as e:
        print(f"Failed to start assistant: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())