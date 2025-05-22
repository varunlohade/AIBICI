#!/usr/bin/env python3
"""
Main module for the AiSpeach project.
"""
import asyncio
import os
from dotenv import load_dotenv
from groktemplate import GrokChatModel, ChatTemplate, ChatExecutor, Message

def get_chat_history(kwargs):
    """Convert chat history to string format"""
    return [Message(msg.role, msg.content) for msg in kwargs.get("chat_history", [])]

async def chat_loop(executor: ChatExecutor):
    """
    Run an interactive chat loop with the AI.
    """
    print("\nWelcome to the AI Chat Interface!")
    print("Type 'quit' to exit, 'summarize' to get a conversation summary.")
    print("------------------------------------------")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for special commands
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'summarize':
                await executor.summarize_history()
                continue
            
            # Get AI response
            response = await executor.invoke(user_input)
            
            # Print the response
            if "choices" in response and response["choices"]:
                print("\nAssistant:", response["choices"][0]["message"]["content"])
            else:
                print("\nUnexpected response format:", response)
                
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")

async def main():
    """
    Main function that serves as the entry point for the application.
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GROK_API_KEY")
    
    # Initialize the chat model
    model = GrokChatModel(api_key=api_key, temperature=0)
    
    # Define the chat template
    template = ChatTemplate([
        ("system", "You are a helpful personal AI assistant named TARS. You have a geeky, clever, sarcastic, and edgy sense of humor. You maintain context from previous conversations through summaries provided to you."),
        ("user", "{input}")
    ])
    
    # Initialize the executor
    executor = ChatExecutor(
        model=model,
        template=template,
        tools=[],  # We can add tools later
        verbose=False  # Set to True for debugging
    )
    
    # Start the chat loop
    await chat_loop(executor)

if __name__ == "__main__":
    asyncio.run(main()) 