import os
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up Google Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyACRb1eHvpR4YBQv2qAW71Wx-aytBqPRks"

# Load property data from a JSON file
with open("/content/property.json", "r") as f:
    property_data = json.load(f)

# Define prompt templates for different stages of conversation
property_search_prompt = PromptTemplate(
    input_variables=["location", "property_type", "bedrooms", "bathrooms", "min_price", "max_price"],
    template="I'm looking for a property in {location} with {property_type} and at least {bedrooms} bedrooms, {bathrooms} bathrooms, and a price range of ${min_price}-${max_price}. Do you have any suggestions?"
)

average_price_prompt = PromptTemplate(
    input_variables=["location", "property_type"],
    template="I'm interested in a property in {location}. What are the average prices for {property_type} in that area?"
)

amenities_prompt = PromptTemplate(
    input_variables=["location", "property_type"],
    template="I'm looking for a {property_type} in {location}. What are the schools and amenities nearby?"
)

property_details_prompt = PromptTemplate(
    input_variables=["address"],
    template="I'm interested in a property at {address}. Can you tell me more about it, such as its features and amenities?"
)

# Function to handle user queries and provide relevant responses
def handle_user_query(query_type, **kwargs):
    # Use the Gemini LLM to generate a response (searching web for real-time data)
    if query_type == "search":
        prompt = property_search_prompt.format(**kwargs)

        # Use the Gemini LLM to search the web for property listings matching user input
        response = llm.invoke([{"role": "user", "content": prompt}])

        # Assuming Gemini returns a structured format, process and display results
        if response:
            web_results = response.content  # Using the web content from Gemini

            # Format and display the result in a user-friendly way
            return f"Here are some properties that match your criteria from the web:\n{web_results}"

    elif query_type == "average_price":
        prompt = average_price_prompt.format(**kwargs)
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content

    elif query_type == "amenities":
        prompt = amenities_prompt.format(**kwargs)
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content

    elif query_type == "details":
        prompt = property_details_prompt.format(**kwargs)
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content

    else:
        return "Invalid query type."


# Create a Gemini LLM instance
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# Memory to store conversation history
memory = ConversationBufferMemory(memory_key="conversation_history")

# Main loop to handle user interactions
while True:
    print("\nSelect query type:\n1. Property Search\n2. Average Prices\n3. Nearby Amenities\n4. Property Details")
    query_type = input("Enter your choice (1-4): ")

    if query_type == "1":
        # Gather input for property search
        location = input("Enter location: ")
        property_type = input("Enter property type: ")
        bedrooms = input("Enter minimum number of bedrooms: ")
        bathrooms = input("Enter minimum number of bathrooms: ")
        min_price = input("Enter minimum price: ")
        max_price = input("Enter maximum price: ")

        response = handle_user_query(
            query_type="search",
            location=location,
            property_type=property_type,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            min_price=min_price,
            max_price=max_price
        )

    elif query_type == "2":
        location = input("Enter location: ")
        property_type = input("Enter property type: ")

        response = handle_user_query(
            query_type="average_price",
            location=location,
            property_type=property_type
        )

    elif query_type == "3":
        location = input("Enter location: ")
        property_type = input("Enter property type: ")

        response = handle_user_query(
            query_type="amenities",
            location=location,
            property_type=property_type
        )

    elif query_type == "4":
        address = input("Enter property address: ")

        response = handle_user_query(
            query_type="details",
            address=address
        )

    else:
        print("Invalid choice. Please try again.")
        continue

    # Display the response
    print("\nAI Response:", response)

    # Save the conversation to memory
    memory.save_context({"input": query_type}, {"output": response})
