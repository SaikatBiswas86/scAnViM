import json
import random
import uuid
import openpyxl  # You may need to install this module using: pip install openpyxl
import os

# Load intents data from intents.json
with open('database/intents_2.json') as file:
    intents = json.load(file)


def handle_review(user_msg):
    if any(keyword in user_msg.lower() for keyword in ["review", "report", "problem"]):
        return "Please provide your statement in the following format: Statement//Rating value (1 to 5)//User contact details"
    else:
        return None

# Add this function to handle the response after saving the review
def handle_review_response():
    return "Thank you for your review. Any other questions or feedback?"

def get_response(user_msg):
    if any(keyword in user_msg.lower() for keyword in ["review", "report", "problem"]):
        # Prompt the user to provide a review
        return "Please provide your review/report/problem in the following format: Statement//Rating value 1 to 5//User contact details"

    elif "//" in user_msg:
        # Save the review to file
        save_response_to_file(user_msg, 'database/reviews.xlsx')
        return "Your statement is recorded.Have other queries?"

    else:
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                if pattern.lower() in user_msg:
                    return random.choice(intent['responses'])

        return "I'm sorry, Ask Something Else."


def save_response_to_file(user_msg, filename):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if the file exists, create it if it doesn't
    if not os.path.isfile(filename):
        wb = openpyxl.Workbook()
        wb.save(filename)

    # Load the workbook and append the data
    wb = openpyxl.load_workbook(filename)
    ws = wb.active
    row = user_msg.split('//')
    ws.append(row)
    wb.save(filename)


def add_new_response_to_json(new_response, new_pattern):
    new_tag = str(uuid.uuid4())

    # Add the new response and intent to intents.json
    intents['intents'].append({
        "tag": new_tag,
        "patterns": [new_pattern],
        "responses": [new_response]
    })

    with open('data/intents.json', 'w') as file:
        json.dump(intents, file, indent=4)

    print("Chatbot: Thanks! I've added your response.")


if __name__ == '__main__':
    print("Chatbot: Hi! How can I assist you today?")

    while True:
        user_msg = input("You: ")

        if user_msg.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        response = get_response(user_msg)

        if response:
            print(f"Chatbot: {response}")
        elif any(keyword in user_msg.lower() for keyword in ["review", "report", "problem"]):
            print("Chatbot: Please provide your review/report/problem in the following format:")
            print("Statement//Rating value (1 to 5)//User contact details")

            user_response = input("You: ")
            save_response_to_file(user_response, 'data/reviews.xlsx')

            print("Chatbot: Your statement is recorded. Need any help?")
        else:
            print(f"Chatbot: I'm sorry, I don't understand that.")
            user_response = input("Chatbot: Do you want to provide an answer? (yes/no) ").lower()
            if user_response == "yes":
                new_response = input("You: What should the response be? ")
                add_new_response_to_json(new_response, user_msg)
            else:
                print("Chatbot: Okay, if you have any other questions, please ask.")
