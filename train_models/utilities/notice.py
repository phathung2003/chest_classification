from utilities.config import MIN_RESULT_WIDTH, MIN_TITLE_WIDTH, MIN_MESSAGE_WIDTH, MIN_NUMBER_WIDTH

def print_title(title):
    print(title)
    print("-" * 100)
    
def print_result(RESULT, TITLE, MESSAGE):
    print(f"{RESULT:<{MIN_RESULT_WIDTH}} {TITLE:<{MIN_TITLE_WIDTH}}: {MESSAGE}")

def print_message(TITLE, MESSAGE):
    print(f"{TITLE:<{MIN_TITLE_WIDTH}}: {MESSAGE}")