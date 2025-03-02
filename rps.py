# Filename: rps.py
import random
def rps():
    options = ["rock", "paper", "scissors"]

    while True:
        user_choice = input("Choose rock, paper, or scissors (or type 'quit' to exit): ").lower()

        if user_choice == "quit":
            print("Thanks for playing!")
            break

        if user_choice not in options:
            print("Invalid entry. Please choose between rock, paper, or scissors.")
            continue

        computer_choice = random.choice(options)
        print(f"Computer choice: {computer_choice}")

        if user_choice == computer_choice:
            print("It's a tie!")
        elif    (user_choice == "rock" and computer_choice == "scissors") or \
                (user_choice == "scissors" and computer_choice == "paper") or \
                (user_choice == "paper" and computer_choice == "rock"):
                print("You win!")
        else:
                print("You lose!")
rps()
