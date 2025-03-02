import random

def guessing_game():
    secret_number = random.randint(1,100)
    attempts = 0

    print("You are now playing the Guessing Game!")
    print("I have selected a number between 1 and 100")
    print("Now try to guess the number!")

    while True:
        try:
            guess = int(input("Enter your number"))
            attempts += 1

            if guess < secret_number:
                print("Too low! Try again.")
            elif guess > secret_number:
                print("Too high! Try again.")
            else:
                print(f"Congratulations! You guessed the number {secret_number} in {attempts}")
                break
        except ValueError:
            print("Invalid input. Please enter a number.")

#Start the game
guessing_game()