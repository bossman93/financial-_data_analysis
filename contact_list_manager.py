import csv
import os

FILENAME = "contact_list.csv"

if not os.path.exists(FILENAME):
    with open(FILENAME, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Phone", "Email"])

def add_contact():
    name = input("Enter name: ")
    phone = input("Enter phone number: ")
    email = input("Enter email:")

    with open(FILENAME, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, phone, email])

    print(f"Contact '{name}' added successfully!")

def delete_contact():
    name_to_delete = input("Enter the name of the contact to delete")

    with open(FILENAME, mode="r", newline="") as file:
        rows = list(csv.reader(file))

    updated_rows = [row for row in rows if row[0].lower() != name_to_delete.lower()]

    if len(updated_rows) == len(rows):
        print(f"No contact found with the name '{name_to_delete}'.")
        return

    with open(FILENAME, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(updated_rows)

    print(f"Contact '{name_to_delete}' deleted successfully!")

def search_contact():
    search_name = input("Enter name to search: ").lower()

    with open(FILENAME, mode="r", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        found_contacts = [row for row in reader if search_name in row[0].lower()]

    if found_contacts:
        print("\nSearchResults:")
        for contact in found_contacts:
            print(f"Name:{contact[0]}, Phone: {contact[1]}, Email: {contact[2]}")
    else: print(f"No contacts found with the name '{search_name}'.")

def display_contacts():
    with open(FILENAME, mode="r", newline="") as file:
        reader = csv.reader(file)
        contacts = list(reader)

    if len(contacts) <=1:
        print("No contacts found.")
        return

    print("\nContact List:")
    for i, contact in enumerate(contacts):
        if i == 0:
            print(f"{contact[0]:<20} {contact[1]:<15} {contact[2]:<25}")
            print("-" * 60)
        else:
            print(f"{contact[0]:<20} {contact[1]:<15} {contact[2]:<25}")

def main():
    """"Main menu for the contact manager."""
    while True:
        print("\nContact Manager")
        print("1. Add Contact")
        print("2. Delete Contact")
        print("3. Search Contact")
        print("4. Display Contacts")
        print("5. Exit")

        choice = input("Choose an option: ")

        if choice == "1":
            add_contact()
        elif choice == "2":
            delete_contact()
        elif choice == "3":
            search_contact()
        elif choice == "4":
            display_contacts()
        elif choice == "5":
            print("Exiting... Goodbye!")
            break
        else:
            print("invalid option. Please try again.")
if __name__ == "__main__":
    main()