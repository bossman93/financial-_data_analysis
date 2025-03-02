import hashlib

# Read the script file
with open("contact_list_manager.py", "rb") as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()

print("SHA-256 Hash:", file_hash)