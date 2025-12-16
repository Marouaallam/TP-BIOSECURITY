import socket
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
import random
import math
import json
import os
import bcrypt
from datetime import datetime
import cv2
import face_recognition
import pickle
import numpy as np

# -------------------------
# Face Recognition Functions
# -------------------------
def capture_face_encoding(max_tries=5, window_title="Face Capture"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Cannot access camera")
        return None

    encodings_collected = []
    tries = 0

    while tries < max_tries:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)

        for (top, right, bottom, left) in faces:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        text = f"Capture {len(encodings_collected)+1}/{max_tries} - Press SPACE"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_title, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if encodings:
                encodings_collected.append(encodings[0])
                tries += 1
                print(f"[✓] Capture {tries}/{max_tries}")
            else:
                print("[!] No face detected")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if encodings_collected:
        return np.mean(encodings_collected, axis=0)

    return None


def compare_faces(encoding1, encoding2, threshold=0.65):
    """Compare two face encodings and return match status and similarity"""
    if encoding1 is None or encoding2 is None:
        return False, 0.0
    
    distance = face_recognition.face_distance([encoding1], encoding2)[0]
    similarity = 1 - distance
    
    return similarity >= threshold, similarity

# -------------------------
# User Database Manager
# -------------------------
class UserDatabase:
    def __init__(self, db_file="users.json", faces_dir="faces"):
        self.db_file = db_file
        self.faces_dir = faces_dir
        os.makedirs(self.faces_dir, exist_ok=True)
        self.load_database()
    
    def load_database(self):
        """Load users from JSON file"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    self.users = json.load(f)
            except Exception as e:
                print(f"Error loading database: {e}")
                self.users = {}
        else:
            self.users = {}
    
    def save_database(self):
        """Save users to JSON file"""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.users, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def hash_password(self, password):
        """Hash a password using bcrypt with salt"""
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password, hashed_password):
        """Verify a password against its hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), 
                                hashed_password.encode('utf-8'))
        except Exception:
            return False
    
    def save_face_encoding(self, username, encoding):
        """Save face encoding for a user"""
        try:
            user_face_dir = os.path.join(self.faces_dir, username)
            os.makedirs(user_face_dir, exist_ok=True)
            face_file = os.path.join(user_face_dir, "face_template.npy")
            np.save(face_file, encoding)
            print(f"[💾] Face encoding saved for {username}")
            return True
        except Exception as e:
            print(f"[!] Error saving face encoding: {e}")
            return False
    
    def load_face_encoding(self, username):
        """Load face encoding for a user"""
        try:
            face_file = os.path.join(self.faces_dir, username, "face_template.npy")
            if os.path.exists(face_file):
                encoding = np.load(face_file)
                print(f"[✓] Face encoding loaded for {username}")
                return encoding
            return None
        except Exception as e:
            print(f"[!] Error loading face encoding: {e}")
            return None
    
    def register_user(self, username, password, face_encoding):
        """Register a new user with password and face"""
        if not username or not password:
            return False, "Username and password cannot be empty"
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters long"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        
        if username in self.users:
            return False, "Username already exists"
        
        if face_encoding is None:
            return False, "Face capture is required for registration"
        
        # Save face encoding
        if not self.save_face_encoding(username, face_encoding):
            return False, "Failed to save face encoding"
        
        # Create user account
        self.users[username] = {
            "password_hash": self.hash_password(password),
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "has_face": True
        }
        
        if self.save_database():
            return True, "Registration successful with face authentication"
        return False, "Error saving user data"
    
    def authenticate_user(self, username, password):
        """Authenticate a user with username and password"""
        if username not in self.users:
            return False, "Invalid username or password"
        
        user_data = self.users[username]
        if self.verify_password(password, user_data["password_hash"]):
            user_data["last_login"] = datetime.now().isoformat()
            self.save_database()
            return True, "Login successful"
        
        return False, "Invalid username or password"
    
    def authenticate_user_by_face(self, username, live_encoding, threshold=0.65):
        """Authenticate a user using face recognition"""
        if username not in self.users:
            return False, "User not found", 0.0
        
        if not self.users[username].get("has_face", False):
            return False, "No face template registered for this user", 0.0
        
        stored_encoding = self.load_face_encoding(username)
        if stored_encoding is None:
            return False, "Could not load face template", 0.0
        
        match, similarity = compare_faces(stored_encoding, live_encoding, threshold)
        
        if match:
            self.users[username]["last_login"] = datetime.now().isoformat()
            self.save_database()
            return True, f"Face authentication successful (similarity: {similarity:.1%})", similarity
        
        return False, f"Face not recognized (similarity: {similarity:.1%})", similarity
    
    def reset_password_with_face(self, username, new_password, live_encoding, threshold=0.65):
        """Reset password after face verification"""
        if username not in self.users:
            return False, "User not found"
        
        if not self.users[username].get("has_face", False):
            return False, "No face template registered. Cannot reset password."
        
        # Verify face
        stored_encoding = self.load_face_encoding(username)
        if stored_encoding is None:
            return False, "Could not load face template"
        
        match, similarity = compare_faces(stored_encoding, live_encoding, threshold)
        
        if not match:
            return False, f"Face verification failed (similarity: {similarity:.1%})"
        
        # Verify new password length
        if len(new_password) < 6:
            return False, "New password must be at least 6 characters long"
        
        # Update password
        self.users[username]["password_hash"] = self.hash_password(new_password)
        
        if self.save_database():
            return True, f"Password reset successful (face similarity: {similarity:.1%})"
        return False, "Error saving new password"
    
    def change_password(self, username, old_password, new_password):
        """Change user password"""
        if username not in self.users:
            return False, "User not found"
        
        if not self.verify_password(old_password, self.users[username]["password_hash"]):
            return False, "Current password is incorrect"
        
        if len(new_password) < 6:
            return False, "New password must be at least 6 characters long"
        
        self.users[username]["password_hash"] = self.hash_password(new_password)
        
        if self.save_database():
            return True, "Password changed successfully"
        return False, "Error saving new password"
    
    def get_user_count(self):
        """Get total number of registered users"""
        return len(self.users)

# -------------------------
# Caesar Cipher
# -------------------------
def caesar_encrypt(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            offset = 65 if char.isupper() else 97
            result += chr((ord(char) - offset + shift) % 26 + offset)
        else:
            result += char
    return result

def caesar_decrypt(text, shift):
    return caesar_encrypt(text, -shift)

# -------------------------
# Vigenère Cipher
# -------------------------
def vigenere_encrypt(text, key):
    result = ""
    key = key.lower()
    key_index = 0
    for char in text:
        if char.isalpha():
            offset = 65 if char.isupper() else 97
            shift = ord(key[key_index % len(key)]) - 97
            result += chr((ord(char) - offset + shift) % 26 + offset)
            key_index += 1
        else:
            result += char
    return result

def vigenere_decrypt(text, key):
    result = ""
    key = key.lower()
    key_index = 0
    for char in text:
        if char.isalpha():
            offset = 65 if char.isupper() else 97
            shift = ord(key[key_index % len(key)]) - 97
            result += chr((ord(char) - offset - shift) % 26 + offset)
            key_index += 1
        else:
            result += char
    return result

# -------------------------
# Substitution Cipher
# -------------------------
def substitution_encrypt(text):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    key = "qwertyuiopasdfghjklzxcvbnm"
    result = ""
    for char in text.lower():
        if char in alphabet:
            index = alphabet.index(char)
            result += key[index]
        else:
            result += char
    return result

def substitution_decrypt(text):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    key = "qwertyuiopasdfghjklzxcvbnm"
    result = ""
    for char in text.lower():
        if char in key:
            index = key.index(char)
            result += alphabet[index]
        else:
            result += char
    return result

# -------------------------
# Transposition Cipher
# -------------------------
def transposition_encrypt(text, key):
    pad_char = 'x'
    clean = text.replace(" ", "")
    rows = (len(clean) + key - 1) // key
    table = []
    idx = 0
    for _ in range(rows):
        row = []
        for _ in range(key):
            if idx < len(clean):
                row.append(clean[idx])
            else:
                row.append(pad_char)
            idx += 1
        table.append(row)
    ciphertext = "".join(table[r][c] for c in range(key) for r in range(rows))
    return ciphertext, len(clean)

def transposition_decrypt(ciphertext, key, original_length=None):
    pad_char = 'x'
    rows = len(ciphertext) // key
    table = [[""] * key for _ in range(rows)]
    idx = 0
    for c in range(key):
        for r in range(rows):
            table[r][c] = ciphertext[idx]
            idx += 1
    plaintext = "".join("".join(row) for row in table)
    if original_length is not None:
        return plaintext[:original_length]
    return plaintext.rstrip(pad_char)

# -------------------------
# RSA Implementation
# -------------------------
def egcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % m

def is_probable_prime(n, k=8):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True

def generate_prime(bits):
    assert bits >= 8
    while True:
        p = random.getrandbits(bits) | (1 << bits - 1) | 1
        if is_probable_prime(p):
            return p

class RSAKey:
    def __init__(self, n, e, d=None):
        self.n = n
        self.e = e
        self.d = d

    @classmethod
    def generate(cls, bits=512):
        half = bits // 2
        p = generate_prime(half)
        q = generate_prime(bits - half)
        while q == p:
            q = generate_prime(bits - half)
        n = p * q
        phi = (p - 1) * (q - 1)
        e = 65537
        if math.gcd(e, phi) != 1:
            for cand in range(3, 65537, 2):
                if math.gcd(cand, phi) == 1:
                    e = cand
                    break
        d = modinv(e, phi)
        return cls(n, e, d)

    def public_tuple(self):
        return (self.n, self.e)

    def encrypt_bytes(self, bdata):
        k = (self.n.bit_length() - 1) // 8
        if k <= 0:
            raise ValueError("Key too small")
        out = []
        for i in range(0, len(bdata), k):
            chunk = bdata[i:i+k]
            m = int.from_bytes(chunk, byteorder='big')
            if m >= self.n:
                raise ValueError("Plaintext chunk too large for modulus")
            c = pow(m, self.e, self.n)
            out.append(c)
        return out

    def decrypt_bytes(self, c_chunks):
        if self.d is None:
            raise ValueError("No private exponent")
        k = (self.n.bit_length() - 1) // 8
        out = bytearray()
        for c in c_chunks:
            m = pow(c, self.d, self.n)
            chunk_bytes = m.to_bytes(k, byteorder='big')
            out.extend(chunk_bytes)
        return bytes(out).rstrip(b'\x00')

    def encrypt_text(self, text):
        b = text.encode('utf-8')
        chunks = self.encrypt_bytes(b)
        return "-".join(format(c, 'x') for c in chunks)

    def decrypt_text(self, hex_str):
        if not hex_str:
            return ""
        chunks = [int(x, 16) for x in hex_str.split("-")]
        b = self.decrypt_bytes(chunks)
        return b.decode('utf-8', errors='ignore')

# -------------------------
# Login Window
# -------------------------
class LoginWindow:
    def __init__(self, root, on_login_success):
        self.root = root
        self.on_login_success = on_login_success
        self.db = UserDatabase()
        self.username = None
        
        self.setup_ui()
    
    def setup_ui(self):
        self.root.title("🔐 Secure Chat Login")
        self.root.geometry("450x550")
        self.root.configure(bg="#2c3e50")
        
        # Title
        title = tk.Label(self.root, text="Secure Chat", 
                        font=("Segoe UI", 24, "bold"),
                        bg="#2c3e50", fg="#ecf0f1")
        title.pack(pady=20)
        
        subtitle = tk.Label(self.root, text="End-to-End Encrypted Messaging with Face Authentication", 
                           font=("Segoe UI", 9),
                           bg="#2c3e50", fg="#95a5a6")
        subtitle.pack()
        
        # Login Frame
        login_frame = tk.Frame(self.root, bg="#34495e", relief="flat")
        login_frame.pack(pady=20, padx=40, fill="both", expand=True)
        
        # Username
        tk.Label(login_frame, text="Username", bg="#34495e", fg="#ecf0f1",
                font=("Segoe UI", 11)).pack(pady=(20, 5))
        self.username_entry = tk.Entry(login_frame, font=("Segoe UI", 11), 
                                       width=25)
        self.username_entry.pack(pady=5)
        
        # Password
        tk.Label(login_frame, text="Password", bg="#34495e", fg="#ecf0f1",
                font=("Segoe UI", 11)).pack(pady=(15, 5))
        self.password_entry = tk.Entry(login_frame, show="●", 
                                       font=("Segoe UI", 11), width=25)
        self.password_entry.pack(pady=5)
        
        # Buttons
        btn_frame = tk.Frame(login_frame, bg="#34495e")
        btn_frame.pack(pady=15)
        
        login_btn = tk.Button(btn_frame, text="Login", command=self.login,
                             bg="#27ae60", fg="white", 
                             font=("Segoe UI", 10, "bold"),
                             width=10, relief="flat", cursor="hand2")
        login_btn.grid(row=0, column=0, padx=5)
        
        face_login_btn = tk.Button(btn_frame, text="👤 Face Login", 
                                   command=self.face_login,
                                   bg="#9b59b6", fg="white", 
                                   font=("Segoe UI", 10, "bold"),
                                   width=12, relief="flat", cursor="hand2")
        face_login_btn.grid(row=0, column=1, padx=5)
        
        register_btn = tk.Button(btn_frame, text="Register", 
                                command=self.register,
                                bg="#3498db", fg="white", 
                                font=("Segoe UI", 10, "bold"),
                                width=10, relief="flat", cursor="hand2")
        register_btn.grid(row=1, column=0, padx=5, pady=10)
        
        # Links frame
        links_frame = tk.Frame(login_frame, bg="#34495e")
        links_frame.pack(pady=10)
        
        #change_pw_btn = tk.Button(links_frame, text="Change Password", 
                                # command=self.change_password,
                                # bg="#34495e", fg="#3498db", 
                                # font=("Segoe UI", 9, "underline"),
                                # relief="flat", cursor="hand2", bd=0)
        #change_pw_btn.pack(side=tk.LEFT, padx=10)
        
        forgot_pw_btn = tk.Button(links_frame, text="Forgot Password?", 
                                 command=self.forgot_password,
                                 bg="#34495e", fg="#e74c3c", 
                                 font=("Segoe UI", 9, "underline"),
                                 relief="flat", cursor="hand2", bd=0)
        forgot_pw_btn.pack(side=tk.LEFT, padx=10)
        
        # User count
        user_count = self.db.get_user_count()
        tk.Label(self.root, 
                text=f"👥 {user_count} registered users", 
                bg="#2c3e50", fg="#95a5a6",
                font=("Segoe UI", 9)).pack(pady=10)
        
        # Bind Enter key
        self.username_entry.bind("<Return>", lambda e: self.password_entry.focus())
        self.password_entry.bind("<Return>", lambda e: self.login())
        
        self.username_entry.focus()
    
    def login(self):
        """Login with username and password"""
        username = self.username_entry.get().strip()
        password = self.password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please enter username and password")
            return
        
        success, message = self.db.authenticate_user(username, password)
        
        if success:
            self.username = username
            messagebox.showinfo("Success", f"Welcome back, {username}!")
            self.root.destroy()
            self.on_login_success(username, password)
        else:
            messagebox.showerror("Login Failed", message)
            self.password_entry.delete(0, tk.END)
    
    def face_login(self):
        """Login using only face recognition"""
        username = self.username_entry.get().strip()
        
        if not username:
            messagebox.showerror("Error", "Please enter your username first")
            return
        
        # Check if user exists
        if username not in self.db.users:
            messagebox.showerror("Error", "User not found")
            return
        
        # Check if user has face registered
        if not self.db.users[username].get("has_face", False):
            messagebox.showerror("Error", "No face template registered for this account")
            return
        
        # Show consent
        consent = messagebox.askyesno(
            "Face Login",
            f"Login as '{username}' using face recognition?\n\n"
            "Your camera will capture your face for authentication.\n"
            "Press SPACE when ready."
        )
        
        if not consent:
            return
        
        # Capture face
        live_encoding = capture_face_encoding(max_tries=3, window_title="Face Login - Press SPACE")
        
        if live_encoding is None:
            messagebox.showerror("Error", "Face capture failed")
            return
        
        # Authenticate
        success, message, similarity = self.db.authenticate_user_by_face(username, live_encoding, threshold=0.65)
        
        if success:
            messagebox.showinfo("Success", f"Face login successful!\n\n{message}")
            self.username = username
            self.root.destroy()
            # Use special password marker for face login
            self.on_login_success(username, "FACE_AUTH")
        else:
            messagebox.showerror("Face Login Failed", message)
            
    def verify_face_only(self, username, live_encoding, threshold=0.65):
        if username not in self.users:
            return False, "User not found", 0.0

        if not self.users[username].get("has_face", False):
            return False, "No face registered for this account", 0.0

        stored_encoding = self.load_face_encoding(username)
        if stored_encoding is None:
            return False, "Could not load face template", 0.0

        match, similarity = compare_faces(stored_encoding, live_encoding, threshold)

        if match:
            return True, "Face verified", similarity

        return False, "Face verification failed", similarity

    def register(self):
        """Register new user with password and MANDATORY face"""
        username = self.username_entry.get().strip()
        password = self.password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please enter username and password")
            return
        
        confirm = simpledialog.askstring("Confirm Password", 
                                        "Re-enter password:", 
                                        show="●")
        
        if confirm != password:
            messagebox.showerror("Error", "Passwords do not match")
            return
        
        # Show MANDATORY consent for face capture
        consent = messagebox.askyesno(
            "⚠️ Face Registration REQUIRED",
            "FACE AUTHENTICATION IS MANDATORY FOR REGISTRATION\n\n"
            "Your face will be captured and stored securely for:\n"
            "• Face-only login option\n"
            "• Secure password recovery\n\n"
            "You have 5 attempts to capture a clear image.\n"
            "Press SPACE when your face is centered.\n\n"
            "Without face registration, account creation is NOT possible.\n\n"
            "Do you consent to face capture?"
        )
        
        if not consent:
            messagebox.showwarning("Registration Cancelled", 
                                  "Face capture is REQUIRED for registration.\n"
                                  "You cannot create an account without it.")
            return
        
        # Capture face with 5 attempts
        messagebox.showinfo("Face Capture", 
                          "Starting face capture...\n\n"
                          "• Ensure good lighting\n"
                          "• Look directly at camera\n"
                          "• Remove glasses if possible\n"
                          "• Press SPACE when ready\n"
                          "• 5 attempts available")
        
        face_encoding = capture_face_encoding(max_tries=5, window_title="Registration - Press SPACE to capture")
        
        if face_encoding is None:
            messagebox.showerror("Registration Failed", 
                               "Face capture failed after 5 attempts.\n"
                               "Registration cancelled.\n\n"
                               "Tips for better capture:\n"
                               "• Ensure good lighting\n"
                               "• Face the camera directly\n"
                               "• Remove glasses\n"
                               "• Hold still when pressing SPACE")
            return
        
        # Register user
        success, message = self.db.register_user(username, password, face_encoding)
        
        if success:
            messagebox.showinfo("✓ Registration Successful", 
                              f"Account created successfully!\n\n"
                              f"Username: {username}\n"
                              f"Face template: Stored ✓\n\n"
                              f"You can now login with:\n"
                              f"1. Username + Password\n"
                              f"2. Username + Face Recognition (65% threshold)")
            self.password_entry.delete(0, tk.END)
            self.username_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Registration Failed", message)
    
    def forgot_password(self):
        """Reset password using face verification (same logic as face login)"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Password Recovery")
        dialog.geometry("400x220")
        dialog.configure(bg="#34495e")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog,
            text="🔄 Password Recovery",
            font=("Segoe UI", 14, "bold"),
            bg="#34495e",
            fg="#ecf0f1"
        ).pack(pady=20)

        tk.Label(
            dialog,
            text="Enter your username:",
            bg="#34495e",
            fg="#ecf0f1",
            font=("Segoe UI", 10)
        ).pack(pady=5)

        user_entry = tk.Entry(dialog, font=("Segoe UI", 10), width=25)
        user_entry.pack(pady=5)

        def do_reset():
            username = user_entry.get().strip()

            if not username:
                messagebox.showerror("Error", "Please enter username")
                return

            # 1️⃣ Check user exists
            if username not in self.db.users:
                messagebox.showerror("Error", "User not found")
                return

            # 2️⃣ Check face registered
            if not self.db.users[username].get("has_face", False):
                messagebox.showerror(
                    "Error",
                    "No face template registered for this account.\n"
                    "Password recovery is not possible."
                )
                return

            # 3️⃣ Consent
            consent = messagebox.askyesno(
                "Face Verification",
                f"Verify identity for '{username}' using face recognition?\n\n"
                "Your camera will capture your face.\n"
                "Minimum 65% similarity required.\n"
                "You have 5 attempts.\n\n"
                "Press YES to continue."
            )

            if not consent:
                return

            # 4️⃣ Capture face (SAME AS FACE LOGIN)
            live_encoding = capture_face_encoding(
                max_tries=3,
                window_title="Face Verification - Password Recovery"
            )

            if live_encoding is None:
                messagebox.showerror("Error", "Face capture failed")
                return

            # 5️⃣ Authenticate using EXISTING FACE LOGIN LOGIC
            success, message, similarity = self.db.authenticate_user_by_face(
                username,
                live_encoding,
                threshold=0.65
            )

            if not success:
                messagebox.showerror(
                    "Access Denied",
                    f"{message}\nSimilarity: {similarity:.1%}"
                )
                return

            # ✅ IDENTITY VERIFIED — NOW allow password change
            new_pw = simpledialog.askstring(
                "New Password",
                "Identity verified ✅\n\nEnter new password (min 6 characters):",
                show="●"
            )

            if not new_pw or len(new_pw) < 6:
                messagebox.showerror(
                    "Error",
                    "Password must be at least 6 characters long"
                )
                return

            confirm_pw = simpledialog.askstring(
                "Confirm Password",
                "Re-enter new password:",
                show="●"
            )

            if new_pw != confirm_pw:
                messagebox.showerror("Error", "Passwords do not match")
                return

            # 6️⃣ Change password (NO FACE LOGIC HERE)
            self.db.users[username]["password_hash"] = self.db.hash_password(new_pw)
            self.db.save_database()

            messagebox.showinfo(
                "Success",
                f"Password reset successful!\n\n"
                f"Face similarity: {similarity:.1%}\n\n"
                f"You can now login with your new password."
            )

            dialog.destroy()

        tk.Button(
            dialog,
            text="Verify Face & Reset",
            command=do_reset,
            bg="#e74c3c",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            width=20,
            relief="flat"
        ).pack(pady=20)

        
    def change_password(self):
        """Change password with current password verification"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Change Password")
        dialog.geometry("350x300")
        dialog.configure(bg="#34495e")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Change Password", 
                font=("Segoe UI", 14, "bold"),
                bg="#34495e", fg="#ecf0f1").pack(pady=20)
        
        tk.Label(dialog, text="Username", bg="#34495e", fg="#ecf0f1").pack(pady=5)
        user_entry = tk.Entry(dialog, font=("Segoe UI", 10), width=25)
        user_entry.pack(pady=5)
        
        tk.Label(dialog, text="Current Password", bg="#34495e", fg="#ecf0f1").pack(pady=5)
        old_pw_entry = tk.Entry(dialog, show="●", font=("Segoe UI", 10), width=25)
        old_pw_entry.pack(pady=5)
        
        tk.Label(dialog, text="New Password", bg="#34495e", fg="#ecf0f1").pack(pady=5)
        new_pw_entry = tk.Entry(dialog, show="●", font=("Segoe UI", 10), width=25)
        new_pw_entry.pack(pady=5)
        
        def do_change():
            username = user_entry.get().strip()
            old_pw = old_pw_entry.get()
            new_pw = new_pw_entry.get()
            
            if not username or not old_pw or not new_pw:
                messagebox.showerror("Error", "All fields are required")
                return
            
            success, message = self.db.change_password(username, old_pw, new_pw)
            
            if success:
                messagebox.showinfo("Success", message)
                dialog.destroy()
            else:
                messagebox.showerror("Error", message)
        
        tk.Button(dialog, text="Change Password", command=do_change,
                 bg="#27ae60", fg="white", font=("Segoe UI", 10, "bold"),
                 width=15, relief="flat").pack(pady=20)
#
# -------------------------
# Client
# -------------------------
class ChatClient:
    def __init__(self, gui, username, password):
        self.gui = gui
        self.username = username
        self.password = password
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Load or generate RSA key pair for this client
        self.rsa_key = self.load_or_generate_rsa_key(username)
        
        # Dictionary to store public keys: {username: (n, e)}
        self.peer_rsa_public_keys = {}
        
        try:
            self.socket.connect(('localhost', 5555))
            
            # Authenticate with server
            if password == "FACE_AUTH":
                auth_msg = f"AUTH|{username}|FACE_AUTH"
            else:
                auth_msg = f"AUTH|{username}|{password}"

            self.socket.send(auth_msg.encode())
            
            response = self.socket.recv(1024).decode()
            if response == "AUTH_SUCCESS":
                self.gui.update_status(f"🟢 Connected as {username}", "green")
                self.gui.display_message(f"✅ Authenticated with server", "green")
            elif response.startswith("AUTH_FAILED"):
                reason = response.split("|", 1)[1] if "|" in response else "Unknown"
                self.gui.update_status("🔴 Authentication Failed", "red")
                messagebox.showerror("Auth Error", f"Server authentication failed: {reason}")
                self.socket.close()
                self.gui.root.quit()
                return
            
            # Send our public key
            self.exchange_keys()
            
        except Exception as e:
            self.gui.update_status("🔴 Disconnected", "red")
            messagebox.showerror("Connection Error", f"Cannot connect to server: {e}")
            self.gui.root.quit()
            return
        
        threading.Thread(target=self.receive_messages, daemon=True).start()

    def load_or_generate_rsa_key(self, username):
        """Load existing RSA key or generate new one if doesn't exist"""
        keys_dir = "client_rsa_keys"
        os.makedirs(keys_dir, exist_ok=True)
        
        key_file = os.path.join(keys_dir, f"{username}_rsa_key.pkl")
        
        # Try to load existing key
        if os.path.exists(key_file):
            try:
                with open(key_file, 'rb') as f:
                    key_data = pickle.load(f)
                    rsa_key = RSAKey(key_data['n'], key_data['e'], key_data['d'])
                    print(f"[🔑] Loaded existing RSA key for {username}")
                    self.gui.display_message("🔑 Using existing RSA key pair", "blue")
                    return rsa_key
            except Exception as e:
                print(f"[!] Error loading RSA key: {e}")
                print(f"[🔄] Generating new RSA key...")
        
        # Generate new key if doesn't exist or loading failed
        print(f"[🔑] Generating new RSA key for {username}...")
        self.gui.display_message("🔑 Generating RSA key pair...", "blue")
        rsa_key = RSAKey.generate(512)
        
        # Save the key
        try:
            key_data = {
                'n': rsa_key.n,
                'e': rsa_key.e,
                'd': rsa_key.d
            }
            with open(key_file, 'wb') as f:
                pickle.dump(key_data, f)
            print(f"[💾] Saved RSA key for {username}")
            self.gui.display_message("✅ RSA key pair saved for future use", "green")
        except Exception as e:
            print(f"[!] Error saving RSA key: {e}")
            self.gui.display_message(f"⚠️ Warning: Could not save RSA key", "orange")
        
        return rsa_key

    def exchange_keys(self):
        """Exchange RSA public keys with all peers"""
        try:
            n, e = self.rsa_key.public_tuple()
            # Format: RSA_KEY|username|n|e
            key_msg = f"RSA_KEY|{self.username}|{n}|{e}"
            self.socket.send(key_msg.encode())
            self.gui.display_message("🔐 RSA public key sent to server", "black")
        except Exception as e:
            self.gui.display_message(f"⚠️ Key exchange error: {e}", "red")
    
    def update_recipient_list(self):
        """Update the recipient dropdown with users who have public keys"""
        available_users = sorted(list(self.peer_rsa_public_keys.keys()))
        self.gui.update_online_users(available_users)
        if available_users:
            self.gui.display_message(f"✓ Can send RSA messages to: {', '.join(available_users)}", "green")

    def receive_messages(self):
        buffer = ""
        while True:
            try:
                data = self.socket.recv(8192)
                if not data:
                    self.gui.update_status("🔴 Disconnected", "red")
                    break

                buffer += data.decode()
                messages_to_process = []
                
                if '\n' in buffer:
                    lines = buffer.split('\n')
                    buffer = lines[-1]
                    messages_to_process = [line for line in lines[:-1] if line]
                else:
                    if buffer.startswith(('SYSTEM|', 'ONLINE_USERS|', 'RSA_KEY|', 'ALL_KEYS|', 'RSA_MSG|')):
                        if buffer.startswith('RSA_MSG|'):
                            messages_to_process = [buffer]
                            buffer = ""
                        elif not buffer.endswith('|'):
                            messages_to_process = [buffer]
                            buffer = ""
                    else:
                        messages_to_process = [buffer]
                        buffer = ""
                
                for encrypted_message in messages_to_process:
                    if not encrypted_message:
                        continue
                    self.process_message(encrypted_message)

            except Exception as e:
                self.gui.update_status("🔴 Disconnected", "red")
                print(f"Receive error: {e}")
                break
    
    def process_message(self, encrypted_message):
        sender = None
        cipher_used = None
        payload = encrypted_message

        # 🔐 Message privé non-RSA
        if encrypted_message.startswith("MSG|"):
            _, cipher_used, sender, recipient, payload = encrypted_message.split("|", 4)

            # ⛔ Ignorer si ce n'est pas pour moi
            if recipient != self.username:
                return

        try:
            # SYSTEM
            if encrypted_message.startswith("SYSTEM|"):
                msg = encrypted_message.split("|", 1)[1]
                self.gui.display_message(f"📢 {msg}", "purple")
                return

            # ONLINE USERS
            if encrypted_message.startswith("ONLINE_USERS|"):
                users = encrypted_message.split("|", 1)[1]
                user_list = [u for u in users.split(",") if u and u != self.username]
                self.gui.update_online_users(user_list)
                return

            if encrypted_message.startswith("ALL_KEYS|"):
                keys_data = encrypted_message.split("|", 1)[1]
                if keys_data:
                    for entry in keys_data.split(","):
                        parts = entry.split(":")
                        if len(parts) == 3:
                            username, n, e = parts
                            self.peer_rsa_public_keys[username] = (int(n), int(e))
                    self.update_recipient_list()
                    self.gui.display_message(
                        f"🔐 Received {len(self.peer_rsa_public_keys)} existing public key(s)", 
                        "blue"
                    )
                return

            # RSA MESSAGE
            if encrypted_message.startswith("RSA_MSG|"):
                _, target, sender_name, encrypted_data = encrypted_message.split("|", 3)

                if target == self.username:
                    decrypted = self.rsa_key.decrypt_text(encrypted_data)
                    self.gui.display_message(
                        f"🔒 {sender_name} → You: {decrypted}",
                        "green"
                    )
                return

            cipher = cipher_used
            key = self.gui.key_var.get()

            # Transposition
            if payload.startswith("TRANS|"):
                _, ct, orig_len = payload.split("|", 2)
                payload = ct
                orig_len = int(orig_len)
            else:
                orig_len = None

            # Déchiffrement
            if cipher == "Caesar":
                decrypted = caesar_decrypt(payload, int(key))

            elif cipher == "Vigenere":
                decrypted = vigenere_decrypt(payload, key)

            elif cipher == "Substitution":
                decrypted = substitution_decrypt(payload)

            elif cipher == "Transposition":
                decrypted = transposition_decrypt(payload, int(key), orig_len)

            else:
                decrypted = "[Unsupported cipher]"

            self.gui.display_message(
                f"🔓 {sender} → You: {decrypted}",
                "green"
            )

        except Exception as e:
            self.gui.display_message(
                f"⚠️ Message processing error: {e}",
                "red"
            )


    def send_message(self, message):
        cipher = self.gui.cipher_choice.get()
        key = self.gui.key_var.get()

        # ✅ TOUJOURS récupérer le destinataire
        recipient = self.gui.get_selected_recipient()
        if not recipient:
            messagebox.showerror("Error", "Veuillez sélectionner un destinataire")
            return

        # RSA (déjà privé)
        if cipher == "RSA":
            try:
                if recipient not in self.peer_rsa_public_keys:
                    messagebox.showerror("Error", f"No RSA public key for {recipient}")
                    return

                # CRÉER UNE CLÉ RSA TEMPORAIRE AVEC LA CLÉ PUBLIQUE DU DESTINATAIRE
                recipient_pub_key = self.peer_rsa_public_keys[recipient]
                temp_rsa = RSAKey(n=recipient_pub_key[0], e=recipient_pub_key[1])
                
                # CHIFFRER AVEC LA CLÉ PUBLIQUE DU DESTINATAIRE
                encrypted = temp_rsa.encrypt_text(message)

                full_msg = f"RSA_MSG|{recipient}|{self.username}|{encrypted}"
                self.socket.send(full_msg.encode())

                self.gui.display_message(
                    f"🔐 You → {recipient}: {message}",
                    "lightblue"
                )
                return
            except Exception as e:
                messagebox.showerror("RSA Error", str(e))
                return
            
        # 🔐 Chiffrements classiques
        if cipher == "Caesar":
            payload = caesar_encrypt(message, int(key))

        elif cipher == "Vigenere":
            payload = vigenere_encrypt(message, key)

        elif cipher == "Substitution":
            payload = substitution_encrypt(message)

        elif cipher == "Transposition":
            payload, orig_len = transposition_encrypt(message, int(key))
            payload = f"TRANS|{payload}|{orig_len}"

        else:
            messagebox.showerror("Error", "Unsupported cipher")
            return

        # ✅ Message privé avec destinataire
        full_msg = f"MSG|{cipher}|{self.username}|{recipient}|{payload}"
        self.socket.send(full_msg.encode())

        # Affichage local
        self.gui.display_message(
            f"🔐 You → {recipient}: {message}",
            "lightgreen"
        )


# -------------------------
# GUI
# -------------------------
class ChatGUI:
    def __init__(self, root, username, password):
        self.root = root
        self.username = username
        self.password = password

        self.root.title(f"🔒 Secure Chat - {username}")
        self.root.geometry("700x600")
        self.root.configure(bg="#1e1e1e")

        top_frame = tk.Frame(root, bg="#1e1e1e")
        top_frame.pack(pady=10)

        tk.Label(top_frame, text="Cipher:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT, padx=5)

        self.cipher_choice = ttk.Combobox(
            top_frame,
            values=["Caesar", "Vigenere", "Substitution", "Transposition", "RSA"],
            width=15
        )
        self.cipher_choice.current(0)
        self.cipher_choice.pack(side=tk.LEFT)

        tk.Label(top_frame, text="Key:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT, padx=5)

        self.key_var = tk.StringVar(value="3")
        tk.Entry(top_frame, textvariable=self.key_var, width=10).pack(side=tk.LEFT)

        self.chat_box = scrolledtext.ScrolledText(
            root, bg="#2b2b2b", fg="white", width=80, height=25
        )
        self.chat_box.pack(padx=10, pady=10)
        self.chat_box.config(state=tk.DISABLED)

        bottom_frame = tk.Frame(root, bg="#1e1e1e")
        bottom_frame.pack(pady=10)

        self.message_entry = tk.Entry(bottom_frame, width=50)
        self.message_entry.pack(side=tk.LEFT, padx=5)
        self.message_entry.bind("<Return>", lambda e: self.send_message())

        tk.Button(bottom_frame, text="Send", command=self.send_message).pack(side=tk.LEFT)

        self.recipient_var = tk.StringVar()
        self.recipient_menu = ttk.Combobox(bottom_frame, textvariable=self.recipient_var, width=15)
        self.recipient_menu.pack(side=tk.LEFT, padx=5)

        self.client = ChatClient(self, username, password)

    def display_message(self, message, color="white"):
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.insert(tk.END, message + "\n")
        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.see(tk.END)

    def update_status(self, text, color):
        self.display_message(text, color)

    def update_online_users(self, users):
        self.recipient_menu["values"] = users

    def get_selected_recipient(self):
        return self.recipient_var.get()

    def send_message(self):
        msg = self.message_entry.get().strip()
        if msg:
            self.client.send_message(msg)
            self.message_entry.delete(0, tk.END)

# -------------------------
# Run App
# -------------------------
def start_chat(username, password):
    chat_root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    app = ChatGUI(chat_root, username, password)
    chat_root.mainloop()

if __name__ == "__main__":
    login_root = tk.Tk()
    login_window = LoginWindow(login_root, start_chat)
    login_root.mainloop()