# server.py
import socket
import threading
import json
import os
import bcrypt

# Store connected clients with their info
# Format: {client_socket: {'username': str, 'address': tuple, 'public_key': {'n': int, 'e': int}}}
clients = {}
clients_lock = threading.Lock()

# Store all public keys: {username: {'n': int, 'e': int}}
public_keys = {}
keys_lock = threading.Lock()

class UserDatabase:
    def __init__(self, db_file="users.json", keys_file="rsa_keys.json"):
        self.db_file = db_file
        self.keys_file = keys_file
        self.lock = threading.Lock()
        self.load_database()
        self.load_rsa_keys()
    
    def load_database(self):
        """Load users from JSON file"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    self.users = json.load(f)
                print(f"[📂] Loaded {len(self.users)} users from database")
            except Exception as e:
                print(f"[!] Error loading database: {e}")
                self.users = {}
        else:
            print("[📂] No existing database found, starting fresh")
            self.users = {}
    
    def load_rsa_keys(self):
        """Load RSA public keys from JSON file"""
        global public_keys
        if os.path.exists(self.keys_file):
            try:
                with open(self.keys_file, 'r') as f:
                    loaded_keys = json.load(f)
                    with keys_lock:
                        public_keys = loaded_keys
                    print(f"[🔑] Loaded {len(public_keys)} RSA public keys from storage")
            except Exception as e:
                print(f"[!] Error loading RSA keys: {e}")
                public_keys = {}
        else:
            print("[🔑] No existing RSA keys file found, starting fresh")
            public_keys = {}
    
    def save_rsa_keys(self):
        """Save RSA public keys to JSON file"""
        try:
            with keys_lock:
                keys_to_save = dict(public_keys)
            
            with open(self.keys_file, 'w') as f:
                json.dump(keys_to_save, f, indent=4)
            print(f"[💾] Saved {len(keys_to_save)} RSA public keys to storage")
            return True
        except Exception as e:
            print(f"[!] Error saving RSA keys: {e}")
            return False
    
    def verify_password(self, password, hashed_password):
        """Verify a password against its hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), 
                                hashed_password.encode('utf-8'))
        except Exception as e:
            print(f"[!] Password verification error: {e}")
            return False
    
    def authenticate_user(self, username, password):
        with self.lock:
            self.load_database()  # Reload to get latest data

            if username not in self.users:
                print(f"[✗] Authentication failed: user '{username}' not found")
                return False, "Invalid username or password"

            user_data = self.users[username]

            # Face login accepted (trusted client)
            if password == "FACE_AUTH":
                print(f"[👤] Face authentication accepted for '{username}'")
                return True, "Face login successful"

            # Normal password login
            if self.verify_password(password, user_data["password_hash"]):
                print(f"[✓] User '{username}' authenticated successfully")
                return True, "Login successful"
            
            print(f"[✗] Authentication failed: incorrect password for '{username}'")
            return False, "Invalid username or password"


# Initialize database
db = UserDatabase()

def handle_client(client_socket, addr):
    print(f"\n[+] New connection from {addr}")
    username = None
    
    try:
        # Wait for authentication with timeout
        client_socket.settimeout(10)
        auth_data = client_socket.recv(1024).decode()
        client_socket.settimeout(None)  # Remove timeout after auth
        
        if not auth_data:
            print(f"[!] No data received from {addr}")
            client_socket.close()
            return
        
        if auth_data.startswith("AUTH|"):
            parts = auth_data.split("|")
            if len(parts) >= 3:
                _, username, password = parts[0], parts[1], parts[2]
                
                print(f"[🔐] Authentication attempt for user: '{username}'")
                success, message = db.authenticate_user(username, password)
                
                if success:
                    # Check if user already connected
                    with clients_lock:
                        for client_info in clients.values():
                            if client_info['username'] == username:
                                print(f"[!] User '{username}' already connected")
                                client_socket.send("AUTH_FAILED|User already connected".encode())
                                client_socket.close()
                                return
                    
                    client_socket.send("AUTH_SUCCESS".encode())
                    print(f"[✓] {username} authenticated successfully from {addr}")
                    
                    # Check if user has existing RSA key
                    existing_key = None
                    with keys_lock:
                        if username in public_keys:
                            existing_key = public_keys[username]
                            print(f"[🔑] Found existing RSA key for {username}")
                    
                    # Store client info
                    with clients_lock:
                        clients[client_socket] = {
                            'username': username,
                            'address': addr,
                            'public_key': existing_key
                        }
                    
                    # Send existing keys FIRST before announcing new user
                    send_all_keys_to_client(client_socket)
                    
                    # Small delay to ensure message separation
                    import time
                    time.sleep(0.1)
                    
                    # Now send welcome message
                    send_system_message(f"{username} joined the chat!", client_socket)
                    
                    # Small delay to ensure message separation
                    time.sleep(0.1)
                    
                    # Send list of online users to everyone
                    broadcast_online_users()
                    
                else:
                    print(f"[✗] Authentication failed for '{username}': {message}")
                    client_socket.send(f"AUTH_FAILED|{message}".encode())
                    client_socket.close()
                    return
            else:
                print(f"[!] Invalid authentication format from {addr}")
                client_socket.send("AUTH_FAILED|Invalid authentication format".encode())
                client_socket.close()
                return
        else:
            print(f"[!] Missing AUTH prefix from {addr}")
            client_socket.send("AUTH_FAILED|Authentication required".encode())
            client_socket.close()
            return
        
        # Main message loop
        while True:
            data = client_socket.recv(8192)
            if not data:
                print(f"[!] No data from {username}, closing connection")
                break
            
            message = data.decode()
            
            # Handle RSA key exchange
            if message.startswith("RSA_KEY|"):
                try:
                    parts = message.split("|")
                    if len(parts) >= 4:
                        _, key_username, n_str, e_str = parts[0], parts[1], parts[2], parts[3]
                        
                        # Store the public key in memory AND persist to disk
                        with keys_lock:
                            public_keys[key_username] = {
                                'n': int(n_str),
                                'e': int(e_str)
                            }
                        
                        # Save to disk immediately
                        db.save_rsa_keys()
                        
                        # Update client info
                        with clients_lock:
                            if client_socket in clients:
                                clients[client_socket]['public_key'] = public_keys[key_username]
                        
                        print(f"[🔐] Received and saved public key from {key_username}")
                        
                        # Broadcast this key to all OTHER clients
                        broadcast(message, client_socket)
                        
                        continue
                except Exception as e:
                    print(f"[!] Error processing RSA key: {e}")
                    continue
            
            # Handle RSA direct messages
            if message.startswith("RSA_MSG|"):
                try:
                    parts = message.split("|", 3)
                    if len(parts) >= 4:
                        _, target_user, sender_user, encrypted_data = parts
                        
                        # Find target client socket
                        target_socket = None
                        with clients_lock:
                            for sock, info in clients.items():
                                if info['username'] == target_user:
                                    target_socket = sock
                                    break
                        
                        if target_socket:
                            # Send only to target
                            try:
                                target_socket.send(message.encode())
                                print(f"[🔒] RSA message from {sender_user} to {target_user}")
                            except Exception as e:
                                print(f"[!] Failed to send RSA message to {target_user}: {e}")
                        else:
                            print(f"[!] Target user {target_user} not found")
                        
                        continue
                except Exception as e:
                    print(f"[!] Error routing RSA message: {e}")
                    continue
            
            # 🔐 Private encrypted messages (non-RSA)
            if message.startswith("MSG|"):
                try:
                    _, cipher, sender, recipient, payload = message.split("|", 4)

                    target_socket = None
                    with clients_lock:
                        for sock, info in clients.items():
                            if info['username'] == recipient:
                                target_socket = sock
                                break

                    if target_socket:
                        target_socket.send(message.encode())
                        print(f"[📩] {cipher} message from {sender} to {recipient}")
                    else:
                        print(f"[!] Recipient '{recipient}' not found")

                    continue

                except Exception as e:
                    print(f"[!] Error routing MSG message: {e}")
                    continue

            
    except socket.timeout:
        print(f"[!] Authentication timeout from {addr}")
        try:
            client_socket.send("AUTH_FAILED|Authentication timeout".encode())
        except:
            pass
    except Exception as e:
        print(f"[!] Error with {addr}: {e}")
    finally:
        if username:
            print(f"[-] {username} disconnected from {addr}")
            send_system_message(f"{username} left the chat", client_socket)
            
            # DON'T remove public key - keep it for next login
            # Only remove from public_keys if explicitly requested
            print(f"[🔑] Keeping RSA key for {username} (will be reused on next login)")
        else:
            print(f"[-] Client {addr} disconnected (not authenticated)")
        
        # Remove client from active connections
        with clients_lock:
            if client_socket in clients:
                del clients[client_socket]
        
        try:
            client_socket.close()
        except:
            pass
        
        # Update online users list
        broadcast_online_users()

def broadcast(message, sender_socket):
    """Send the message to all clients except the sender."""
    failed_clients = []
    
    with clients_lock:
        clients_list = list(clients.items())
    
    for client_socket, client_info in clients_list:
        if client_socket != sender_socket:
            try:
                client_socket.send(message.encode())
            except Exception as e:
                print(f"[!] Failed to send to {client_info['username']}: {e}")
                failed_clients.append(client_socket)
    
    # Clean up failed clients
    if failed_clients:
        with clients_lock:
            for client_socket in failed_clients:
                if client_socket in clients:
                    del clients[client_socket]
                try:
                    client_socket.close()
                except:
                    pass

def send_system_message(message, exclude_socket=None):
    """Send a system message to all clients"""
    sys_msg = f"SYSTEM|{message}"
    
    with clients_lock:
        clients_list = list(clients.keys())
    
    for client_socket in clients_list:
        if client_socket != exclude_socket:
            try:
                client_socket.send(sys_msg.encode())
            except Exception as e:
                print(f"[!] Failed to send system message: {e}")

def broadcast_online_users():
    """Send list of online users to all clients"""
    with clients_lock:
        usernames = [info['username'] for info in clients.values() if info['username']]
        clients_list = list(clients.keys())
    
    users_str = ",".join(usernames)
    online_msg = f"ONLINE_USERS|{users_str}"
    
    print(f"[👥] Broadcasting online users: {users_str if users_str else '(none)'}")
    
    for client_socket in clients_list:
        try:
            client_socket.send(online_msg.encode())
        except Exception as e:
            print(f"[!] Failed to send online users list: {e}")

def send_all_keys_to_client(client_socket):
    """Send all existing public keys to a newly connected client"""
    with keys_lock:
        if not public_keys:
            print(f"[🔑] No existing keys to send to new client")
            return
        
        # Format: ALL_KEYS|username1:n:e,username2:n:e,...
        keys_list = []
        
        with clients_lock:
            client_username = clients.get(client_socket, {}).get('username')
        
        for username, key_info in public_keys.items():
            # Don't send the client their own key
            if username == client_username:
                continue
            keys_list.append(f"{username}:{key_info['n']}:{key_info['e']}")
    
    if keys_list:
        all_keys_msg = "ALL_KEYS|" + ",".join(keys_list)
        try:
            client_socket.send(all_keys_msg.encode())
            print(f"[🔑] Sent {len(keys_list)} existing public key(s) to new client")
        except Exception as e:
            print(f"[!] Error sending keys to client: {e}")

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind(('localhost', 5555))
        server.listen(5)
        print("=" * 60)
        print("🚀 Secure Chat Server Started")
        print("=" * 60)
        print(f"📡 Listening on: localhost:5555")
        print(f"📂 Database file: users.json")
        print(f"🔑 RSA Keys file: rsa_keys.json")
        print("\n📋 Features:")
        print("   ✓ User authentication with bcrypt")
        print("   ✓ Persistent RSA key storage")
        print("   ✓ RSA key distribution")
        print("   ✓ End-to-end encryption support")
        print("   ✓ Multiple cipher support (Caesar, Vigenère, Substitution, Transposition, RSA)")
        print("   ✓ Thread-safe client management")
        print("\n⏳ Waiting for clients...")
        print("=" * 60 + "\n")

        while True:
            client_socket, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(client_socket, addr))
            thread.daemon = True
            thread.start()
            
    except KeyboardInterrupt:
        print("\n\n[!] Server shutting down...")
        db.save_rsa_keys()  # Save keys one last time
        server.close()
        print("[✓] Server closed successfully")
    except Exception as e:
        print(f"\n[!] Server error: {e}")
        server.close()

if __name__ == "__main__":
    start_server()