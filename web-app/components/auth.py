"""
Authentication Manager for Efficient MedSAM2 Web Application
==========================================================
Handles user registration, login, session management, and security.
"""

import streamlit as st
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
import re
import os

class AuthManager:
    """Manages user authentication and session handling"""
    
    def __init__(self, db_path="users.db"):
        """Initialize authentication manager with database"""
        self.db_path = os.path.join(os.path.dirname(__file__), "..", db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize user database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    role TEXT DEFAULT 'user',
                    full_name TEXT,
                    institution TEXT
                )
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    activity_type TEXT NOT NULL,
                    description TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
                """)
                
                conn.commit()
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        return password_hash.hex(), salt
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_password(self, password: str) -> tuple:
        """Validate password strength"""
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def register_user(self, username: str, email: str, password: str, full_name: str = "", institution: str = "") -> tuple:
        """Register new user"""
        try:
            # Validate inputs
            if not username or len(username) < 3:
                return False, "Username must be at least 3 characters long"
            
            if not self.validate_email(email):
                return False, "Invalid email format"
            
            is_valid_password, password_errors = self.validate_password(password)
            if not is_valid_password:
                return False, "; ".join(password_errors)
            
            # Check if user already exists
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
                if cursor.fetchone():
                    return False, "Username or email already exists"
                
                # Hash password
                password_hash, salt = self.hash_password(password)
                
                # Insert new user
                cursor.execute("""
                INSERT INTO users (username, email, password_hash, salt, full_name, institution)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (username, email, password_hash, salt, full_name, institution))
                
                user_id = cursor.lastrowid
                
                # Log registration activity
                self.log_activity(user_id, "registration", f"User {username} registered", conn)
                
                conn.commit()
                return True, f"Successfully registered user: {username}"
        
        except Exception as e:
            return False, f"Registration error: {str(e)}"
    
    def login_user(self, username_or_email: str, password: str) -> tuple:
        """Authenticate user login"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find user by username or email
                cursor.execute("""
                SELECT id, username, email, password_hash, salt, is_active, full_name
                FROM users 
                WHERE (username = ? OR email = ?) AND is_active = TRUE
                """, (username_or_email, username_or_email))
                
                user_data = cursor.fetchone()
                
                if not user_data:
                    return False, "Invalid credentials", None
                
                user_id, username, email, stored_hash, salt, is_active, full_name = user_data
                
                # Verify password
                password_hash, _ = self.hash_password(password, salt)
                
                if password_hash != stored_hash:
                    self.log_activity(user_id, "login_failed", f"Failed login attempt for {username}", conn)
                    return False, "Invalid credentials", None
                
                # Update last login
                cursor.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                """, (user_id,))
                
                # Create session
                session_token = self.create_session(user_id, conn)
                
                # Log successful login
                self.log_activity(user_id, "login_success", f"Successful login for {username}", conn)
                
                conn.commit()
                
                user_info = {
                    'id': user_id,
                    'username': username,
                    'email': email,
                    'full_name': full_name,
                    'session_token': session_token
                }
                
                return True, "Login successful", user_info
        
        except Exception as e:
            return False, f"Login error: {str(e)}", None
    
    def create_session(self, user_id: int, conn: sqlite3.Connection) -> str:
        """Create new user session"""
        session_token = secrets.token_urlsafe(64)
        expires_at = datetime.now() + timedelta(hours=24)  # 24-hour session
        
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO sessions (user_id, session_token, expires_at)
        VALUES (?, ?, ?)
        """, (user_id, session_token, expires_at))
        
        return session_token
    
    def validate_session(self, session_token: str) -> tuple:
        """Validate active session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT s.user_id, u.username, u.email, u.full_name
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = ? 
                AND s.is_active = TRUE 
                AND s.expires_at > CURRENT_TIMESTAMP
                AND u.is_active = TRUE
                """, (session_token,))
                
                session_data = cursor.fetchone()
                
                if session_data:
                    user_id, username, email, full_name = session_data
                    return True, {
                        'id': user_id,
                        'username': username,
                        'email': email,
                        'full_name': full_name
                    }
                else:
                    return False, None
        
        except Exception as e:
            return False, None
    
    def logout_user(self, session_token: str = None):
        """Logout user and invalidate session"""
        try:
            if session_token:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Get user info for logging
                    cursor.execute("""
                    SELECT user_id FROM sessions WHERE session_token = ?
                    """, (session_token,))
                    
                    result = cursor.fetchone()
                    if result:
                        user_id = result[0]
                        
                        # Invalidate session
                        cursor.execute("""
                        UPDATE sessions SET is_active = FALSE WHERE session_token = ?
                        """, (session_token,))
                        
                        # Log logout activity
                        self.log_activity(user_id, "logout", "User logged out", conn)
                        
                        conn.commit()
            
            # Clear Streamlit session
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.session_state.session_token = None
            
        except Exception as e:
            st.error(f"Logout error: {str(e)}")
    
    def logout(self):
        """Simple logout for Streamlit session"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_id = None
        if 'session_token' in st.session_state:
            self.logout_user(st.session_state.session_token)
    
    def log_activity(self, user_id: int, activity_type: str, description: str, conn: sqlite3.Connection):
        """Log user activity"""
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO user_activity (user_id, activity_type, description)
            VALUES (?, ?, ?)
            """, (user_id, activity_type, description))
        except Exception as e:
            pass  # Logging errors shouldn't break the main flow
    
    def get_user_stats(self, user_id: int) -> dict:
        """Get user statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get user info
                cursor.execute("""
                SELECT username, email, created_at, last_login, full_name, institution
                FROM users WHERE id = ?
                """, (user_id,))
                
                user_info = cursor.fetchone()
                
                # Get activity count
                cursor.execute("""
                SELECT COUNT(*) FROM user_activity WHERE user_id = ?
                """, (user_id,))
                
                activity_count = cursor.fetchone()[0]
                
                if user_info:
                    username, email, created_at, last_login, full_name, institution = user_info
                    return {
                        'username': username,
                        'email': email,
                        'created_at': created_at,
                        'last_login': last_login,
                        'full_name': full_name,
                        'institution': institution,
                        'activity_count': activity_count
                    }
                else:
                    return None
        
        except Exception as e:
            return None
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                UPDATE sessions SET is_active = FALSE 
                WHERE expires_at <= CURRENT_TIMESTAMP AND is_active = TRUE
                """)
                conn.commit()
        except Exception as e:
            pass  # Silent cleanup