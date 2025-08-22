#!/usr/bin/env python3
"""
Port Management Utility for GPT-OSS HF Server
Handles port availability checking and process management
"""

import socket
import psutil
import os
import sys
import signal
import time
from typing import Optional, List, Tuple


def check_port_available(host: str = '0.0.0.0', port: int = 8000) -> bool:
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except (OSError, socket.error):
        return False


def find_process_using_port(port: int) -> Optional[Tuple[int, str]]:
    """Find process using specific port
    Returns: (pid, process_name) or None
    """
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            connections = proc.connections(kind='inet')
            for conn in connections:
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    return proc.pid, proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def kill_process_on_port(port: int, force: bool = False) -> bool:
    """Kill process using specific port
    Returns: True if successful, False otherwise
    """
    process_info = find_process_using_port(port)
    if not process_info:
        return False
    
    pid, name = process_info
    try:
        process = psutil.Process(pid)
        if force:
            process.kill()  # SIGKILL
        else:
            process.terminate()  # SIGTERM
        
        # Wait for process to terminate
        try:
            process.wait(timeout=5)
        except psutil.TimeoutExpired:
            if not force:
                # Try force kill if gentle termination failed
                process.kill()
                process.wait(timeout=3)
        
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        print(f"Error killing process {pid}: {e}")
        return False


def find_available_port(start_port: int = 8000, max_tries: int = 10) -> Optional[int]:
    """Find an available port starting from start_port"""
    for i in range(max_tries):
        port = start_port + i
        if check_port_available(port=port):
            return port
    return None


def interactive_port_resolution(desired_port: int = 8000) -> int:
    """Interactive resolution for port conflicts
    Returns: Final port to use
    """
    if check_port_available(port=desired_port):
        return desired_port
    
    # Port is in use
    process_info = find_process_using_port(desired_port)
    if process_info:
        pid, name = process_info
        print(f"\n⚠️  Port {desired_port} is already in use by process '{name}' (PID: {pid})")
    else:
        print(f"\n⚠️  Port {desired_port} is already in use by an unknown process")
    
    print("\nWhat would you like to do?")
    print("1. Kill the existing process and use port", desired_port)
    print("2. Use a different port")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                # Kill existing process
                print(f"\nAttempting to kill process on port {desired_port}...")
                if kill_process_on_port(desired_port):
                    print("✅ Process killed successfully")
                    time.sleep(1)  # Give OS time to release the port
                    
                    # Verify port is now available
                    if check_port_available(port=desired_port):
                        print(f"✅ Port {desired_port} is now available")
                        return desired_port
                    else:
                        print(f"❌ Port {desired_port} still not available")
                        # Fall through to find alternative
                else:
                    print("❌ Failed to kill process")
                
                # Try to find alternative port
                alt_port = find_available_port(desired_port + 1)
                if alt_port:
                    print(f"Using alternative port: {alt_port}")
                    return alt_port
                else:
                    print("❌ No available ports found")
                    sys.exit(1)
                    
            elif choice == '2':
                # Find and use different port
                alt_port = find_available_port(desired_port + 1)
                if alt_port:
                    print(f"✅ Using alternative port: {alt_port}")
                    return alt_port
                else:
                    print("❌ No available ports found in range")
                    sys.exit(1)
                    
            elif choice == '3':
                print("Exiting...")
                sys.exit(0)
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


def auto_port_resolution(desired_port: int = 8000, auto_kill: bool = False) -> int:
    """Automatic port resolution without user interaction
    
    Args:
        desired_port: Preferred port to use
        auto_kill: If True, automatically kill process using the port
    
    Returns:
        Available port number
    """
    if check_port_available(port=desired_port):
        return desired_port
    
    process_info = find_process_using_port(desired_port)
    if process_info:
        pid, name = process_info
        print(f"⚠️  Port {desired_port} is in use by '{name}' (PID: {pid})")
        
        if auto_kill:
            print(f"Auto-killing process on port {desired_port}...")
            if kill_process_on_port(desired_port):
                time.sleep(1)
                if check_port_available(port=desired_port):
                    print(f"✅ Port {desired_port} is now available")
                    return desired_port
    
    # Find alternative port
    alt_port = find_available_port(desired_port + 1)
    if alt_port:
        print(f"Using alternative port: {alt_port}")
        return alt_port
    
    raise RuntimeError(f"No available ports found starting from {desired_port}")


def list_server_processes() -> List[Tuple[int, str, int]]:
    """List all running server processes
    Returns: List of (pid, name, port) tuples
    """
    server_processes = []
    server_names = ['server_v', 'uvicorn', 'python']
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if it's a server process
            is_server = False
            for server_name in server_names:
                if server_name in proc.info['name']:
                    is_server = True
                    break
                if proc.info['cmdline']:
                    for arg in proc.info['cmdline']:
                        if 'server_v' in arg:
                            is_server = True
                            break
            
            if is_server:
                # Find listening ports
                connections = proc.connections(kind='inet')
                for conn in connections:
                    if conn.status == 'LISTEN':
                        server_processes.append((
                            proc.info['pid'],
                            proc.info['name'],
                            conn.laddr.port
                        ))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return server_processes


if __name__ == "__main__":
    # Test the port manager
    import argparse
    
    parser = argparse.ArgumentParser(description="Port Manager Utility")
    parser.add_argument('--port', type=int, default=8000, help='Port to check/manage')
    parser.add_argument('--list', action='store_true', help='List all server processes')
    parser.add_argument('--check', action='store_true', help='Check if port is available')
    parser.add_argument('--kill', action='store_true', help='Kill process on port')
    parser.add_argument('--auto', action='store_true', help='Auto mode (no interaction)')
    
    args = parser.parse_args()
    
    if args.list:
        processes = list_server_processes()
        if processes:
            print("\n📋 Running server processes:")
            for pid, name, port in processes:
                print(f"  PID: {pid:6} | Port: {port:5} | Name: {name}")
        else:
            print("No server processes found")
    
    elif args.check:
        if check_port_available(port=args.port):
            print(f"✅ Port {args.port} is available")
        else:
            process_info = find_process_using_port(args.port)
            if process_info:
                pid, name = process_info
                print(f"❌ Port {args.port} is in use by '{name}' (PID: {pid})")
            else:
                print(f"❌ Port {args.port} is in use")
    
    elif args.kill:
        if kill_process_on_port(args.port):
            print(f"✅ Killed process on port {args.port}")
        else:
            print(f"❌ No process found on port {args.port}")
    
    else:
        # Interactive mode
        if args.auto:
            port = auto_port_resolution(args.port, auto_kill=True)
        else:
            port = interactive_port_resolution(args.port)
        print(f"\n✅ Final port: {port}")