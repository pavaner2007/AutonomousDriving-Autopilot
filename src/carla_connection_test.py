#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA Connection Test
=====================

Simple script to verify CARLA simulator connection and display server info.

Author: Pavan (pavaner2007)
Date: 2026
License: MIT

Usage:
    1. Start CARLA server: ./CarlaUE4.sh (Linux) or CarlaUE4.exe (Windows)
    2. Run this script: python carla_connection_test.py
"""

import carla
import time

def main():
    """Test connection to CARLA server and display basic information."""
    print("Attempting to connect to CARLA server...")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        
        # Get the world object
        world = client.get_world()
        
        print(f"Successfully connected to CARLA Server!")
        print(f"Client Version: {client.get_client_version()}")
        print(f"Server Version: {client.get_server_version()}")
        print(f"Current Map: {world.get_map().name}")
        
        # List some actors to verify
        actors = world.get_actors()
        print(f"Number of actors in the world: {len(actors)}")
        
    except RuntimeError as e:
        print(f"Error connecting to CARLA: {e}")
        print("Make sure the CARLA server is running (./CarlaUE4.sh)")

if __name__ == '__main__':
    main()
