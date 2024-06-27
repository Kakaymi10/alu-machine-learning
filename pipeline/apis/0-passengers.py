#!/usr/bin/env python3
'''
Performing api manipulation
'''


import requests


def availableShips(passengerCount):
    '''
    method that returns the list of ships
    that can hold a given number of passengers
    '''
    url = 'https://swapi-api.alx-tools.com/api/starships/?page=1'
    response = requests.get(url).json()
    ships = []
    while response['next'] is not None:
        for ship in response['results']:
            if ship['passengers'] != 'n/a' and ship['passengers'] != 'unknown':
                ship['passengers'] = ship['passengers'].replace(',', '')
                if int(ship['passengers']) >= passengerCount:
                    ships.append(ship['name'])
        url = response['next']
        response = requests.get(url).json()
    if len(ships) == 0:
        return []
    return ships
