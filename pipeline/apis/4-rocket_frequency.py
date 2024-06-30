#!/usr/bin/env python3
'''
Get rocket names from launch details
'''


import requests
from collections import defaultdict


def get_rocket_name():
    '''return rockets with their num of launches'''
    url = 'https://api.spacexdata.com/v4/launches'
    response = requests.get(url)
    rockets = defaultdict(int)
    for launch in response.json():
        rockets[launch['rocket']] += 1

    names = defaultdict(str)

    for rocket in rockets:
        url = 'https://api.spacexdata.com/v4/rockets/{}'.format(rocket)
        response = requests.get(url)
        names[rocket] = response.json().get('name', 'Unknown rocket')

    # Print rocket names with the number of launches in decreasing order
    sorted_rockets = sorted(rockets.items(), key=lambda x: x[1], reverse=True)
    for rocket, count in sorted_rockets:
        print('{}: {}'.format(names.get(rocket, 'Unknown rocket'), count))


if __name__ == '__main__':
    get_rocket_name()
