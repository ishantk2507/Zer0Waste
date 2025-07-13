import random


def get_route(origin, destination):
    """
    Mock routing function returning distance and CO2 emissions between two locations.

    Args:
        origin (str): City name of origin (e.g., 'Delhi').
        destination (str): City name of destination.

    Returns:
        dict: { 'distance_km': float, 'emissions_kg': float }
    """
    # Generate a plausible distance between 5 and 15 km
    distance_km = round(random.uniform(5.0, 15.0), 2)
    # Assume emissions factor of 0.25 kg CO2 per km
    emissions_kg = round(distance_km * 0.25, 2)

    return {
        'distance_km': distance_km,
        'emissions_kg': emissions_kg
    }