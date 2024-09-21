import os
from typing import Any

from langchain_community.graphs import Neo4jGraph
import numpy as np

def _get_current_hospitals()->list[str]:
    """Fetch a list of current hospital names from a Neo4j database."""
    graph = Neo4jGraph(
        url = os.getenv("NEO4J_URI"),
        username = os.getenv("NEO4J_USERNAME"),
        password = os.getenv("NEO4J_PASSWORD")
    )

    current_hospitals = graph.query(
        """
        MATCH (h:Hospital) 
        RETURN h.name AS hospital_name
        """
    )

    return [h["hospital_name"].lower() for h in current_hospitals]

def _get_current_wait_time_minutes(hospital:str)->int:
    """Get the current wait time in minutes for a given hospital."""
    current_hospitals = _get_current_hospitals()

    if hospital.lower() not in current_hospitals:
        return -1
    
    return np.random.randint(low=0, high=600)

def get_current_wait_times(hospital:str)->int:
    """Get the current wait time in minutes for a given hospital formatted as a string."""

    wait_time_in_minutes = _get_current_wait_time_minutes(hospital)

    if wait_time_in_minutes == -1:
        return f"Hospital {hospital} does not exist"
    
    hours, minutes = divmod(wait_time_in_minutes, 60)

    if hours == 0:
        return f"{minutes} minutes"
    
    return f"{hours} hours and {minutes} minutes"

def get_most_available_hospital(_:Any) -> dict[str, float]:
    """ Find the hospital with the lowest wait time. """
    current_hospitals = _get_current_hospitals()

    current_wait_times = [_get_current_wait_time_minutes(h) for h in current_hospitals]

    best_time_idx = np.argmin(current_wait_times)
    best_hospital = current_hospitals[best_time_idx]
    best_wait_time = current_wait_times[best_time_idx]

    return {best_hospital: best_wait_time}

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    print(get_current_wait_times('Wallace-Hamilton'))
    print(get_current_wait_times('Fake Hospital'))

    print(get_most_available_hospital(None))