"""Unit tests for Data2txt natural-language chunk formatting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.text_chunker import chunk_data2txt


def test_chunk_data2txt_formats_hours_and_attributes_naturally():
    source_info = {
        'name': 'Finch & Fork',
        'hours': {
            'Monday': '17:30-23:0',
            'Sunday': '9:0-14:0',
        },
        'attributes': {
            'RestaurantsReservations': True,
            'OutdoorSeating': False,
            'WiFi': 'free',
            'RestaurantsTakeOut': True,
            'RestaurantsGoodForGroups': True,
            'BusinessParking': {
                'garage': True,
                'street': True,
                'valet': True,
            },
        },
    }

    contexts = chunk_data2txt(source_info)
    full_text = "\n".join(contexts)

    assert 'Monday: 5:30 PM to 11:00 PM' in full_text
    assert 'Sunday: 9:00 AM to 2:00 PM' in full_text
    assert 'The restaurant accepts reservations.' in full_text
    assert 'Outdoor seating is not available.' in full_text
    assert 'Free WiFi is available.' in full_text
    assert 'Takeout is available.' in full_text
    assert 'The venue is good for groups.' in full_text
    assert 'Parking options include garage, street, valet.' in full_text
    assert 'Business attributes: Reservations: Yes.' not in full_text
