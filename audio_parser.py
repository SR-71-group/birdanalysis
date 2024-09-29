# audio_parser.py
import re

def parse_filename(filename):
    """
    Extract relevant information from the audio filename.
    The pattern is: Julianisches Datum - Ort - tiefste und höchste Frequenz -
    Anfangszeit und Endzeit des Rufs in der Aufnahme - Species-Code.

    Returns a dictionary with extracted components.
    """
    # Example format: "2459995.723424_Tautenburg___1652-6323kHz___10-20.3s___s..wav"
    pattern = r"([0-9.]+)_([A-Za-z]+)___([0-9-]+kHz)___([0-9.]+-[0-9.]+s)___([A-Za-z]+)"

    match = re.match(pattern, filename)
    if match:
        return {
            'julian_date': match.group(1),
            'location': match.group(2),
            'frequency_range': match.group(3),
            'time_range': match.group(4),
            'species_code': match.group(5)
        }
    else:
        raise ValueError("Filename does not match expected format.")

# Example usage:
# filename = "2459995.723424_Tautenburg___1652-6323kHz___10-20.3s___s..wav"
# parsed = parse_filename(filename)
# print(parsed)
