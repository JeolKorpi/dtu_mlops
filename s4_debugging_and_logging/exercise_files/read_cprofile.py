import pstats
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent
profile_path = script_dir / 'profile.txt'

p = pstats.Stats(str(profile_path))
p.sort_stats('cumulative').print_stats(20)