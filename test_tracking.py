from track_objects import track_objects_in_video

video_path = "home_tour.mp4"
memory = track_objects_in_video(video_path)

# Save the memory to a JSON file (optional)
import json
with open('memory_log.json', 'w') as f:
    json.dump(memory, f, indent=2)

# Print object memory
for mem in memory:
    print(mem)

# Example: Find all frames where keys appeared
print("\n🔑 Keys found in frames:")
for mem in memory:
    if mem['object'].lower() == 'keys':
        print(f"Frame {mem['frame']} - {mem['bbox']} - ID: {mem['id']}")
