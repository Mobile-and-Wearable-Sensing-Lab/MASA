import os

# --- CONFIGURATION ---
# Actual path to where your .npy files are
POSE_ROOT = 'Data/ISL_GOA/Pose'
# Where to save the list
OUTPUT_FILE = 'Data/ISL_GOA/Annotations/pretrain_list.txt'
# Where to save the class map
CLASS_MAP_FILE = 'Data/ISL_GOA/Annotations/class_map.txt'

def generate_labels():
    samples = []
    label_to_id = {}
    current_id = 0
    
    print(f"Scanning: {POSE_ROOT}")

    # Walk through the folders
    # followlinks=True ensures we follow the shortcuts if you have symlinks
    for root, dirs, files in os.walk(POSE_ROOT, followlinks=True):
        for file in files:
            if file.endswith('.npy'):
                full_path = os.path.join(root, file)
                
                # --- THE KEY CHANGE IS HERE ---
                # We calculate path relative to 'Data/ISL_GOA/Pose'
                # Result: User001/User001_Keypoints/File.npy
                try:
                    rel_path = os.path.relpath(full_path, POSE_ROOT)
                except ValueError:
                    # Fallback if paths depend on symlinks
                    rel_path = os.path.join(os.path.basename(root), file)

                # Extract Class Name (e.g. "Restaurant")
                try:
                    class_name = file.split('__')[0]
                except:
                    continue

                # Assign ID
                if class_name not in label_to_id:
                    label_to_id[class_name] = current_id
                    current_id += 1
                
                label_id = label_to_id[class_name]
                samples.append(f"{rel_path} {label_id}")

    # Save files
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(samples))

    with open(CLASS_MAP_FILE, 'w') as f:
        for name, id_ in sorted(label_to_id.items(), key=lambda item: item[1]):
            f.write(f"{id_} {name}\n")

    print(f"Done! Processed {len(samples)} videos.")
    print(f"Path Example: {samples[0] if samples else 'None'}")

if __name__ == "__main__":
    generate_labels()
