import os

# --- CONFIGURATION ---
# Path to your 'Pose' folder containing User001, User002, etc.
POSE_ROOT = 'Data/ISL_GOA/Pose'
# Where to save the new list
OUTPUT_FILE = 'Data/ISL_GOA/Annotations/pretrain_list.txt'
# Where to save the list of class names (so you know 0 = Restaurant)
CLASS_MAP_FILE = 'Data/ISL_GOA/Annotations/class_map.txt'

def generate_labels():
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    samples = []
    label_to_id = {}
    current_id = 0
    
    print("Scanning files...")

    # Walk through User001, User002, etc.
    for root, dirs, files in os.walk(POSE_ROOT, followlinks=True):
        for file in files:
            if file.endswith('.npy'):
                # 1. Get the full path relative to Data/ISL_GOA
                full_path = os.path.join(root, file)
                # This makes the path: Pose/User002/U_2/Restaurant...
                rel_path = os.path.relpath(full_path, 'Data/ISL_GOA')

                # 2. Extract the Class Name from the filename
                # format: Restaurant__session141__clip003.npy
                try:
                    # Split by double underscore to get the word
                    class_name = file.split('__')[0]
                    
                    # Safety check: if filename doesn't have __, skip or fallback
                    if class_name == file:
                        # Fallback logic if naming convention breaks
                        # Try finding the word based on folder structure? 
                        # For now, let's assume the format holds.
                        pass
                except Exception as e:
                    print(f"Skipping bad file: {file}")
                    continue

                # 3. Assign ID
                if class_name not in label_to_id:
                    label_to_id[class_name] = current_id
                    current_id += 1
                
                label_id = label_to_id[class_name]

                # 4. Add to list
                samples.append(f"{rel_path} {label_id}")

    # Write the main pretrain list
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(samples))

    # Write the class map (ID -> Word) for your reference
    with open(CLASS_MAP_FILE, 'w') as f:
        for name, id_ in sorted(label_to_id.items(), key=lambda item: item[1]):
            f.write(f"{id_} {name}\n")

    print(f"Done! Processed {len(samples)} videos.")
    print(f"Found {len(label_to_id)} unique words/classes.")
    print(f"Saved list to: {OUTPUT_FILE}")
    print(f"Saved class map to: {CLASS_MAP_FILE}")

if __name__ == "__main__":
    generate_labels()
