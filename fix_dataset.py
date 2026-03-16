import os

dataset_dir = r"d:\AI_Practice\image_captioning\flickr8k"
token_file = os.path.join(dataset_dir, "Flickr8k.token.txt")
image_dir = os.path.join(dataset_dir, "Flicker8k_Dataset")

print("Checking dataset for missing images...")

with open(token_file, "r") as f:
    lines = f.readlines()

valid_lines = []
missing_images = set()

for line in lines:
    tokens = line.split('\t')
    if len(tokens) < 2:
        continue
    
    # Extract the raw image name
    raw_img = tokens[0].split('#')[0]
    
    # Check if we successfully clean the known .1 typo
    img = raw_img.replace(".jpg.1", ".jpg")
    
    img_path = os.path.join(image_dir, img)
    if os.path.exists(img_path):
        # Image actually exists, safe to keep!
        # Make sure that if it was the .1 typo, we save the CLEAN version
        clean_line = line.replace(".jpg.1", ".jpg")
        valid_lines.append(clean_line)
    else:
        missing_images.add(img)

if len(missing_images) > 0:
    print(f"Found {len(missing_images)} missing images referenced in the text file.")
    print("Rewriting Flickr8k.token.txt to ignore missing imagery...")
    with open(token_file, "w") as f:
        f.writelines(valid_lines)
    print("Dataset cleaned successfully!")
else:
    print("Dataset is already clean!")
