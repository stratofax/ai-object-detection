# read a list of images from a directory and select the ones that have people
import os

import detect_people

img_dir = "/Volumes/ExtSSD/Media/NHCC/slideshow/selected"

# Get a list of all the image files in the directory
# filter out any files that are not image files

image_files = [
    file for file in os.listdir(img_dir) if file.endswith((".png", ".jpg", ".jpeg"))
]

# create a list of image files that have people
people_detected = []
no_people_detected = []
not_processed = []

# loop through the list of image files to detect people
for image_file in image_files:
    image_file = f"{img_dir}/{image_file}"
    print(f"Checking for people in {image_file}...")
    try:
        if detect_people.people_detected(image_file):
            people_detected.append(image_file)
            print(f"Found people in {image_file}")
        else:
            no_people_detected.append(image_file)
            print(f"No people found in {image_file}")

    except RuntimeError as e:
        not_processed.append(image_file)
        print(f"Error occurred while detecting people in {image_file}: {str(e)}")
    except UnidentifiedImageError:
        not_processed.append(image_file)
        print(f"Not an mage file: {image_file}: {str(e)}")


# print the lists of people detected and people not detected
print(f"Images with people detected: \n{people_detected}")
print(f"Images with no people detected: \n{no_people_detected}")
print(f"Images not processed: \n{not_processed}")

# write the lists of people detected and people not detected to a file
with open("people_detected.txt", "w") as f:
    f.write("Images with people detected:\n")
    for image in people_detected:
        f.write(image + "\n")

with open("people_not_detected.txt", "w") as f:
    f.write("Images with no people detected:\n")
    for image in no_people_detected:
        f.write(image + "\n")

with open("not_processed.txt", "w") as f:
    f.write("Images not processed:\n")
    for image in not_processed:
        f.write(image + "\n")
