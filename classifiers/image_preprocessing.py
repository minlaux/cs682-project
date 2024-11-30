import random
import torch
from torchvision import transforms
import os
from PIL import Image
from torchvision.utils import save_image
import time
from data_preprocessing import image_names, not_in_table

# define paths to input and output folders
input_folder = '../data/images_gz2/images'
# output_folder = '../data/images_gz2/processed_images'
output_folder = '../data/images_gz2/images_processed'

# define transformation function
def compute_train_transform(seed=123456):
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    colour_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    train_transform = transforms.Compose([
        # randomly resize and crop to 32 x 32
        transforms.RandomResizedCrop(32),  
        # horizontal flip with 50% probability
        transforms.RandomHorizontalFlip(p=0.5),  
        # colour jitter with 80% probability
        transforms.RandomApply([colour_jitter], p=0.8),  
        # convert to greyscale with 20% probability
        transforms.RandomGrayscale(p=0.2),  
        # convert image to tensor
        transforms.ToTensor(),  
    ])
    return train_transform

# # load image for testing
# image_path = "/content/images_gz2/images/223272.jpg"  # Update with your image path
# img = Image.open(image_path)

# train_transform = compute_train_transform()
# transformed_img_tensor = train_transform(img)

# transformed_img = transforms.ToPILImage()(transformed_img_tensor)

# # display the processed image
# plt.imshow(transformed_img)
# plt.axis('off')
# plt.show()


# # ensure the output folder exists
# os.makedirs(output_folder, exist_ok=True)

# # loop over all images in the input folder
# for filename in os.listdir(input_folder):
#     img_path = os.path.join(input_folder, filename)
    
#     # skip non-image files
#     if not filename.endswith(('.jpg', '.jpeg', '.png')):
#         continue
    
#     # load image
#     img = Image.open(img_path)
    
#     # apply transformation
#     train_transform = compute_train_transform()
#     img_transformed = train_transform(img)
    
#     # save transformed image to output folder
#     # IMPORTANT: keep image filename the same as for not processed
#     # image filename corresponds to asset id in dataframe
#     output_path = os.path.join(output_folder, filename)
#     save_image(img_transformed, output_path)
    
#     print(f"Saved processed image: {output_path}")

# Keep track of how long this will take
start_process_time = time.time()

current_image = 0
for image in image_names:
    if image in not_in_table:
        # these images do not need to be processed as they are not in our data table
        continue
    else:
        # apply transformation
        train_transform = compute_train_transform()
        img_transformed = train_transform(image)

        # save transformed image to output folder
        # IMPORTANT: keep image filename the same as for not processed
        # image filename corresponds to asset id in dataframe
        output_path = os.path.join(output_folder, image)
        save_image(img_transformed, output_path)

        # # process the image and save as a PNG file
        # process_image(image, save_dir=PROCESSED_IMAGES_DIR, visualize=False)
    # except Exception as e:
    #     print(f"❌ Image `{image}` failed to process (current_image int= `{current_image}`)")
    #     traceback.print_exc()
    #     break
    current_image += 1
    if current_image % 10_000 == 0:
        print(f"  Processed {current_image:,} image files")
        
print("Processing complete")

_hr, _remainder = divmod(time.time() - start_process_time, 3600)
_min, _sec = divmod(_remainder, 60)
print(f"--- Time Taken: {int(_hr):02d}:{int(_min):02d}:{int(_sec):02d} ---")