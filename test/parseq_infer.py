import os
import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
import time

def batch_images(images, batch_size):
    for i in range(0, len(images), batch_size):
        yield images[i:i + batch_size]

def process_image_content(img, parseq, device):
    logits = parseq(img)
    logits.shape
    pred = logits.softmax(-1)
    label, _ = parseq.tokenizer.decode(pred)
    return label

def process_folder(folder_path, output_folder, parseq, img_transform, device, batch_size):
    images = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))],
                    key=lambda x: int(os.path.splitext(x)[0]))

    if len(images) == 0:
        print("No images found in the folder.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for image_file in images:
        image_path = os.path.join(folder_path, image_file)
        image_name = os.path.splitext(image_file)[0]

        output_file_path = os.path.join(output_folder, f'{image_name}.txt')

        with open(output_file_path, 'w+') as out:
            for batch in batch_images([image_file], batch_size):
                image_paths = [os.path.join(folder_path, img_file) for img_file in batch]
                img = [Image.open(img_path).convert('RGB') for img_path in image_paths]

                img = [img_transform(im) for im in img]
                img = torch.stack(img).to(device)

                st = time.time()
                results = process_image_content(img, parseq, device)
                fn = time.time()

                for res in results:
                    res = res.lower()
                    out.write(res + '\n')

                print(f"Processed {image_file} in {fn-st:.4f} seconds.")

def main():
    main_folder_path = '/DATA/Tawheed/data/crr-wrr/UPTI/images'
    output_folder_path = '/DATA/Tawheed/data/crr-wrr/UPTI/pred'
    
    model_path = '/home/tawheed/parseq/outputs/parseq/2025-03-23_05-59-14/checkpoints/epoch=10-step=1274154-val_accuracy=31.3888-val_NED=92.0311.ckpt'
    parseq = load_from_checkpoint(model_path).eval().to('cuda')  # Move the model to GPU
    
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 96
    process_folder(main_folder_path, output_folder_path, parseq, img_transform, device, batch_size)

if __name__ == "__main__":
    main()
