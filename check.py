from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from glob import glob
from tqdm import tqdm
import torchvision.transforms as T
import os
from PIL import Image
from dataclasses import dataclass

@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float

def print_results_table(results: list[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    for res in results:
        print(
            f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
            f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |',
            file=file,
        )

# Parameters
root_dir = 'data'
train_dir = 'test'
batch_size = 16
img_size = [32, 400]
charset_train = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[_`abcdefghijklmnopqrstuvwxyz{|}~¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿıŁłŒœŠšŸŽžƒˆˇˉ˘˙˚˛˜˝Ωπ؆؇؈؉؊؋،؍؎؏ؘؙؚؐؑؒؓؔؕؖؗ؛؝؞؟ؠءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْٕٖٓٔٗ٘٠٪٫٬٭ٰٱٴٹٻپٿچڈڐڑژڙکڪګڬڭڮگڰڱڲڳڴڵڶڷڸڹںڻڽھڿۀہۂۃۄۅۉۊیۍۏېۑےۓ۔ەۮۯ۰۱۲۳۴۵۶۷۸۹ۺۻۼ''""․ﷺ"
charset_test = charset_train
max_label_length = 120
augment = False
num_workers = 15
remove_whitespace = False
normalize_unicode = False

# Initialize data module
datamodule = SceneTextDataModule(root_dir, train_dir, img_size, max_label_length,
                                charset_train, charset_test, batch_size, num_workers, augment, 
                                remove_whitespace=remove_whitespace, normalize_unicode=normalize_unicode)

# Load the latest checkpoint
checkpoints = glob('/home/tawheed/parseq/outputs/parseq/2025-03-16_06-25-16/checkpoints/last.ckpt')
checkpoint_path = sorted(checkpoints)[-1]
print("Loaded checkpoint:", checkpoint_path)
model = load_from_checkpoint(checkpoint_path, charset_test=charset_test).eval().to('cuda')

# Validation dataloader
dataloader = datamodule.val_dataloader()

# Metrics computation
total = 0
correct = 0
ned = 0
confidence = 0
label_length = 0

for imgs, labels in tqdm(iter(dataloader)):
    res = model.test_step((imgs.to(model.device), labels), -1)['output']
    total += res.num_samples
    correct += res.correct
    ned += res.ned
    confidence += res.confidence
    label_length += res.label_length

# Calculate metrics
accuracy = 100 * correct / total
mean_ned = 100 * (1 - ned / total)
mean_conf = 100 * confidence / total
mean_label_length = label_length / total

# Create result object
validation_result = Result(
    dataset="Validation",
    num_samples=total,
    accuracy=accuracy,
    ned=mean_ned,
    confidence=mean_conf,
    label_length=mean_label_length
)

# Print results in table format
print("\nEvaluation Results:")
print_results_table([validation_result])

# Save images and labels if needed
output_dir = 'images_predicted'
os.makedirs(output_dir, exist_ok=True)
transform = T.ToPILImage()
ground_truth_file = os.path.join(output_dir, 'labels_ground_truth.txt')

with open(ground_truth_file, 'w') as gt_f:
    for images, labels in datamodule.val_dataloader():
        results = model.test_step((images.to(model.device), labels), -1)['output']
        
        for i, img in enumerate(images):
            pil_img = transform(img)
            image_path = os.path.join(output_dir, f"image_{i}.png")
            pil_img.save(image_path)
            gt_f.write(f"Image: {image_path}, Ground Truth: {labels[i]}\n")
        break

print(f"\nGround truth saved in: {ground_truth_file}")