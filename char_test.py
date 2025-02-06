from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from glob import glob
from tqdm import tqdm
import os
from dataclasses import dataclass

@dataclass
class CharResult:
    dataset: str
    total_chars: int
    correct_chars: int
    char_accuracy: float
    avg_ned: float
    avg_confidence: float

def print_char_results_table(results: list[CharResult]):
    print('| Dataset | Total Chars | Correct Chars | Char Accuracy | NED | Confidence |')
    print('|:--------|------------:|--------------:|--------------:|----:|-----------:|')
    for res in results:
        print(f'| {res.dataset:<8} | {res.total_chars:>11} | {res.correct_chars:>12} | '
              f'{res.char_accuracy:>12.2f} | {res.avg_ned:>4.2f} | {res.avg_confidence:>10.2f} |')

# Initialize parameters
root_dir = 'data'
train_dir = 'test'
batch_size = 8
img_size = [64, 500]
charset_train = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[_`abcdefghijklmnopqrstuvwxyz{|}~¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿıŁłŒœŠšŸŽžƒˆˇˉ˘˙˚˛˜˝Ωπ؆؇؈؉؊؋،؍؎؏ؘؙؚؐؑؒؓؔؕؖؗ؛؝؞؟ؠءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْٕٖٓٔٗ٘٠٪٫٬٭ٰٱٴٹٻپٿچڈڐڑژڙکڪګڬڭڮگڰڱڲڳڴڵڶڷڸڹںڻڽھڿۀہۂۃۄۅۉۊیۍۏېۑےۓ۔ەۮۯ۰۱۲۳۴۵۶۷۸۹ۺۻۼ''""․ﷺ"
charset_test = charset_train
max_label_length = 120
num_workers = 15

datamodule = SceneTextDataModule(root_dir, train_dir, img_size, max_label_length,
                                charset_train, charset_test, batch_size, num_workers,
                                augment=False, normalize_unicode=False)

checkpoints = glob('/home/tawheed/parseq/outputs/parseq/urdu_500x64/checkpoints/last.ckpt')
checkpoint_path = sorted(checkpoints)[-1]
print(f"Loading checkpoint: {checkpoint_path}")
model = load_from_checkpoint(checkpoint_path, charset_test=charset_test).eval().to('cuda')

# Evaluation
total_samples = 0
total_correct = 0 
total_ned = 0
total_confidence = 0
total_chars = 0

for imgs, labels in tqdm(datamodule.val_dataloader()):
    batch_result = model.test_step((imgs.to(model.device), labels), -1)['output']
    total_samples += batch_result.num_samples
    total_correct += batch_result.correct
    total_ned += batch_result.ned
    total_confidence += batch_result.confidence
    total_chars += batch_result.label_length

# Calculate metrics
char_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
avg_ned = total_ned / total_samples if total_samples > 0 else 0
avg_confidence = total_confidence / total_samples if total_samples > 0 else 0

result = CharResult(
    dataset="Validation",
    total_chars=total_chars,
    correct_chars=total_correct,
    char_accuracy=char_accuracy,
    avg_ned=avg_ned,
    avg_confidence=avg_confidence
)

print("\nCharacter-Level Evaluation Results:")
print_char_results_table([result])