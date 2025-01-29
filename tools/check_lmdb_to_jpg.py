import os
import cv2
import numpy as np
import lmdb
# from strhub.data.utils import CharsetAdapter
import unicodedata

#charset = "ಀಁಂಃ಄ಅಆಇಈಉಊಋಌಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಱಲಳವಶಷಸಹ಼ಽಾಿೀುೂೃೄೆೇೈೊೋೌ್ೕೖೞೠೡೢೣ೦೧೨೩೪೫೬೭೮೯ೱೲ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789"

def check_label_charset(label, charset):
    for char in label:
        if char not in charset:
            return False
    return True

def unpack_mdb(data_dir, output_dir, gt_dir):
    """Unpack data from .mdb files and save images with labels"""

    # Open the .mdb environment
    env = lmdb.open(data_dir, readonly=True)

    # Open the transaction for data retrieval
    txn = env.begin()

    # from other code
    #charset_adapter = CharsetAdapter(charset)
    labels_list = []
    img_name = 0

    # Retrieve the data and labels
    with open(gt_dir+'/'+'gt.txt', 'w+', encoding='utf-8') as f:
        with txn.cursor() as cursor:
            for key, value in cursor:
                # Convert the image data to numpy array
                image_data = np.frombuffer(value, dtype=np.uint8)

                # Decode and save the image
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                #image_name = key.decode()
                image_name = img_name
                image_path = os.path.join(output_dir, f'{image_name}.jpg')
                # cv2.imwrite(image_path, image)


                # Extract and save the label
                label_key = key.decode()
                label_key = f'label-{label_key.split("-")[1]}'.encode()
                print(label_key)
                try:
                    label = txn.get(label_key).decode()
                except:
                    pass

                # try:
                #     label = txn.get(key).decode('utf-8')
                # except UnicodeDecodeError:
                #     try:
                #         label = txn.get(key).decode('utf-16')
                #     except UnicodeDecodeError:
                #         try:
                #             label = txn.get(key).decode('utf-16le')
                #         except UnicodeDecodeError:
                #             label = txn.get(key).decode('utf-8-sig')

                label_path = os.path.join(output_dir, f'{image_name}.txt')

                lbl = str(image_name) + ".jpg" + " " + str(label)

                check = True
                # label check if necessary
                # if len(charset) != 0:
                #     check = check_label_charset(str(label), charset)

                if check == True:

                    labels_list.append(lbl)

                    img_name += 1
                    try:
                        cv2.imwrite(image_path, image)
                        f.write(lbl+"\n")
                    except:
                        pass

        # with open('gt.txt', 'w+', encoding='utf-16') as f:
        #     for i in len(labels_list):
        #         f.write(labels_list[i]+"\n")

    env.close()

    print('Unpacking completed successfully.')


# Set the path to the directory containing the .mdb files and the output directory
data_dir = '/DATA/Tawheed/parseq_data/train/train_synth4'
gt_dir = '/DATA/Tawheed/train_synth4'
output_dir = '/DATA/Tawheed/train_synth4/images'

# Unpack the data and labels from the .mdb files
unpack_mdb(data_dir, output_dir, gt_dir)
