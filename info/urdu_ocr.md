# data generation using TRDG
- create a folder named ur in trdg/fonts and add urdu fonts in it
- run command (to generate 10 million images of urdu):
    ```
        python run.py -l ur -c 10 -w 5 -f 64 -d 3 -k 5 -rk -bl 2 -rbl -b 3 -na 2 --word_split -t 5 --output_dir out/
    ```

    ```
    python run.py -l ur -c 10 -w 12 -f 120 -d 3 -k 2 -rk -bl 1 -rbl -b 3 -na 2 --word_split -t 5 --output_dir out/ -stw 4 -fd fonts/ab -fi -m 1 -sw 0.5
    ```
    Here -l means language to select
    -c means count
    -w means how many words to include in each image (width of image)
    -f means height of image
    -d defines the distortion
    -k defines the sekwness of text in image, -rk will randomly select skewness in range [-k, k]
    -bl defines the blur value in image, -rbl selects blur value randomly in range [0, bl]
    -b defines the backgound type
    -na defines name format
    --word_split : as urdu is a non-latin language in which, when characters are combined they make sense , so we need words here not characters only
    -t means how many threads to make

# Data generation using synthTiger

1. first of all prepare vocabulary for urdu language, add text file to `resources/corpus/`
2. Add fonts to `resources/font/`
3. Add charset to `resources/charset`
4. Now In the config files edit the vocab files , charset file and font file (as in `examples/synthtiger/config_horizontal.yaml`)
5. Run the following command to generate images:

    ```
    synthtiger -o results/ -w 12 -c 40 -v examples/synthtiger/template.py SynthTiger examples/synthtiger/config_horizontal.yaml
    ```

    where
    - -c: count of images
    - -w: max width of image
    - -v: verbose

# Create Lmbd dataset from images +gt.txt
Our parseq model gets trained on .lmbd dataset, so we need to convert the images to .lmbd files.

```
python tools/create_lmdb_dataset.py <path_to_images_folder> <path_to_ground_truth_file> <path_to_output_folder>
```

# Training parseq model

1. add your charset to `config/charset`
2. add training data into `data/train/`
3. add val data to `data/val/` and test data to `data/test/`
4. change the conifguration of main.yaml as per your needs
5. If you want to see 10 different checkpoint details, change `save_top_k` variable to 10 in train.py
6. run the following command to train the model: `python train.py`

# Evaluating the model
```
python test.py outputs/parseq/2025-01-11_13-44-12/checkpoints/last.ckpt --batch_size=32
```

# Read images
```
python read.py outputs/parseq/2025-02-04_23-24-10/checkpoints/last.ckpt --images demo_images/urdu/test2/out/* > demo_images/urdu/test2/predicted.txt
```


# How to calculate CRR and WRR
first take a dataset with images and gt file in some folder, copy path of images folder and paste in `parseq_infer.py`. 

after that you can run infer files.


# Additional information
After getting low crr and wrr, we made a vocab set (using create_vocabulary.py) from the test dataset and added that to the orignal vocabulary so that  when we generate the data out data should contain words which are present in test set so that the model trains well on them

also we took an analysis of which characters are getting missed in the test set (using infer_with_missing_characters.py)

we also created words and sentences of those characters which were getting missed by the model


we generated new images (image information present in images.txt)

then we trained our model once more on 32x400 image size