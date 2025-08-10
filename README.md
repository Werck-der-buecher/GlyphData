# Synthesis of historical glyph data for skeletonization tasks
A software package for synthetic data generation of glyphs in historical printed documents. The main purpose of these synthetic glyphs is their use in deep learning-based skeletonization methods. 

For realistic stylization of the synthetic glyphs, we use "AdaAttN" by Liu et al. (https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_AdaAttN_Revisit_Attention_Mechanism_in_Arbitrary_Neural_Style_Transfer_ICCV_2021_paper.pdf).

**Attention**: The software package needs additional refactoring and will be updated in due time with more instructions on how to best use the software.

## Installation

#### 1. Install miniconda: 

   https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html

#### 2. Clone the repository:
To get started, simply clone the repository to your computer. Open a terminal (Command Prompt, PowerShell, or a terminal window on Mac/Linux) and run the following command:

```bash
git clone --recurse-submodules git@github.com/Werck-der-buecher/GlyphData.git
```

This command will clone the main project repository and automatically download and set up the AdaAttN submodule in the current version.

#### 3. Navigate to main project folder

```bash
cd GlyphData
```

#### 4. Create the conda environment from the provided environment.yml and activate it:

```bash
conda env create -f environment.yml
conda activate wdb_data
```
   
#### 5. Install wdb-glyphdata package

```bash
pip install -e .
```
  
## Downloads
A collection of data to get you started will be uploaded shortly.

   
## Console Apps
This project provides three command‑line tools:

### 1. `rasterize`

Take one or more font files (.otf or .ttf) and rasterize them for further processing in this package. For each input file, the rasterization output is a compressed file. This format is used within WdB to exchange glyph data between modules.

```bash
wdb-data-rasterize
    --font-dir /path/to/font_dir
    --output-dir /path/to/output_dir
```
#### Options:

- "--font-dir", "-f", \[DIRECTORY\]
: Directory that contains the font files (.otf, .ttf) that should be rasterized. \[Required.\]

- "--output-dir", "-o", \[DIRECTORY\]
: Directory where rasterized font files will be saved. \[Required.\].

### 2. `image-dump`
You can create an image dump of the glyph images. Optionally, you can also save the respective skeleton images. This utility function is useful if you want to evaluate the rasterization results or if you want to directly use the synthetic glyph images for stylization with AdaAttN without further augmentation.  

```bash
wdb-data-dump   
    --raster-dir /path/to/rasterized_fonts
    --output-dir /path/to/output_dir
    --image-size 128
    --output-format image_only
    --color-mode black_on_white
```
#### Options:

- "--raster-dir", "-r", \[DIRECTORY\]
: Directory that contains the rasterized files (.pbz2) for which an image dump should be
                                  performed. \[Required.\].          

- "--output-dir", "-o", \[DIRECTORY\]     
: Directory where the image dump will be saved. \[Required.\].

- "--image-size", "-s", \[INTEGER\]
: Square image size as a single integer (e.g., 128). 

- "--output-format", "-f", \[image_only|skeleton_only|all\]
: Output format: one of 'image_only', 'skeleton_only', or 'all'.

- "--color-mode", "-c", \[black_on_white|white_on_black\]
: Color made: one of 'black_on_white' or 'skelewhite_on_blackton_only'.


### 3. `augment`

It is generally benefitial to augment the rasterized glyph images before stylization with AdaAttN to obtain a much greater diversity of image characteristics. The augmentation uses several different randomized processing steps, including: 
1. morphological erosion and dilation, 
2. rotation, 
3. scaling, 
4. shifting, 
5. integration of ink traces of adjacent glyphs in the same line or from the text line above and below, 6. various levels of physical ink diffusion, 
7. slight elastic deformation,
8. Perlin noise, 
9. local intensity variations across the image region, 
10. random global intensity,
11. splatter background noise 

**Notes**
- The output of this process is divided into three subdirectories: "_sized", "_residual", and "composite". For further processing, e.g., for stylization with AdaAttN, you want to use "composite" as the image source. There, the augmented images are stored.
- Please note that the number of augmentations you specify in the command below is *per rasterized glyph*. For example, if you have 250 rasterized glyphs and define "num-augmentation" as 20, you'll obtain 250*20=5000 augmented images.

```bash
wdb-data-augment   
  --raster-dir /path/to/rasterized_fonts
  --output-dir /path/to/output_dir
  --num-augmentations 20
  --image-size 128
```
#### Options:

- "--raster-dir", "-r", \[DIRECTORY\]
: Directory that contains the rasterized files (.pbz2) for which an image dump should be
                                  performed. \[Required.\].          

- "--output-dir", "-o", \[DIRECTORY\]     
: Directory where the augmented images will be saved. \[Required.\].

- "--num-augmentations", "-n", \[INTEGER\]
: The number of augmented version that shall be generated per rasterized glyph. \[Required.\].

- "--image-size", "-s", \[INTEGER\]
: Square image size as a single integer (e.g., 128). 

### 4. `stylize`

To stylize the synthetic glyph images (or their augmented variant), we use the style transfer technique AdaAttN. Given *content* images and a handful of *style* images, this deep learning method generates images that adapt the style of the content images while retaining their original structure. This is highly useful and demonstrated great empirical results.

```bash
wdb-data-stylize   
  --image-dir /path/to/image_dump OR /path/to/augmented_images
  --style-dir /path/to/style_images
  --output-dir /path/to/output/results
  --image-size 128
  --max-results 100000
```
#### Options:

- "--image-dir", "-c", \[DIRECTORY\]
: Directory that contains the content images which shall be adapted in their style. \[Required.].          

- "--style-dir", "-t", \[DIRECTORY\]     
: Directory that contains the a handful of image files that shall serve as style references. \[Required.\].

- "--output-dir", "-o", \[DIRECTORY\]     
: Directory where the stylized images will be saved. \[Required.\].

- "--image-size", "-s", \[INTEGER\]
: Square image size as a single integer (e.g., 128). 

- "--max-results", "-m", \[INTEGER\]
: Maximum number of output files.

