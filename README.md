# AstroLens: A Celestial Taxonomical Identifier + Moon Analysis Toolkit

## Core Functionality
**AstroLens** is a dual-function computer-vision-powered space object identifier and a lunar analysis tool.
It uses two custom trained machine learning models combined with computer vision to analyze celestial objects, classify them, and extract meaningful details from Moon photographs such as crater counts, moon phase, and enhanced imagey.

This project is currently in the MVP stage, and I am actively working on creating a new fully custom Solar System dataset. The current dataset I used for the Solar System classifier “Planets and Moons Dataset – AI in Space” by EMİRHAN BULUT consists mostly of synthetic 3D renderings and does not fully encompass the breadth of celestial objects in the Solar System that I believe are worth identifying (such as the Galilean moons, Titan, Ceres, Uranus’s major moons, Triton, and others).

### Function 1: Dual-model 
AstroLens uses two models in a heirarchical pipeline:

#### Model 1: Solar System Classifier
Currently, the Model I'm using in the MVP product is from EMİRHAN BULUT, but as previously stated, the Model did not give me the results I intended, and I am currently building a brand new custom dataset which contains the following Solar System Objects.

**Star:**
- The Sun

**Planets:**
- Mercury
- Venus
- Earth
- Mars
- Jupiter
- Saturn
- Uranus
- Neptune

**Dwarf Planets:**
- Pluto
- Ceres
- Eris
- Makemake
- Haumea

**Moons:**
- The Moon/Luna (Earth)
- Phobos (Mars)
- Deimos (Mars)
- Ganymede (Jupiter)
- Callisto (Jupiter)
- Io (Jupiter)
- Europa (Jupiter)
- Titan (Saturn)
- Rhea (Saturn)
- Enceladus (Saturn)
- Iapetus (Saturn)
- Tethys (Saturn)
- Mimas (Saturn)
- Rhea (Saturn)
- Dione (Saturn)
- Titania (Uranus)
- Ariel (Uranus)
- Umbriel (Uranus)
- Oberon (Uranus)
- Miranda (Uranus)
- Triton (Neptune)
- Proteus (Neptune)
- Charon (Pluto)
- Nix (Pluto)
- Hydra (Pluto)

The dataset size for each individual object is determined by the number of verifiable photos taken, angles/perspectives of said photos, as well as the general "uniqueness" of a Celestial object's external attributes.

**Please Note:** Dataset objects are not set-and-stone yet. As much as I wish I could create a class for every object in outer space, we know too little about some celestial bodies for this tool to semi-reliably work for every object. 🙁

#### Model 2: SpaceNet Deep-Sky Classification
To recognize objects outside the Solar System, AstroLens uses a second model trained on the SpaceNet-FLARE Astronomy Dataset created by Raza Imam & Mohammed Talha Alam, which includes just under 13,000 images consisting of the following celestial object classes:
- Asteroid
- Black Hole
- Comet
- Constellation
- Galaxy
- Nebula
- Planet
- Star

Currently, in the model, it acts as the first stage in the classification pipeline. Its job is to determine the broader object family before attempting the Solar System-specific refinement.

AstroLens uses SpaceNet to essentially answer questions such as:
- "Does this object look like a planet? Or is it closer to a star in resemblance?"
- "Is this bright object a Nebula, or is it a Star?"
- "Could this object be a comet?"
- "Does this look like a black hole?"

Currently, if SpaceNet confidently predicts "Planet", only then does AstroLens forward the image to the Solar System classifier for an even more specific classification (Earth, Jupiter, Makemake, etc.). If it predicts any other object besides "Planet" such as "Galaxy" or "Constellation", the app bypasses the Solar System dataset, which ensures faster inference, more accurate Solar System object identification, and provides a clear divide between deep space celestial objects and local celestial objects.

**As a side note: NONE** of the datasets I use to train the models will be included in this repository.
To respect copyright and data licensing terms, all training assets, such as downloadable images and the SpaceNet Kaggle Dataset, are kept entirely local and will not be redistributed or uploaded with this codebase. The references for the images I use to train the dataset are in this link:
[References](https://docs.google.com/document/d/18lAmfc0Urb-9y7zoctn-mRGmPS7Pr7295Z1oLwnXlzk/edit?usp=sharing)

### Function 2: Moon-Only tools:
AstroLens includes key Moon analysis information, gated by a Solar System classifier to ensure accuracy. If the model detects an image as the Moon with high confidence, the following tools are unlocked:

**Moon Phase Estimate**
Uses threshold and pixel segmentation to estimate light coverage and classify the moon image as:
- New Moon
- Crescent
- Quarter
- Gibbous
- Full Moon

**Crater Detection**
Uses a custom OpenCV pipeline that enhances contrast and applies circular hough translations into likely craters, as well as returning the total count of craters in an image.

**Image Enhancement**
Pre-processing improves clarity by using noise reduction, histogram equalization, and sharpening filters.

## UI Design
Currently, I am using **Streamlit**, and the MVP product includes:
- A two-tab layout (Moon Tools vs Object Classification)
- Drag and Drop Image uploads
- Real-time celestial object classification and Moon analysis tools
- Side Panels, which contain the metadata (resolution, confidence, crater count, brightness)

In the future, this MVP will be followed by a React/Tailwind web app for a full-production web application.

## Model Architecture
Both models currently being used use a ResNet-18 backbone for both classifiers. The models are trained from scratch or using ImageNet weights depending on dataset licensing. Custom data augmentation is used to maximize robustness, and the models are exported as a .pth file for deployment.

## Engineering Highlight
- Use PyTorch for model training and inference
- Use OpenCV for crater detection, crater count, and moon enhancement
- Code structure is modular, with script training, classifier classes, and CV utilities
- Uses a two-model decision logic
- GPU training is supported
- Caching is clean
- Lazy-loading for fast inference

## Dataset Notes Summarized
- SpaceNet Astronomy dataset (Kaggle)
- Planets and Moons dataset (Kaggle) is currently being used, although I am currently creating my own dataset for more precise + additional objects not included
- Citation list for custom dataset is included in docs, will cite all resources used in the final app
- Images and datasets are **NOT** being redistributed in this repository for licensing restrictions and ethical computing purposes.

# Happy to have you follow along with me in this journey! 🙂
