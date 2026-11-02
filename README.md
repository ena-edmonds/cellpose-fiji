# cellpose-fiji
Repository for my ImageJ2/Fiji plugin to implement [Cellpose 4.2.1.1](https://github.com/MouseLand/cellpose) access with custom-trained models via [Appose](https://github.com/apposed/appose)

AI DISCLAIMER: Code for this plugin was written with partial help from Dia browser using Claude Opus 4.8

Initial version made as part of the 2026 STARS Summer program at Simmons University.

# installation
1. Download and install the plugin using normal means. Drag the cellpose-fiji-x.x.x.jar file into the "plugins" Fiji folder.
2. To add a custom model, go to "Process" -> "Cellpose" -> "Add Model" in the top menus. A file explorer window will appear, use it to select a previously trained model file output using Cellpose training tools. Expect a short wait while the plugin generates its Appose environment.

# useage
Cellpose-Fiji comes with several different tools for processing data in Fiji.

## segmentation
With an image open, go to "Process" -> "Cellpose" -> "Segment Image". A dialogue will appear with a dropdown menu to select from a list of models added using the "Add Model" option, along with the default untrained cpsam_v2 model. Select a model, choose whether or not to allow Cellpose to use the GPU to run (Recommended! Cellpose runs very slowly without a GPU!) the segmentation algorithm, and press "Ok" to begin the run. The progress of the run will be displayed in the status bar.

Advanced settings are also available, selecting "Advanced Mode" and clicking "Ok" will open a second dialogue with extra parameters for the Cellpose run. Settings appear as their defaults used in non-advanced mode. When the model run is complete it will open an ROI manager window with the segmentations output by Cellpose.

An entire folder of images can also be segmented at once. Select "Batch Segment Folder" from the "Process -> Cellpose" menus and select a folder of images with .png, .jpg. .tif, .tiff, or .jpeg extensions. The run will begin, any previously segmented images with matching "_rois.zip" will be skipped. At the end of the run, a set of "_rois.zip" files will be output to the same folder along with a summary .csv with status on all images.

## corrected fluorescence
After segmentation produces a set of ROIs, corrected fluorescence can be calculated using this plugin. With a single image and its ROIs open in Fiji, select "Process" -> "Calculate Fluorescence". The plugin will calculate the corrected fluorescence and output a new window with stats for each ROI.

This function can also be performed on a folder with many images. After processing the images using Batch Segmentation, go to "Process" -> "Batch" -> "Calculate Fluorescence (Folder)" and select a folder of images. CSV files with the corrected fluorescence will be output for each image in the folder with a matching "_rois.zip" file.

## batch conversion of .czi files
Files output from ZEISS microscopes can come in the form of .czi stack images, which are 3D. To use with Cellpose available within this plugin, these images must be flattened into a single plane. A max-intensity Z-projection can be performed on an entire folder of .czi files using this plugin. Go to "Process" -> "Batch" -> "Flatten Stacks" and select a folder of .czi files. The plugin will leave .czi files and output in the same folder flattened images with _MAX.png extensions.