package com.cellpose;

import org.apposed.appose.Service;
import org.apposed.appose.NDArray;

import ij.IJ;
import ij.ImagePlus;

import ij.io.DirectoryChooser;
import ij.plugin.frame.RoiManager;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;
import org.scijava.plugin.Menu;
import org.scijava.menu.MenuConstants;

import java.io.File;
import java.util.*;

/**
 * Batch Cellpose segmentation over a folder of images.
 *
 * <p>Points at a folder, runs Cellpose on every top-level image with a standard
 * extension, and saves per-image outputs next to each image:
 * <ul>
 *   <li>{@code {name}_Rois.zip} — an ImageJ RoiSet (reopens via the ROI Manager)</li>
 *   <li>{@code {name}_masks.tif} — the integer label image</li>
 * </ul>
 * Images whose {@code _Rois.zip} already exists are skipped, so the batch is
 * resumable.
 *
 * <p>The Appose environment is built once and a single Python worker processes
 * the whole batch; images are opened and channel-extracted in Java (reusing the
 * same logic as {@link SegmentImage}) and handed to the worker one at a time.
 */
@Plugin(type = Command.class, menu = {
    @Menu(label = MenuConstants.PROCESS_LABEL,
          weight = MenuConstants.PROCESS_WEIGHT,
          mnemonic = MenuConstants.PROCESS_MNEMONIC),
    @Menu(label = "Cellpose", weight = 100, mnemonic = 'c'),
    @Menu(label = "Batch Segment Folder", weight = 2, mnemonic = 'b')
})
public class BatchSegmentFolder implements Command {

    private static final List<String> IMAGE_EXTENSIONS = Arrays.asList(".tif", ".tiff", ".png", ".jpg", ".jpeg");

    @Override
    public void run() {
        try {
            // Pick the folder
            DirectoryChooser dc = new DirectoryChooser("Choose a folder of images to segment");
            String dirPath = dc.getDirectory();
            if(dirPath == null) return; // canceled
            File dir = new File(dirPath);

            // Gather images in selected folder (no sub-folders) with valid extensions
            File[] all = dir.listFiles();
            List<File> images = new ArrayList<>();
            if(all != null) {
                for(File f : all) if(f.isFile() && hasImageExtension(f.getName())) images.add(f);
            }
            if(images.isEmpty()) {
                IJ.showMessage("Batch Segment", "No images (.tif/.tiff/.png/.jpg) found in:\n" + dirPath);
                return;
            }

            // Get run settings from user
            SettingsManager settings = new SettingsManager();
            settings.collectSettings();
            if(settings.wasCanceled()) return;

            // Set up variables for tracking progress and final output
            int total = images.size();
            int processed = 0, skipped = 0, failed = 0;
            List<String> summaryRows = new ArrayList<>(); // "image name, roi count, mean roi size (px)" for each image

            // Try building and initializing the environment
            IJ.showStatus("Cellpose: preparing environment (first run may take minutes)…");
            IJ.showProgress(0.0);
            try (Service python = CellposeEnv.init()) {
                for (int idx = 0; idx < total; idx++) {
                    File imgFile = images.get(idx);
                    File roiZip = new File(dir, stripExtension(imgFile.getName()) + "_Rois.zip");

                    IJ.showStatus("Cellpose batch: " + (idx + 1) + "/" + total + " — " + imgFile.getName());
                    IJ.showProgress((double) idx / total);

                    if(roiZip.exists()) { skipped++; continue; }

                    ImagePlus imp = IJ.openImage(imgFile.getAbsolutePath());
                    if (imp == null) { failed++; IJ.log("Cellpose batch: could not open " + imgFile.getName()); continue; }

                    try {
                        ProcessImage processor = new ProcessImage(imp);
                        NDArray ndImg = processor.getNdImg();
                        NDArray ndMask = processor.getNdMask();
                        if (ndImg == null) { failed++; IJ.log("Cellpose batch: no image data in " + imgFile.getName()); continue; }

                        Map<String, Object> inputs = new HashMap<>();
                        inputs.put("image", ndImg);
                        inputs.put("mask", ndMask);
                        inputs.putAll(settings.getChoices());

                        Service.Task task = CellposeEnv.run(python, inputs); // segment the image!
                        task.waitFor();

                        // Build the label image and save mask + ROIs.
                        processor.readMask(ndMask);
                        ImagePlus labelImp = processor.getLabelImp();

                        ndImg.close();
                        ndMask.close();

                        // Per-image stats, computed directly from the label array:
                        // ROI count = number of distinct nonzero labels present;
                        // mean size = mean pixel count over those labels.
                        int maxLabel = 0;
                        for (int v : processor.getMaskInts()) if (v > maxLabel) maxLabel = v;
                        long[] pixelsPerLabel = new long[maxLabel + 1];
                        for (int v : processor.getMaskInts()) if (v > 0) pixelsPerLabel[v]++;
                        int roiCount = 0;
                        long totalLabeledPixels = 0;
                        for (int label = 1; label <= maxLabel; label++) {
                            if (pixelsPerLabel[label] > 0) {
                                roiCount++;
                                totalLabeledPixels += pixelsPerLabel[label];
                            }
                        }
                        double meanSizePx = roiCount > 0 ? (double) totalLabeledPixels / roiCount : 0.0;
                        summaryRows.add(String.format(Locale.ROOT, "%s,%d,%.2f", csvField(imgFile.getName()), roiCount, meanSizePx));

                        // Derive ROIs and save as a RoiSet .zip.
                        RoiManager rm = ProcessImage.deriveRois(labelImp, true);
                        if (rm != null && rm.getCount() > 0) {
                            rm.runCommand("Save", roiZip.getAbsolutePath());
                            rm.reset();
                        }
                        processed++;
                    } finally {
                        imp.close(); // don't accumulate open windows across the batch
                    }
                }
            }

            IJ.showProgress(1.0);
            IJ.showStatus("Cellpose batch: done");

            // Write one summary CSV for the whole batch.
            if (!summaryRows.isEmpty()) {
                File csv = new File(dir, "cellpose_batch_summary.csv");
                StringBuilder sb = new StringBuilder();
                sb.append("image,roi_count,mean_roi_size_px\n");
                for (String row : summaryRows) sb.append(row).append('\n');
                java.nio.file.Files.write(csv.toPath(),
                    sb.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
                IJ.log("Cellpose batch: wrote summary " + csv.getAbsolutePath());
            }

            IJ.log(String.format(Locale.ROOT,
                "Cellpose batch complete: %d processed, %d skipped, %d failed (of %d) in %s",
                processed, skipped, failed, total, dirPath));
        } catch (Exception e) {
            IJ.showProgress(1.0);
            IJ.handleException(e);
        }
    }

    // helper functions for batch mode

    /** Quote a CSV field if it contains a comma, quote, or newline. */
    private static String csvField(String s) {
        if (s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\n') >= 0) {
            return '"' + s.replace("\"", "\"\"") + '"';
        }
        return s;
    }

    private static boolean hasImageExtension(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        for (String ext : IMAGE_EXTENSIONS) if (lower.endsWith(ext)) return true;
        return false;
    }

    private static String stripExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }
}
