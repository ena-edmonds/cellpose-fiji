package com.cellpose;

import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.NDArray;
import org.apposed.appose.NDArray.DType;
import org.apposed.appose.NDArray.Shape;
import org.apposed.appose.NDArray.Shape.Order;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Roi;
import ij.gui.GenericDialog;
import ij.io.DirectoryChooser;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.plugin.filter.ThresholdToSelection;
import ij.plugin.frame.RoiManager;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;
import org.scijava.plugin.Menu;
import org.scijava.menu.MenuConstants;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

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
    @Menu(label = MenuConstants.ANALYZE_LABEL,
          weight = MenuConstants.ANALYZE_WEIGHT,
          mnemonic = MenuConstants.ANALYZE_MNEMONIC),
    @Menu(label = "Cellpose", weight = 100, mnemonic = 'c'),
    @Menu(label = "Batch Segment Folder", weight = 2, mnemonic = 'b')
})
public class BatchSegmentFolder implements Command {

    private static final List<String> IMAGE_EXTENSIONS =
        Arrays.asList(".tif", ".tiff", ".png", ".jpg", ".jpeg");

    // Imported at worker startup via Service.init: importing numpy/torch/cellpose in a
    // task on Windows hangs the worker (Appose #23), so pre-import them here.
    private static final String INIT_SCRIPT = """
import numpy
import torch
try:
    torch.cuda.is_available()
except Exception:
    pass
from cellpose import models
""";

    // Per-image worker: reads the (c,h,w) float32 input, writes an (h,w) int32
    // mask into the preallocated output, reports the device once via gpu_available.
    private static final String WORKER = """
import numpy as np
import torch
from cellpose import models

cuda_ok = torch.cuda.is_available()
mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
task.outputs["gpu_available"] = bool(cuda_ok or mps_ok)

if use_gpu and cuda_ok:
    device = torch.device("cuda")
elif use_gpu and mps_ok:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

img = image.ndarray()
if img.shape[0] == 1:
    img = img[0]
    ch_axis = None
else:
    ch_axis = 0

gpu = device.type != "cpu"
model = models.CellposeModel(gpu=gpu, device=device, pretrained_model=custom_model)
masks, flows, styles = model.eval(img, channel_axis=ch_axis, flow_threshold=0.0)

res = mask.ndarray()
res[:] = masks.astype("int32")
""";

    @Override
    public void run() {
        try {
            // Pick the folder.
            DirectoryChooser dc = new DirectoryChooser("Choose a folder of images to segment");
            String dirPath = dc.getDirectory();
            if (dirPath == null) return; // canceled
            File dir = new File(dirPath);

            // Gather top-level images with a standard extension.
            File[] all = dir.listFiles();
            List<File> images = new ArrayList<>();
            if (all != null) {
                for (File f : all) {
                    if (f.isFile() && hasImageExtension(f.getName())) images.add(f);
                }
            }
            if (images.isEmpty()) {
                IJ.showMessage("Batch Segment", "No images (.tif/.tiff/.png/.jpg) found in:\n" + dirPath);
                return;
            }

            // Model + GPU dialog, applied to the whole batch.
            String home = System.getProperty("user.home");
            File registry = new File(home, ".cellpose/models/gui_models.txt");
            List<String> guiModels = new ArrayList<>();
            guiModels.add("untrained");
            // Registry may not exist on a fresh machine; treat missing as empty.
            if (registry.exists()) {
                for (String line : java.nio.file.Files.readAllLines(registry.toPath())) {
                    String trimmed = line.trim();
                    if (!trimmed.isEmpty()) guiModels.add(trimmed);
                }
            }
            String[] models = guiModels.toArray(new String[0]);

            GenericDialog gd = new GenericDialog("Batch Segment Folder");
            gd.addMessage(images.size() + " image(s) found in:\n" + dirPath);
            gd.addChoice("Model", models, models[0]);
            boolean showGpuBox = GpuSupport.shouldShowCheckbox();
            if (showGpuBox) gd.addCheckbox("Use GPU", GpuSupport.defaultChecked());
            gd.addCheckbox("Skip images that already have _Rois.zip", true);
            gd.addCheckbox("Write summary CSV (image, ROI count, mean size)", true);
            gd.showDialog();
            if (gd.wasCanceled()) return;

            String model = gd.getNextChoice();
            if ("untrained".equals(model)) model = "cpsam_v2";
            boolean useGpu = false;
            if (showGpuBox) {
                useGpu = gd.getNextBoolean();
                GpuSupport.recordChoice(useGpu);
            }
            boolean skipExisting = gd.getNextBoolean();
            boolean writeCsv = gd.getNextBoolean();

            // Build the environment once for the whole batch.
            IJ.showStatus("Cellpose: preparing environment (first run may take minutes)…");
            IJ.showProgress(0.0);
            Environment env = CellposeEnv.build();

            int total = images.size();
            int processed = 0, skipped = 0, failed = 0;
            boolean recordedAvailability = false;
            // CSV rows: "image name, roi count, mean roi size (px)" for each processed image.
            List<String> summaryRows = new ArrayList<>();

            // One worker process for the whole batch.
            try (Service python = env.python().init(INIT_SCRIPT)) {
                for (int idx = 0; idx < total; idx++) {
                    File imgFile = images.get(idx);
                    String base = stripExtension(imgFile.getName());
                    File roiZip = new File(dir, base + "_Rois.zip");
                    File maskTif = new File(dir, base + "_masks.tif");

                    IJ.showStatus("Cellpose batch: " + (idx + 1) + "/" + total + " — " + imgFile.getName());
                    IJ.showProgress((double) idx / total);

                    if (skipExisting && roiZip.exists()) { skipped++; continue; }

                    ImagePlus imp = IJ.openImage(imgFile.getAbsolutePath());
                    if (imp == null) { failed++; IJ.log("Cellpose batch: could not open " + imgFile.getName()); continue; }

                    try {
                        int w = imp.getWidth(), h = imp.getHeight();
                        int n = w * h;
                        List<float[]> planes = extractChannels(imp, n);
                        int c = Math.min(planes.size(), 3);
                        if (c < 1) { failed++; IJ.log("Cellpose batch: no image data in " + imgFile.getName()); continue; }

                        Shape inShape = new Shape(Order.C_ORDER, c, h, w);
                        NDArray ndImg = new NDArray(DType.FLOAT32, inShape);
                        FloatBuffer fb = ndImg.buffer().asFloatBuffer();
                        for (int k = 0; k < c; k++) fb.put(planes.get(k), 0, n);

                        Shape outShape = new Shape(Order.C_ORDER, h, w);
                        NDArray ndMask = new NDArray(DType.INT32, outShape);

                        Map<String, Object> inputs = new HashMap<>();
                        inputs.put("image", ndImg);
                        inputs.put("mask", ndMask);
                        inputs.put("use_gpu", useGpu);
                        inputs.put("custom_model", model);

                        Service.Task task = python.task(WORKER, inputs);
                        task.waitFor();

                        // Cache GPU availability once per batch (same on every image).
                        if (!recordedAvailability) {
                            Object avail = task.outputs.get("gpu_available");
                            if (avail instanceof Boolean) {
                                GpuSupport.recordAvailability((Boolean) avail);
                                recordedAvailability = true;
                            }
                        }

                        // Build the label image and save mask + ROIs.
                        int[] maskInts = new int[n];
                        ndMask.buffer().asIntBuffer().get(maskInts);
                        FloatProcessor fp = new FloatProcessor(w, h);
                        for (int i = 0; i < n; i++) fp.setf(i, (float) maskInts[i]);
                        ImagePlus labelImp = new ImagePlus(base + "_masks", fp);

                        ndImg.close();
                        ndMask.close();

                        // Per-image stats, computed directly from the label array:
                        // ROI count = number of distinct nonzero labels present;
                        // mean size = mean pixel count over those labels.
                        if (writeCsv) {
                            int maxLabel = 0;
                            for (int v : maskInts) if (v > maxLabel) maxLabel = v;
                            long[] pixelsPerLabel = new long[maxLabel + 1];
                            for (int v : maskInts) if (v > 0) pixelsPerLabel[v]++;
                            int roiCount = 0;
                            long totalLabeledPixels = 0;
                            for (int label = 1; label <= maxLabel; label++) {
                                if (pixelsPerLabel[label] > 0) {
                                    roiCount++;
                                    totalLabeledPixels += pixelsPerLabel[label];
                                }
                            }
                            double meanSizePx = roiCount > 0 ? (double) totalLabeledPixels / roiCount : 0.0;
                            summaryRows.add(String.format(Locale.ROOT, "%s,%d,%.2f",
                                csvField(imgFile.getName()), roiCount, meanSizePx));
                        }

                        // Save the label mask as TIFF.
                        IJ.saveAsTiff(labelImp, maskTif.getAbsolutePath());

                        // Derive ROIs and save as a RoiSet .zip.
                        RoiManager rm = deriveRois(labelImp);
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
            if (writeCsv && !summaryRows.isEmpty()) {
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

    // ============================ helpers ============================

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

    private List<float[]> extractChannels(ImagePlus imp, int n) {
        List<float[]> planes = new ArrayList<>();
        if (imp.getType() == ImagePlus.COLOR_RGB) {
            ColorProcessor cp = (ColorProcessor) imp.getProcessor();
            byte[] r = new byte[n], g = new byte[n], b = new byte[n];
            cp.getRGB(r, g, b);
            planes.add(toFloat(r)); planes.add(toFloat(g)); planes.add(toFloat(b));
        } else {
            ImageStack st = imp.getStack();
            for (int i = 1; i <= st.getSize(); i++)
                planes.add((float[]) st.getProcessor(i).convertToFloat().getPixels());
        }
        return planes;
    }

    private float[] toFloat(byte[] arr) {
        float[] out = new float[arr.length];
        for (int i = 0; i < arr.length; i++) out[i] = (float)(arr[i] & 0xFF);
        return out;
    }

    // Convert a label image into named ROIs in a (hidden) RoiManager.
    private RoiManager deriveRois(ImagePlus labelImp) {
        ImageProcessor ip = labelImp.getProcessor();
        int max = (int) ip.getMax();
        if (max < 1) return null;
        RoiManager rm = RoiManager.getInstance();
        if (rm == null) rm = new RoiManager(false); // false = don't show the window during batch
        rm.reset();
        ThresholdToSelection tts = new ThresholdToSelection();
        for (int label = 1; label <= max; label++) {
            ip.setThreshold(label, label, ImageProcessor.NO_LUT_UPDATE);
            Roi roi = tts.convert(ip);
            if (roi != null) { roi.setName("Cell_" + label); rm.addRoi(roi); }
        }
        ip.resetThreshold();
        return rm;
    }
}
