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
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.plugin.filter.ThresholdToSelection;
import ij.plugin.frame.RoiManager;

import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.plugin.Menu;
import org.scijava.menu.MenuConstants;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Cellpose segmentation, ported from the segment_image script.
 * <p>Menu position is controlled by the trailing {@code @Menu} weight
 * (1 = top of the Cellpose submenu).
 */
@Plugin(type = Command.class, menu = {
    @Menu(label = MenuConstants.PROCESS_LABEL,
          weight = MenuConstants.PROCESS_WEIGHT,
          mnemonic = MenuConstants.PROCESS_MNEMONIC),
    @Menu(label = "Cellpose", weight = 100, mnemonic = 'c'),
    @Menu(label = "Segment Image", weight = 1, mnemonic = 's')
})
public class SegmentImage implements Command {

    @Parameter
    private ImagePlus imp;

    @Override
    public void run() {
        try {
            SettingsManager settings = new SettingsManager();
            settings.collectSettings(); // show dialogue to get run params from user
            
            // start prepping image for the run
            int w = imp.getWidth(), h = imp.getHeight();
            int n = w * h;

            IJ.showStatus("Cellpose: preparing image…");
            IJ.showProgress(0.05);

            // extract channels from any source: grayscale, RGB PNG, or multi-plane stack
            List<float[]> planes = extractChannels(imp, n);
            int c = Math.min(planes.size(), 3);
            if (c < 1) { IJ.showProgress(1.0); throw new IllegalArgumentException("No image data found."); }

            // build the Cellpose environment (slow only on first run of any part of the plugin)
            IJ.showStatus("Cellpose: preparing environment (first run may take minutes)…");
            IJ.showProgress(0.15);
            Environment env = CellposeEnv.build();

            // --- Input: channel-first (c, h, w) float32 ---
            Shape inShape = new Shape(Order.C_ORDER, c, h, w);
            NDArray ndImg = new NDArray(DType.FLOAT32, inShape);
            FloatBuffer fb = ndImg.buffer().asFloatBuffer();
            for (int k = 0; k < c; k++) fb.put(planes.get(k), 0, n);

            // --- Output: pre-allocated (h, w) int32 label image ---
            Shape outShape = new Shape(Order.C_ORDER, h, w);
            NDArray ndMask = new NDArray(DType.INT32, outShape);

            // --- Python: emits stage updates via task.update(...) ---
            // Device selection respects the "Use GPU" checkbox. numpy/torch/cellpose are
            // imported in the .init script at worker startup — importing them in-task hangs
            // the worker on Windows (Appose #23), so they must be pre-imported.
            String script = """
import numpy as np
import torch
from cellpose import models

task.update("Loading model", current=1, maximum=3)

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

if cell_diam == 0:
    dia = None
else:
    dia = cell_diam

gpu = device.type != "cpu"
model = models.CellposeModel(gpu=gpu, device=device, pretrained_model=custom_model)

task.update("Segmenting (" + device.type + ")", current=2, maximum=3)
masks, flows, styles = model.eval(img, channel_axis=ch_axis, flow_threshold=flow_thresh, diameter=dia, min_size=min_cell_size, cellprob_threshold=cell_prob)

task.update("Writing masks", current=3, maximum=3)
res = mask.ndarray()
res[:] = masks.astype("int32")
""";

            // Animate an indeterminate "busy" bar during the long segmentation step
            Thread busy = startBusyAnimation();

            // .init imports numpy/torch/cellpose at worker startup. Appose warns that
            // importing numpy in a task on Windows hangs (appose #23); the init script is
            // the sanctioned fix. Must be valid Python with real newlines (a text block).
            String initScript = """
import numpy
import torch
try:
    torch.cuda.is_available()
except Exception:
    pass
from cellpose import models
""";
            try (Service python = env.python().init(initScript)) {
                Map<String, Object> inputs = new HashMap<>();
                inputs.put("image", ndImg);
                inputs.put("mask", ndMask);
                inputs.put("use_gpu", settings.getGpuChoice());
                inputs.put("custom_model", settings.getModel());
                inputs.put("cell_diam", settings.getCellDiameter());
                inputs.put("flow_thresh", settings.getFlowThreshold());
                inputs.put("cell_prob", settings.getCellProbThreshold());
                inputs.put("min_cell_size", settings.getMinSize());
                Service.Task task = python.task(script, inputs);
                // Forward worker stage updates to the Fiji status bar
                task.listen(e -> {
                    if (e.message != null) {
                        IJ.showStatus("Cellpose: " + e.message + "…");
                        if (e.maximum > 0)
                            IJ.showProgress((double) e.current / (double) e.maximum);
                    }
                });
                task.waitFor();

                // Cache whether this machine actually has a usable GPU, so future runs
                // can show/hide the checkbox correctly (see GpuSupport).
                Object avail = task.outputs.get("gpu_available");
                if (avail instanceof Boolean) {
                    GpuSupport.recordAvailability((Boolean) avail);
                }
            }

            busy.interrupt();    // stop the animation

            // --- Build a label image in memory, used only to derive ROIs ---
            IJ.showStatus("Cellpose: building label image…");
            IJ.showProgress(0.9);
            int[] maskInts = new int[n];
            ndMask.buffer().asIntBuffer().get(maskInts);
            FloatProcessor fp = new FloatProcessor(w, h);
            for (int i = 0; i < n; i++) fp.setf(i, (float) maskInts[i]);
            ImagePlus labelImp = new ImagePlus("Cellpose labels", fp);

            ndImg.close();
            ndMask.close();

            // --- Output: ROIs only (the label image is not displayed) ---
            IJ.showStatus("Cellpose: creating ROIs…");
            addLabelsToRoiManager(labelImp);

            IJ.showStatus("Cellpose: done");
            IJ.showProgress(1.0);
        } catch (Exception e) {
            IJ.showProgress(1.0);
            IJ.handleException(e);
        }
    }

    // ============================ helper methods ============================

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

    private RoiManager addLabelsToRoiManager(ImagePlus labelImp) {
        ImageProcessor ip = labelImp.getProcessor();
        int max = (int) ip.getMax();
        if (max < 1) return null;
        RoiManager rm = RoiManager.getInstance();
        if (rm == null) rm = new RoiManager();
        rm.reset();
        ThresholdToSelection tts = new ThresholdToSelection();
        for (int label = 1; label <= max; label++) {
            ip.setThreshold(label, label, ImageProcessor.NO_LUT_UPDATE);
            Roi roi = tts.convert(ip);
            if (roi != null) { roi.setName("Cell_" + label); rm.addRoi(roi); }
        }
        ip.resetThreshold();
        rm.runCommand("Show All");
        return rm;
    }

    // Indeterminate progress: cycles the Fiji progress bar while segmentation runs
    private Thread startBusyAnimation() {
        Thread t = new Thread(() -> {
            double p = 0.4;
            try {
                while (!Thread.currentThread().isInterrupted()) {
                    p += 0.02;
                    if (p > 0.85) p = 0.4;
                    IJ.showProgress(p);
                    Thread.sleep(150);
                }
            } catch (InterruptedException ignored) {}
        });
        t.setDaemon(true);
        t.start();
        return t;
    }
}
