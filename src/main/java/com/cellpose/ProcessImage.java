package com.cellpose;

import org.apposed.appose.NDArray;
import org.apposed.appose.NDArray.*;
import org.apposed.appose.NDArray.Shape.Order;

import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Roi;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;
import ij.plugin.filter.ThresholdToSelection;
import ij.plugin.frame.RoiManager;
import ij.process.FloatProcessor;

import java.util.ArrayList;
import java.util.List;
import java.nio.FloatBuffer;


public class ProcessImage {
    private ImagePlus imp;
    private int w;
    private int h;
    private int n;
    private int[] maskInts;

    public ProcessImage(ImagePlus i) {
        imp = i;
        w = imp.getWidth();
        h = imp.getHeight();
        n = w * h;
        maskInts = new int[n];
    }
    
    public static List<float[]> extractChannels(ImagePlus imp, int n) {
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

    public static float[] toFloat(byte[] arr) {
        float[] out = new float[arr.length];
        for (int i = 0; i < arr.length; i++) out[i] = (float)(arr[i] & 0xFF);
        return out;
    }

    // Convert a label image into named ROIs in an RoiManager.
    public static RoiManager deriveRois(ImagePlus labelImp, boolean silent) {
        ImageProcessor ip = labelImp.getProcessor();
        int max = (int) ip.getMax();
        if (max < 1) return null;
        RoiManager rm = RoiManager.getInstance();
        if(rm == null) rm = (silent) ? new RoiManager(silent) : new RoiManager();
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

    public NDArray getNdImg() {
        List<float[]> planes = extractChannels(imp, n);
        int c = Math.min(planes.size(), 3);
        if (c < 1) { return null; } // failed
        Shape inShape = new Shape(Order.C_ORDER, c, h, w);
        NDArray ndImg = new NDArray(DType.FLOAT32, inShape);
        FloatBuffer fb = ndImg.buffer().asFloatBuffer();
        for (int k = 0; k < c; k++) fb.put(planes.get(k), 0, n);
        return ndImg;
    }

    public NDArray getNdMask() {
        int w = imp.getWidth(), h = imp.getHeight();
        Shape outShape = new Shape(Order.C_ORDER, h, w);
        return new NDArray(DType.INT32, outShape);
    }

    public ImagePlus getLabelImp() {
        FloatProcessor fp = new FloatProcessor(w, h);
        for (int i = 0; i < n; i++) fp.setf(i, (float) maskInts[i]);
        return new ImagePlus("Cellpose labels", fp);
    }

    public int[] getMaskInts() {return maskInts;}

    public void readMask(NDArray ndMask) {
        ndMask.buffer().asIntBuffer().get(maskInts);
    }
}
