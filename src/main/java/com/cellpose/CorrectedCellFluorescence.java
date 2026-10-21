package com.cellpose;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.Roi;
import ij.gui.ShapeRoi;
import ij.measure.Measurements;
import ij.measure.ResultsTable;
import ij.plugin.RoiEnlarger;
import ij.plugin.frame.RoiManager;
import ij.process.ImageProcessor;
import ij.process.ImageStatistics;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import java.awt.Color;
import java.awt.Rectangle;
import java.util.*;

/**
 * CTCF = Corrected Total Cell Fluorescence
 * Calculates CTCF for every ROI in the ROI Manager, using an automatically
 * generated background band offset from the merged NMJ outline. Bright
 * obstructions in the band are rejected as outliers before averaging.
 */
@Plugin(type = Command.class, menuPath = "Process>Get ROIs CTCF")
public class CorrectedCellFluorescence implements Command {

    private static final int GAP = 5;            // px between NMJ edge and band
    private static final int THICKNESS = 15;     // px width of background band
    private static final double REJECT_SD = 3;   // outlier cutoff in robust SDs
    private static final double WARN_PCT = 50;   // log a warning below this % of band used
    private static final boolean HIDE_BAND = true; // keep band in ROI Manager but draw it invisible
    private static final String BG_PREFIX = "bg";

    @Override
    public void run() {
        ImagePlus imp = WindowManager.getCurrentImage();
        if (imp == null) {
            IJ.error("CTCF", "No image open.");
            return;
        }
        RoiManager rm = RoiManager.getInstance();
        if (rm == null || rm.getCount() == 0) {
            IJ.error("CTCF", "Add your NMJ ROIs to the ROI Manager first.");
            return;
        }

        // Replace the background band from any previous run
        for (int i = rm.getCount() - 1; i >= 0; i--) {
            String name = rm.getName(i);
            if (name != null && name.startsWith(BG_PREFIX)) {
                rm.select(i);
                rm.runCommand("Delete");
            }
        }

        // Collect area ROIs (skip lines/points and background ROIs)
        List<Roi> cells = new ArrayList<>();
        for (Roi r : rm.getRoisAsArray()) {
            String name = r.getName();
            if (r.isArea() && (name == null || !name.startsWith(BG_PREFIX))) {
                cells.add(r);
            }
        }
        if (cells.isEmpty()) {
            IJ.error("CTCF", "No area ROIs found.");
            return;
        }

        // 1. Merge copies of all ROIs into one overall NMJ ROI
        ShapeRoi merged = new ShapeRoi(cells.get(0));
        for (int i = 1; i < cells.size(); i++) {
            merged.or(new ShapeRoi(cells.get(i)));
        }

        // 2. Background = ring THICKNESS px wide, starting GAP px
        //    outside the merged ROI, clipped to the image bounds
        Roi inner = RoiEnlarger.enlarge(merged, GAP);
        Roi outer = RoiEnlarger.enlarge(merged, GAP + THICKNESS);
        ShapeRoi band = new ShapeRoi(outer).not(new ShapeRoi(inner));
        band.and(new ShapeRoi(new Roi(0, 0, imp.getWidth(), imp.getHeight())));
        band.setName(BG_PREFIX + "_band");
        if (HIDE_BAND) band.setStrokeColor(new Color(0, 0, 0, 0));
        rm.addRoi(band);

        // 3. Background mean from the band with bright obstructions rejected
        double[] bg = robustBackground(imp.getProcessor(), band);
        double bgMean = bg[0];
        double bgPctUsed = bg[2] > 0 ? 100.0 * bg[1] / bg[2] : 0;
        if (Double.isNaN(bgMean)) {
            IJ.error("CTCF", "Background band is empty (NMJ may fill the image).");
            return;
        }
        if (bgPctUsed < WARN_PCT) {
            IJ.log(imp.getTitle() + ": only " + IJ.d2s(bgPctUsed, 1)
                    + "% of background band used - check for obstructions");
        }

        // 4. CTCF for each ROI using the robust background mean
        int opts = Measurements.AREA | Measurements.MEAN;
        ResultsTable rt = new ResultsTable();
        for (int i = 0; i < cells.size(); i++) {
            Roi r = cells.get(i);
            imp.setRoi(r);
            ImageStatistics s = imp.getStatistics(opts);
            double intDen = s.area * s.mean;
            String name = r.getName() != null ? r.getName() : "ROI " + (i + 1);

            rt.incrementCounter();
            rt.addValue("ROI", name);
            rt.addValue("Area (px)", s.area);
            rt.addValue("Mean", s.mean);
            //rt.addValue("IntDen", intDen);
            //rt.addValue("BG Mean", bgMean);
            //rt.addValue("BG % used", bgPctUsed);
            rt.addValue("CTCF", intDen - s.area * bgMean);
        }

        // Leave the image with no active selection
        imp.deleteRoi();
        rm.deselect();
        rt.show("CTCF Results");
    }

    /**
     * Mean of the ROI's pixels after iteratively rejecting outliers
     * (> REJECT_SD robust SDs from the median).
     * Returns {mean, pixelsUsed, pixelsTotal}.
     */
    private static double[] robustBackground(ImageProcessor ip, Roi roi) {
        Rectangle b = roi.getBounds();
        ImageProcessor mask = roi.getMask();
        double[] vals = new double[b.width * b.height];
        int n = 0;
        for (int y = 0; y < b.height; y++) {
            for (int x = 0; x < b.width; x++) {
                if (mask == null || mask.get(x, y) != 0) {
                    vals[n++] = ip.getPixelValue(b.x + x, b.y + y);
                }
            }
        }
        if (n == 0) return new double[] {Double.NaN, 0, 0};
        vals = Arrays.copyOf(vals, n);

        double[] kept = vals;
        for (int iter = 0; iter < 10; iter++) {
            double med = median(kept);
            double[] dev = new double[kept.length];
            for (int i = 0; i < kept.length; i++) dev[i] = Math.abs(kept[i] - med);
            double sigma = 1.4826 * median(dev);
            if (sigma == 0) break;

            double lo = med - REJECT_SD * sigma;
            double hi = med + REJECT_SD * sigma;
            double[] next = Arrays.stream(vals).filter(v -> v >= lo && v <= hi).toArray();
            if (next.length == kept.length || next.length == 0) break;
            kept = next;
        }

        double sum = 0;
        for (double v : kept) sum += v;
        return new double[] {sum / kept.length, kept.length, n};
    }

    private static double median(double[] a) {
        double[] s = a.clone();
        Arrays.sort(s);
        int m = s.length / 2;
        return (s.length % 2 == 0) ? (s[m - 1] + s[m]) / 2 : s[m];
    }
}