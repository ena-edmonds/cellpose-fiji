package com.cellpose;

import org.apposed.appose.Service;
import org.apposed.appose.NDArray;

import ij.IJ;
import ij.ImagePlus;

import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.plugin.Menu;
import org.scijava.menu.MenuConstants;

import java.util.*;

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
            if(settings.wasCanceled()) return;
            
            // prepare image for the run
            IJ.showStatus("Cellpose: preparing image…");
            IJ.showProgress(0.05);
            ProcessImage processor = new ProcessImage(imp);
            NDArray ndImg = processor.getNdImg(); // input: channel-first (c, h, w) float32
            NDArray ndMask = processor.getNdMask(); // output: pre-allocated (h, w) int32 label image
            if (ndImg == null) { IJ.showProgress(1.0); throw new IllegalArgumentException("No image data found."); }
            
            // build the Cellpose environment (slow on first run of any part of the plugin)
            IJ.showStatus("Cellpose: preparing environment (first run may take minutes)…");
            IJ.showProgress(0.15);
            try (Service python = CellposeEnv.init()) {
                Map<String, Object> inputs = new HashMap<>();
                inputs.put("image", ndImg);
                inputs.put("mask", ndMask);
                inputs.putAll(settings.getChoices());

                Thread busy = startBusyAnimation(); // Animate an indeterminate "loading" bar during segmentation

                Service.Task task = CellposeEnv.run(python, inputs); // segment the image!

                // Forward worker stage updates to the Fiji status bar
                task.listen(e -> {
                    if (e.message != null) {
                        IJ.showStatus("Cellpose: " + e.message + "…");
                        if (e.maximum > 0)
                            IJ.showProgress((double) e.current / (double) e.maximum);
                    }
                });
                task.waitFor();
                busy.interrupt(); // stop the busy animation
            }

            // --- Build a label image in memory, used only to derive ROIs ---
            IJ.showStatus("Cellpose: building label image…");
            IJ.showProgress(0.9);
            processor.readMask(ndMask);
            ndImg.close();
            ndMask.close();

            ProcessImage.deriveRois(processor.getLabelImp(),false).runCommand("Show all");

            IJ.showStatus("Cellpose: done");
            IJ.showProgress(1.0);
        } catch (Exception e) {
            IJ.showProgress(1.0);
            IJ.handleException(e);
        }
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
