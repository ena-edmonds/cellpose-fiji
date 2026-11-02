package com.cellpose;

import java.io.*;
import java.util.*;
import ij.IJ;
import ij.Prefs;
import ij.gui.GenericDialog;

public class SettingsManager {
    private static final String GPU_LAST_CHOICE = "cellpose.useGpu";

    private String model;
    private boolean useGpu;
    private boolean canceled;

    private int cell_diameter;
    private int min_cell_size;
    private double cellprob_threshold;
    private double flow_threshold;

    public SettingsManager() {
        cell_diameter = 0;
        min_cell_size = 15;
        cellprob_threshold = 0.0;
        flow_threshold = 0.0;
    }

    public void collectSettings() {
        canceled = false;
        try {
            // Find the text file Cellpose uses to track imported custom models
            File modelsList = new File(System.getProperty("user.home"), ".cellpose/models/gui_models.txt");
            List<String> guiModels = new ArrayList<>();
            guiModels.add("untrained"); // add an option for the default untrained cpsam model

            // Cellpose might not have made the models list yet, treat it as an empty array if true
            if (modelsList.exists()) {
                for (String line : java.nio.file.Files.readAllLines(modelsList.toPath())) {
                    String trimmed = line.trim();
                    if (!trimmed.isEmpty()) guiModels.add(trimmed); // skip blank lines
                }
            }
            String[] models = guiModels.toArray(new String[0]);

            // Populate model run settings dialogue
            GenericDialog gd = new GenericDialog("Segment Image");
            gd.addChoice("Model", models, models[0]);
            gd.addCheckbox("Use GPU", Prefs.get(GPU_LAST_CHOICE, false));
            gd.addCheckbox("Advanced mode", false);

            gd.showDialog();
            if (gd.wasCanceled()) {canceled = true; return;}

            model = gd.getNextChoice();
            if (model.equals("untrained")) model = "cpsam_v2";
            useGpu = gd.getNextBoolean();
            Prefs.set(GPU_LAST_CHOICE, Boolean.toString(useGpu));
            Prefs.savePreferences();
            if(gd.getNextBoolean()) collectAdvancedSettings();

        } catch(Exception e) {
            IJ.handleException(e);
        }
    }

    public void collectAdvancedSettings() {
        // Populate advanced settings dialogue with extra settings
        GenericDialog advDialog = new GenericDialog("Advanced Settings");
        advDialog.addNumericField("Cell diameter (px)", 0);
        advDialog.addSlider("Flow threshold", 0.0, 3.0, 0.0, 0.1);
        advDialog.addSlider("Cellprob threshold", -6.0, 6.0, 0.0, 0.1);
        advDialog.addNumericField("Min ROI size (px)", 15);
        
        advDialog.showDialog();
        if (advDialog.wasCanceled()) {canceled = true; return;}

        // Reassign params affected by advanced settings
        cell_diameter = (int) advDialog.getNextNumber();
        flow_threshold = advDialog.getNextNumber();
        cellprob_threshold = advDialog.getNextNumber();
        min_cell_size = (int) advDialog.getNextNumber();
    }

    // getters for the various parameters
    public boolean wasCanceled() {return canceled;}

    public Map<String, Object> getChoices() {
        Map<String, Object> settings = new HashMap<>();
        settings.put("use_gpu", useGpu);
        settings.put("custom_model", model);
        settings.put("cell_diam", cell_diameter);
        settings.put("flow_thresh", flow_threshold);
        settings.put("cell_prob", cellprob_threshold);
        settings.put("min_cell_size", min_cell_size);
        return settings;
    }
}
