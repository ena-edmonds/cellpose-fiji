package com.cellpose;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.gui.ShapeRoi;
import ij.io.RoiDecoder;
import ij.measure.ResultsTable;
import ij.io.DirectoryChooser;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip. ZipInputStream;

/**
 * Batch CTCF: for every NAME_ROIs.zip in a folder, opens the matching image
 * (NAME.<any extension>), runs the same CTCF analysis as CorrectedCellFluorescence,
 * and saves NAME_CTCF.csv to the same folder. Input files are not modified.
 */
@Plugin(type = Command.class, menuPath = "Process>Batch>Calculate Fluorescence (Folder)")
public class BatchCTCF implements Command {

    private static final String ROI_SUFFIX = "_Rois.zip";
    private static final String CSV_SUFFIX = "_CTCF.csv";
    private static final Set<String> NON_IMAGE_EXT =
            new HashSet<>(Arrays.asList("zip", "csv", "txt", "roi", "json", "npy", "md", "log"));

    private int channel = 1;

    @Override
    public void run() {
        // Pick the folder
        DirectoryChooser dc = new DirectoryChooser("Choose a folder");
        String dirPath = dc.getDirectory();
        if(dirPath == null) return; // canceled
        File folder = new File(dirPath);

        File[] files = folder.listFiles();
        if (files == null) {
            IJ.error("CTCF Batch", "Can't read folder: " + folder);
            return;
        }
        Arrays.sort(files);

        // Index images by base name (file name without extension)
        Map<String, File> images = new HashMap<>();
        for (File f : files) {
            String name = f.getName();
            int dot = name.lastIndexOf('.');
            if (f.isDirectory() || name.startsWith(".") || dot < 0) continue;
            if (NON_IMAGE_EXT.contains(name.substring(dot + 1).toLowerCase())) continue;
            String base = name.substring(0, dot);
            if (images.containsKey(base)) {
                IJ.log("Multiple images named " + base + ".* - using " + images.get(base).getName());
            } else {
                images.put(base, f);
            }
        }

        List<File> zips = new ArrayList<>();
        for (File f : files) {
            if (f.getName().endsWith(ROI_SUFFIX)) zips.add(f);
        }
        if (zips.isEmpty()) {
            IJ.error("CTCF Batch", "No *" + ROI_SUFFIX + " files found in " + folder);
            return;
        }

        int done = 0;
        for (int i = 0; i < zips.size(); i++) {
            File zip = zips.get(i);
            String base = zip.getName().substring(0, zip.getName().length() - ROI_SUFFIX.length());
            IJ.showStatus("CTCF batch: " + base + " (" + (i + 1) + "/" + zips.size() + ")");
            IJ.showProgress(i, zips.size());

            File imageFile = images.get(base);
            if (imageFile == null) {
                IJ.log("Skipped " + zip.getName() + ": no matching image " + base + ".*");
                continue;
            }

            ImagePlus imp = IJ.openImage(imageFile.getPath());
            if (imp == null) {
                IJ.log("Skipped " + base + ": could not open " + imageFile.getName());
                continue;
            }
            try {
                if (channel > imp.getNChannels()) {
                    IJ.log("Skipped " + base + ": image has only " + imp.getNChannels() + " channel(s)");
                    continue;
                }
                imp.setPosition(channel, 1, 1);

                List<Roi> cells = CorrectedCellFluorescence.cellRois(loadRois(zip));
                if (cells.isEmpty()) {
                    IJ.log("Skipped " + base + ": no area ROIs in " + zip.getName());
                    continue;
                }

                ShapeRoi band = CorrectedCellFluorescence.makeBand(cells, imp);
                ResultsTable rt = CorrectedCellFluorescence.measure(imp, cells, band);
                if (rt == null) {
                    IJ.log("Skipped " + base + ": background band is empty");
                    continue;
                }

                File csv = new File(folder, base + CSV_SUFFIX);
                if (rt.save(csv.getPath())) {
                    done++;
                } else {
                    IJ.log("Failed to save " + csv.getName());
                }
            } catch (IOException e) {
                IJ.log("Skipped " + base + ": could not read " + zip.getName() + " (" + e.getMessage() + ")");
            } finally {
                imp.flush();
            }
        }

        IJ.showProgress(1.0);
        IJ.showStatus("");
        IJ.log("CTCF batch: saved " + done + " of " + zips.size() + " CSV(s) to " + folder);
    }

    /** Reads all .roi entries from an ROI Manager zip without opening the ROI Manager. */
    private static Roi[] loadRois(File zip) throws IOException {
        List<Roi> rois = new ArrayList<>();
        try (ZipInputStream in = new ZipInputStream(new FileInputStream(zip))) {
            ZipEntry entry;
            byte[] buf = new byte[8192];
            while ((entry = in.getNextEntry()) != null) {
                String name = entry.getName();
                if (!name.endsWith(".roi")) continue;

                ByteArrayOutputStream out = new ByteArrayOutputStream();
                int len;
                while ((len = in.read(buf)) > 0) out.write(buf, 0, len);

                String label = name.substring(0, name.length() - 4);
                Roi roi = new RoiDecoder(out.toByteArray(), label).getRoi();
                if (roi != null) {
                    if (roi.getName() == null) roi.setName(label);
                    rois.add(roi);
                }
            }
        }
        return rois.toArray(new Roi[0]);
    }
}