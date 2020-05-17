//
//  ViewController.swift
//  DogFinder
//
//  Created by Christian Schneider on 23.12.19.
//  Copyright © 2019 NonameCompany. All rights reserved.
//

import Cocoa
import CoreML
import Quartz
import Vision
import ImageIO

class ViewController: NSViewController {

    @IBOutlet weak var imageView: NSImageView!
    @IBOutlet weak var filepathLabel: NSTextField!
    @IBOutlet weak var filechooserButton: NSButton!
    @IBOutlet weak var predictionLabel: NSTextField!
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }

    @IBAction func pickImage(_ sender: Any) {
        let dialog = NSOpenPanel();
        
        dialog.title                   = "Choose a .jpg file";
        dialog.showsResizeIndicator    = true;
        dialog.showsHiddenFiles        = false;
        dialog.canChooseDirectories    = true;
        dialog.canCreateDirectories    = true;
        dialog.allowsMultipleSelection = false;
        dialog.allowedFileTypes        = ["jpg"];
        
        if (dialog.runModal() == NSApplication.ModalResponse.OK) {
            let result = dialog.url
            
            if (result != nil) {
                filepathLabel.stringValue = result!.path
                self.imageView.image = NSImage(byReferencingFile: self.filepathLabel.stringValue)
                updateClassifications()
            }
        } else {
            // Cancel
            return
        }
    }
    
    lazy var classificationRequest: VNCoreMLRequest = {
        do {
//            let model = try VNCoreMLModel(for: dog_classifier().model)
            let model = try VNCoreMLModel(for: ourdog_classifier().model)

            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()

    func updateClassifications() {
        self.predictionLabel.stringValue = "Classifying..."
        
        guard let ciImage = CIImage(contentsOf: URL(fileURLWithPath: self.filepathLabel.stringValue)) else { fatalError("Unable to create \(CIImage.self).") }

        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage)
            do {
                try handler.perform([self.classificationRequest])
            } catch {
                print("Failed to perform classification.\n\(error.localizedDescription)")
            }
        }
    }

    func processClassifications(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            guard let results = request.results else {
                self.predictionLabel.stringValue = "Unable to classify image.\n\(error!.localizedDescription)"
                return
            }
            let classifications = results as! [VNClassificationObservation]

            if classifications.isEmpty {
                self.predictionLabel.stringValue = "Nothing recognized."
            } else {
                let topClassifications = classifications.prefix(2)
                let descriptions = topClassifications.map { classification in
                    return String(format: "  (%.2f) %@", classification.confidence, classification.identifier)
                }
                self.predictionLabel.stringValue = "Classification:\n" + descriptions.joined(separator: "\n")
            }
        }
    }
    
    override var representedObject: Any? {
        didSet {
        }
    }

}
