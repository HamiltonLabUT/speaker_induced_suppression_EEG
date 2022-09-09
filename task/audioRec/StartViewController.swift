//
//  StartViewController.swift
//  audioRec
//
//  Created by Garret Kurteff on 1/1/19.
//  Copyright © 2019 Kurteff, Garret. All rights reserved.
//

import UIKit

class StartViewController: UIViewController {

//
// Initialize variables
//
    var version:String = "onsetProd v0.0"
    var currpid:String = "OP0001" // default pid
    var currblock:String = "B1" // default block
    var doAllTrials = true // if false, do the practice only
    var totalRepeats = 3 // default # of repeats
    var ecogBlock = 0
    
    
//
// Initialize UI variables (@IBOutlets)
//
    @IBOutlet weak var versionLabel: UILabel! // gets our version text
    
//
// UI buttons (@IBActions)
//
// (None as of now)
    
//
// viewDidLoad(), what happens when the Settings view is activated
//
    override func viewDidLoad() {
        super.viewDidLoad()
        let version = versionLabel.text
        print("\(String(format:version!)) successfully loaded.")
    }
    
//
// prepare(), lets us pass values to other view controllers
//
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.destination is ViewController {
            let vc = segue.destination as? ViewController
            vc?.doAllTrials = doAllTrials
            vc?.currpid = currpid
            vc?.currblock = currblock
            vc?.totalRepeats = totalRepeats
            vc?.ecogBlock = ecogBlock
        }
    }
    
}
