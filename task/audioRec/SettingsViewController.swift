//
//  SettingsViewController.swift
//  audioRec
//
//  Created by Garret Kurteff on 1/1/19.
//  Copyright © 2019 Kurteff, Garret. All rights reserved.
//

import UIKit

class SettingsViewController: UIViewController {

//
// Initialize variables
//
    var currpid:String = "OP0001" // default pid
    var currblock:String = "B1" // default block
    var doAllTrials = true // if false, do the practice only
    var totalRepeat = 3
    
    
//
// Initialize UI variables (@IBOutlets)
//
    @IBOutlet weak var practiceSwitch: UISwitch! // toggle from mini_mocha to mocha_100, mocha_100 by default
    @IBOutlet weak var pid: UITextField! // participant ID, eg. OP0001
    @IBOutlet weak var block: UITextField! // block #, B1 by default
    @IBOutlet weak var totalRepeats: UITextField!
    @IBOutlet weak var ecogButton: UISegmentedControl!
    
//
// UI buttons (@IBActions)
//
    @IBAction func flipPracticeSwitch(_ sender: Any) {
        doAllTrials = practiceSwitch.isOn
        print("Flipped switch")
        print(doAllTrials)
    }
    
//
// viewDidLoad(), what happens when the Settings view is activated
//
    override func viewDidLoad() {
        super.viewDidLoad()
        pid.text = currpid
        block.text = currblock
        totalRepeats.text = String(totalRepeat)
    }
    
//
// prepare(), lets us pass values to other view controllers
//
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.destination is StartViewController {
            let vc = segue.destination as? StartViewController
            vc?.doAllTrials = doAllTrials
            vc?.currpid = pid.text!
            vc?.currblock = block.text!
//            let totalRepeat = NumberFormatter().number(from:totalRepeats .titleForSegment(at: totalRepeats.selectedSegmentIndex)!)!.floatValue
            vc?.totalRepeats = Int(totalRepeats.text!)!
            let thisRate = NumberFormatter().number(from: ecogButton.titleForSegment(at: ecogButton.selectedSegmentIndex)!)!.floatValue
            vc?.ecogBlock = Int(thisRate)
        }
    }
    
}
