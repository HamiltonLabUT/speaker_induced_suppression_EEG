//
//  ViewController.swift
//  audioRec
//
//  Created by Kurteff, Garret on 11/13/18.
//  Copyright © 2018 Kurteff, Garret. All rights reserved.
//

import UIKit
import AVFoundation
import MessageUI

//
// Extensions add additional functions to certain classes of data
//

extension MutableCollection {
// Shuffles the contents of this collection.
    mutating func shuffle() {
        let c = count
        guard c > 1 else { return }
        
        for (firstUnshuffled, unshuffledCount) in zip(indices, stride(from: c, to: 1, by: -1)) {
            let d: Int = numericCast(arc4random_uniform(numericCast(unshuffledCount)))
            let i = index(firstUnshuffled, offsetBy: d)
            swapAt(firstUnshuffled, i)
        }
    }
}

extension Sequence {
// Returns an array with the contents of this sequence, shuffled.
    func shuffled() -> [Element] {
        var result = Array(self)
        result.shuffle()
        return result
    }
}

extension String {
// necessary for writeToLog() func
    func appendLineToURL(fileURL: URL) throws {
        try (self + "\n").appendToURL(fileURL: fileURL)
    }

    func appendToURL(fileURL: URL) throws {
        let data = self.data(using: String.Encoding.utf8)!
        try data.append(fileURL: fileURL)
    }
}

extension Data {
// necessary for String extension
    func append(fileURL: URL) throws {
        if let fileHandle = FileHandle(forWritingAtPath: fileURL.path) {
            defer {
                fileHandle.closeFile()
            }
            fileHandle.seekToEndOfFile()
            fileHandle.write(self)
        }
        else {
            try write(to: fileURL, options: .atomic)
        }
    }
}

class ViewController: UIViewController, AVAudioRecorderDelegate, AVAudioPlayerDelegate, MFMailComposeViewControllerDelegate {
    
//
// Initialize variables
//
    var audioPlayer:AVAudioPlayer!
    var audioRecorder:AVAudioRecorder!
    var echolaliaComplete:Bool = false // flags the end of the echolalia block
    var currentBlock:String = "" // echolalia or shuffled or break
    var currblock:String = "B1" // Recording block, sorry my var names here are so confusing...
    var currentFile = 0 // counter for which file to play
    var currpid:String = "OP0999" // Subject ID
    var currentRecording = 0 // counter for the filename of the recording, similar to currentTrial but not reset per block
    var currentRepeat = 0 // lets us know what repeat we are currently on, starts at 0
    var currentText:String! // text of current trial
    var currentTrial:Int = 0 // counter for the number of trials
    var doAllTrials = true
    var fileList = [String]() // list of all text files to display
    var fileListShuffled = [String]() // list of text files, shuffled for the "shuffled" block (this is the shuffled sentence text)
    var logFile = "OP0999_B1"
    var filename:String = "mocha_50.txt"
    var newLogLine:String = "shut up Xcode" // stifles warnings when we are trying to add a new log line...
    var nextButtonPressed:Bool = true // this is prob a bad idea but whatever....
    var numTrials = 1 // overall trial count
    var randomIndexList = [Int]() // list of the randomized indices for the shuffled block (rather than the sentences themselves)
    var randomIndex:Int = 0 // This is actually overwritten but is for getting indices of the shuffled FileListShuffled
    var randomTrial = 0 // need to actually init this so we can idx which trial they heard for the log
    var refresh = 0 // refresh counter for update()
    var timeInterval:Double = 0.0 // timestamp, updated in our getTimestamp() function
    var totalTrials = 50 // the total number of trials in each block (echolalia/shuffled), so this *2 = total sentences read by participant
    var totalRepeats = 3 // how many combination echolalia/shuffled blocks we are doing beyond the first, default is 3 (for 400 trials)
    var trialPart = "readRepeat"
    var ecogBlock = 0
    var ecogBlockSize = 20 // how many trials in an ecog block
    var inPlayback = false // whether we're in the playback portion of the trial or not
    
//
// Initialize UI variables (@IBOutlets)
//
    @IBOutlet weak var breakOverLabel: UIButton!
    @IBOutlet weak var breakText: UILabel!
    @IBOutlet weak var cross: UIImageView!
    @IBOutlet weak var emailButtonLabel: UIButton!
    @IBOutlet weak var MOCHAText: UILabel! // text box that displays the MOCHA sentences
    @IBOutlet weak var nextButtonLabel: UIButton! // Next button
    @IBOutlet weak var taskCompleteText: UILabel! // hidden, displays when experimenter is finished
    
//
// Define functions
//
    func addLineToLogFile() {
    // appends our log
    // our columns in our log file are: "Trial, CumTrial, Trial Part, CurrentBlock, Time, MOCHARead, MOCHAHeard, CurrentRepetition"
        if currentBlock == "echolalia" {
            newLogLine = "\(currentTrial)\t\(currentRecording)\t\(trialPart)\t\(currentBlock)\t\(timeInterval)\t\(String(fileList[currentFile-1]))\t\(String(fileList[currentFile-1]))\t\(currentRepeat)\n"
        }
        if currentBlock == "shuffled" {
            newLogLine = "\(currentTrial)\t\(currentRecording)\t\(trialPart)\t\(currentBlock)\t\(timeInterval)\t\(String(fileList[currentFile-1]))\t\(String(fileListShuffled[currentFile-1]))\t\(currentRepeat)\n"
        }
        if currentBlock == "break" {
            newLogLine = "0\t0\tbreak\t\(currentBlock)\t\(timeInterval)\tbreak\tbreak\tbreak\n"
        }
        
        print(newLogLine)
        writeToLog(data: newLogLine as String)
    }
    
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool){
    // hides the Next button until playback is finished
        if flag == true && inPlayback == true { // inPlayback has to have happened so that the button won't be enabled during the listening part
            nextButtonLabel.isEnabled = true
            nextButtonLabel.isHidden = false
            preloadSound(fname: "click")
            inPlayback = false
        }
    }
    
    func displayAlert(title:String, message:String) {
    // displays an alert, lets us know if audio recorder is broken.
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "dismiss", style: .default, handler: nil))
        present(alert, animated: true, completion: nil)
    }
    
    @objc func displayCross() {
    // displays the cross for 1000ms (can be changed in update())
        cross.isHidden = false
        refresh = 0 // sets the refresh cycle for our update() function back to 0
        // uses CADisplayLink to accurately track timing:
        let displayLink = CADisplayLink(target: self, selector: #selector(update))
        displayLink.add(to: .current, forMode: .commonModes)
    }
    
    func getAndDisplayTextFile() {
    // updates our current text file then displays it in a text box
        MOCHAText.isHidden = false
        if currentFile == fileList.count { // are we at the end of the file?
            currentFile = 0 //+ (ecogBlockSize*(ecogBlock-1))
        }
        else { // if we aren't at the end of the file
            MOCHAText.text = String(fileList[currentFile]) // Set MOCHA label text to the next line in the file
            currentFile += 1
            currentTrial += 1
            currentRecording += 1
        }
    }
    
    func getDirectory() -> URL {
    // gets path to directory, for audio playback
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let documentDirectory = paths[0]
        return documentDirectory
    }
    
    func getTimestamp() -> Double {
    // having this in a separate function allows us to get more accurate timestamps
        let stopSpeech = DispatchTime.now() // << Reaction time
        timeInterval = Double(stopSpeech.uptimeNanoseconds) / 1_000_000_000 // Technically could overflow for long running tests
        return timeInterval
    }
    
    @objc func listenPart() {
    // part of trial where you listen
    // plays back what they said (or something random)
        trialPart = "listen"
        displayCross()
        addLineToLogFile() // append our log file

    }
    
    @objc func loadMOCHA() {
    // loads the MOCHA corpus from a specified text file
    // called when the main ViewController is loaded, so at the start of the app basically
    // also called when the second (shuffled) block starts
        print(filename)
        if let path = Bundle.main.path(forResource: filename, ofType: "txt") {
            do { // try to read text file
                let text = try String(contentsOfFile: path, encoding: String.Encoding.utf8)
                fileList = text.components(separatedBy: "\n")
                fileList = fileList.filter { $0 != "" }
            }
            catch {
                print("Failed to read text from \(filename)")
            }
        }
        else {
            print("Failed to load file from app bundle \(filename)")
        }
        numTrials = fileList.count
        currentTrial = 0
        currentFile = 0 //+ (ecogBlockSize*(ecogBlock-1))
        if ecogBlock == 0 {
            fileListShuffled = fileList.shuffled()
        }
        else {
            fileList = Array(fileList[((ecogBlock-1)*ecogBlockSize)..<(ecogBlock*ecogBlockSize)])
            fileListShuffled = fileList.shuffled()
            print(fileList)
            print(fileListShuffled)
        }
        for elem in fileListShuffled {
            randomIndex = fileList.index(of: elem)!
            randomIndexList.append(randomIndex)
        }
        print(randomIndexList)
    }
    
    func mailComposeController(_ controller: MFMailComposeViewController, didFinishWith result: MFMailComposeResult, error: Error?) {
    // necessary for sendEmail() to work properly
        controller.dismiss(animated: true)
    }
    
    @objc func playBackTrial(block: String) {
    // plays back the previous trial using information from the var currentTrial
        inPlayback = true
        if block == "echolalia" {
            nextButtonLabel.isEnabled = false
            nextButtonLabel.isHidden = true
            let path = getDirectory().appendingPathComponent("\(currentRecording).caf")
            do{
                audioPlayer = try AVAudioPlayer(contentsOf: path)
                audioPlayer.delegate = self
                audioPlayer.prepareToPlay() // minimizes lag after ".play()" is called
                audioPlayer.play()
            }
            catch {
            }
        }
        if block == "shuffled" {
            nextButtonLabel.isEnabled = false
            nextButtonLabel.isHidden = true
            
            //randomTrial = Int(arc4random_uniform(UInt32(totalTrials))) // picks a random file
            randomTrial = randomIndexList[currentTrial-1]
            // this next line is ugly to make sure we are repeating the trial from the same repeat
            // which is kind of a nitpicky detail so we could just discard it...
            let path = getDirectory().appendingPathComponent("\((currentRepeat*(2*totalTrials))+randomTrial+1).caf")
            do{
                audioPlayer = try AVAudioPlayer(contentsOf: path)
                audioPlayer.delegate = self
                audioPlayer.prepareToPlay() // minimizes lag after ".play()" is called
                audioPlayer.play()
            }
            catch {
                print("It picked the bad random number: \(randomTrial)")
            }
        }
        if currentTrial == totalTrials  { // if we've done 10/100/X echolalia trials, move to the second block.
        // This has to happen after the listening/playback part has occurred
            currentBlock = "break" // start the break
            echolaliaComplete = !echolaliaComplete // flips value of bool
            print("Echolalia complete: \(echolaliaComplete)")
        }
    }
    
    @objc func playSound(fname: String) {
    // plays back a sound effect that was loaded with preloadSound()
        audioPlayer.play()
    }
    
    @objc func preloadSound(fname: String) {
    // Preloads a sound to play. This does the same thing
    // that playSound used to do, but doesn't actually play the sound
        let path = NSURL(fileURLWithPath: Bundle.main.path(forResource: fname, ofType: "wav")!)
        do{
            audioPlayer = try AVAudioPlayer(contentsOf: path as URL)
            audioPlayer.delegate = self
            audioPlayer.prepareToPlay()
        }
        catch {
            print("Error locating file \(path)")
        }
    }
    
    @objc func readRepeatPart() {
    // Part of the trial where you read something on the screen and repeat it
        trialPart = "readRepeat"
        if currentTrial == 0 {
            playSound(fname: "boop_RMS-26")
        }
        else {
            playSound(fname: "click")
        }
        nextButtonLabel.setTitle("Stop Recording", for: .normal) // Do this before the transition happens, otherwise it looks weird and flickery
        displayCross()
        getAndDisplayTextFile() // this ticks up the trial counter
        addLineToLogFile() // append our log
        // startRecording is called after playSound() is complete, so we do not call it here
        
    }
    
    func sendEmail() {
        let logFile = "\(currpid)_\(currblock)"
        let DocumentDirURL = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        let fileURL = DocumentDirURL.appendingPathComponent(logFile).appendingPathExtension("txt")
        do {
            let text2 = try String(contentsOf: fileURL, encoding: .utf8)
            print(text2)
            
            if MFMailComposeViewController.canSendMail() {
                let mail = MFMailComposeViewController()
                mail.mailComposeDelegate = self
                mail.setToRecipients(["onsetPr.9ra865e4iufuqxov@u.box.com"])
                mail.setSubject("Log file \(currpid) \(currblock)")
                mail.setMessageBody("Log file", isHTML: false)
                
                if let data = (text2 as NSString).data(using: String.Encoding.utf8.rawValue){
                    //Attach File
                    mail.addAttachmentData(data, mimeType: "text/plain", fileName: "\(logFile).txt")
                    present(mail, animated: true)
                }
                else {
                    print("Can't send email... check logs")
                }
            }
        }
        catch {/* error handling here */
            print("Error: no log file!")
        }
    }
    
    @objc func startRecording() {
    // start recording audio
            let filename = getDirectory().appendingPathComponent("\(currentRecording).caf") // "1.caf" , "2.caf", etc
            let settings = [AVFormatIDKey: Int(kAudioFormatAppleLossless), AVSampleRateKey: 44100, AVNumberOfChannelsKey: 1, AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue]
            do {
                audioRecorder = try AVAudioRecorder(url: filename, settings: settings)
                audioRecorder.delegate = self
                audioRecorder.record()
                nextButtonLabel.isEnabled = true
                nextButtonLabel.isHidden = false
                nextButtonLabel.setTitle("Stop Recording", for: .normal)
            }
            catch {
                displayAlert(title: "Oops!", message: "Recording failed")
            }
    }
    
    @objc func stopRecording() {
    // stop recording audio
        audioRecorder.stop()
        audioRecorder = nil
        nextButtonLabel.setTitle("Next", for: .normal)
    }
    
    @objc func takeABreak() {
    // this displays the break screen
        breakText.isHidden = false
        breakOverLabel.isHidden = false
        breakOverLabel.isEnabled = true
        addLineToLogFile()
    }
    
    @objc func update(displayLink: CADisplayLink) {
    // updates a counter for a specified number of frames (refresh) for timing purposes
    // functions differently if we are in the readRepeat part or the listen part of our trial.
        refresh += 1 // keeps track of how many frames we've been updating for
        if refresh > 70 { // 60fps app, so 30 frames = 500ms, 60 frames = 1000ms
            displayLink.invalidate() // stop updating
            cross.isHidden = true
            if trialPart == "readRepeat" {
                startRecording()
            }
            else if trialPart == "listen" {
                playBackTrial(block: currentBlock)
            }
        }
    }

    func writeToLog(data: String) {
    // ... does what it says it does. stolen from SpeechSong
        let logFile = "\(currpid)_\(currblock)"
        let DocumentDirURL = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let fileURL = DocumentDirURL.appendingPathComponent(logFile).appendingPathExtension("txt")
        do {
            try data.appendToURL(fileURL: fileURL)
        }
        catch let error as NSError {
            print("failed to write to URL ")
            print(error)
        }
    }

//
// UI buttons (@IBActions)
//
    @IBAction func breakOver(_ sender: Any) {
    // this button is pressed at the end of a break
        cross.isHidden = true
        breakText.isHidden = true
        breakOverLabel.isEnabled = false
        breakOverLabel.isHidden = true
        if echolaliaComplete == false { // end of shuffled block
            currentRepeat += 1
            print("Starting repeat number \(currentRepeat) of \(totalRepeats)!")
            currentBlock = "echolalia"
        }
        else { // end of echolalia block
            print("Starting shuffle!")
            currentBlock = "shuffled"
        }
        currentTrial = 0 // do this instead of loading mocha again
        currentFile = 0 //+ (ecogBlockSize*(ecogBlock-1)) // do this instead of loading mocha again
        readRepeatPart() // Run the reading+repeat portion
    }
    
    @IBAction func emailButton(_ sender: Any) {
    // button that unhides at end of task, hitting it sends the log file to Box
        sendEmail()
    }
    
    @IBAction func nextButton(_ sender: Any) {
    // this button advances the task
    // has a lot of functions depending on where we are in the task (described below)
        nextButtonLabel.isEnabled = false
        nextButtonLabel.isHidden = true
        timeInterval = getTimestamp() // this updates our timestamp for our log file every time the button is pressed
        if audioRecorder == nil {
            if nextButtonPressed == true {
                print("You pressed the button too fast my dude.")
            }
            else {
                nextButtonPressed = true
                // "Next" button
                if currentTrial == totalTrials && echolaliaComplete == false && currentRepeat == totalRepeats {
                // end of task
                // playSound(fname: "click") // for some reason, calling this here breaks the end of the task...
                    nextButtonLabel.isEnabled = false
                    nextButtonLabel.isHidden = true
                    taskCompleteText.isHidden = false
                    emailButtonLabel.isHidden = false
                }
                if currentBlock == "break" && taskCompleteText.isHidden == true {
                // if the button is pressed during the break, do this
                // currentBlock is set to break at the end of block one, in the playBackTrial function
                    playSound(fname: "click")
                    takeABreak()
                }
                else if currentBlock != "break" {
                // anywhere else in the task (not the break or the end), run the reading + repeat portion
                    readRepeatPart()
                }
            }
            nextButtonPressed = false
        }
        else {
        // "Stop Recording" button, also advances to the listenPart of the trial
            stopRecording()
            playSound(fname: "click")
            MOCHAText.isHidden = true
            listenPart()
            nextButtonPressed = false
        }
    }
    override var keyCommands: [UIKeyCommand]? {
        return [
            UIKeyCommand(input: "n", modifierFlags: .command, action: #selector(nextButton(_:)), discoverabilityTitle: "Next")
        ]
    }
//
// viewDidLoad(), what happens when our main view is activated (AKA when you click "Start" pretty much)
//
    override func viewDidLoad() {
        super.viewDidLoad()
        if ecogBlock == 0 {
            if doAllTrials == true { // mocha_50.txt
                filename = "mocha_50"
                totalTrials = 50
            }
            else { // mini_mocha.txt
                filename = "mini_mocha"
                totalTrials = 5
            }
        }
        else {
            print("ECOG MODE, BABY!!!") // we did it fam
            print("Ecog block: \(ecogBlock)")
            filename = "mocha_100"
            totalRepeats = 0
            totalTrials = ecogBlockSize // 20
        }

        print("Current block: \(currblock)")
        print("Participant ID: \(currpid)")
        if ecogBlock == 0 {
        print("Do all trials?: \(doAllTrials)")
        }
        print("Each block will be \(totalTrials) trials long with \(totalRepeats) repetitions beyond the first. So, the participant will read and repeat \(2*totalTrials*(totalRepeats+1)) sentences across the task (\(totalTrials*(totalRepeats+1)) echolalia and \(totalTrials*(totalRepeats+1)) shuffled).")
        
    
        view.backgroundColor = .black
        nextButtonLabel.titleLabel?.textAlignment = NSTextAlignment.center // centers the nextButton text. for some godforsaken reason Xcode has no way in the UI to do this!!!
        emailButtonLabel.titleLabel?.textAlignment = NSTextAlignment.center // centers the emailButton text. for some godforsaken reason Xcode has no way in the UI to do this!!!
        preloadSound(fname: "boop_RMS-26")
        loadMOCHA()
        currentBlock = "echolalia"
        
        // Set up the file for logging the behavior
        // Header, then column labels in a second line separated by tabs
        logFile = "\(currpid)_\(currblock)" // update our log file from the placeholder
        print(logFile)
        let DocumentDirURL = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let fileURL = DocumentDirURL.appendingPathComponent(logFile).appendingPathExtension("txt")
        print("File Path: \(fileURL.path)")
        let currentDateTime = Date() // get the current date and time
        let formatter = DateFormatter() // initialize the date formatter
        formatter.timeStyle = .medium // set style of date formatter
        formatter.dateStyle = .long // set style of date formatter
        let currentDate = formatter.string(from: currentDateTime) // January 2, 2019 at 10:08:19 AM
        print(formatter.string(from: currentDateTime))
        // This next line sets up the format of our colums
        let writeString = "\(currentDate)\nONSET PROD LOG FILE\n\(currpid)_\(currblock)\nTrial\tCumTrial\tTrialPart\tCurrentBlock\tTime\tMOCHARead\tMOCHAHeard\tCurrentRepetition\n"
        do {
            try writeString.write(to: fileURL, atomically: true, encoding: String.Encoding.utf8)
            print(writeString)
        }
        catch let error as NSError {
            print("failed to write to URL ")
            print(error)
        }
        timeInterval = getTimestamp()
        readRepeatPart() // start our task! :)
        }
    }
