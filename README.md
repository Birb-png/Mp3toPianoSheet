MP3 to Piano Sheet: End-to-End Transcription
An AI-powered pipeline that converts polyphonic piano audio (MP3/WAV) into structured MIDI and printable sheet music (PDF/MusicXML). This project utilizes deep learning for note detection and algorithmic music theory for notation.

Features
Automatic Speech Recognition (ASR) Adaptation: Uses a Wav2Vec2 backbone fine-tuned on the MAESTRO dataset for high-accuracy piano transcription.

Audio-to-MIDI: Extracts precise note onsets, offsets, and velocities.

Intelligent Sheet Music Generation: Uses music21 to handle complex notation logic, including automatic splitting of notes into Right Hand (Treble) and Left Hand (Bass) staves.

Format Support: Exports to MIDI, MusicXML, and PDF.

Technical Pipeline
Preprocessing: Audio is normalized and converted into spectrograms or raw waveforms suitable for the transformer model.

Transcription Model: The model (Wav2Vec2 based) predicts the pitch and timing of notes from the audio.

Post-processing: Raw predictions are cleaned to remove "ghost notes" and quantized for musical rhythm.

Notation Engine:

MIDI events are parsed via music21.

Heuristic-based hand splitting (Right/Left hand).

Key signature and time signature detection.

----- Start with trainfinal.py --> convert_checkpoint.py --> extract.py --> convert.py
