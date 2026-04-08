import librosa
import librosa.display
import matplotlib.pyplot as plt
import pretty_midi
import numpy as np
import torch
from piano_transcription_inference import PianoTranscription

# --- 1. เตรียมพาธไฟล์ ---
audio_path = 'my_piano2.mp3' 
model_path = 'best_piano_model_fixed.pt' # ไฟล์โมเดลที่แก้ไขแล้ว
output_midi_path = 'output.mid'

print(f"กำลังโหลดโมเดลจาก: {model_path} ...")
# สร้าง Model โดยระบุ checkpoint_path ไปที่ไฟล์ .pt ของคุณ
transcriptor = PianoTranscription(
    device='cuda' if torch.cuda.is_available() else 'cpu', 
    checkpoint_path=model_path,
    segment_samples=16000*10
)

print("กำลังโหลดเสียงและประมวลผล...")
# Run transcription and get the output dict (do not write MIDI yet)
audio, sr = librosa.load(audio_path, sr=16000, mono=True)
transcribed_dict = transcriptor.transcribe(audio, None)

# ตรวจสอบและกรอง NaN ใน note events
est_note_events = transcribed_dict.get('est_note_events', [])
est_pedal_events = transcribed_dict.get('est_pedal_events', [])

# กรอง event ที่มี NaN
valid_notes = [note for note in est_note_events if not any(np.isnan(val) for val in [note['onset_time'], note['offset_time'], note['midi_note'], note['velocity']])]
valid_pedals = [pedal for pedal in est_pedal_events if not any(np.isnan(val) for val in [pedal['onset_time'], pedal['offset_time']])]

print(f"Filtered {len(est_note_events) - len(valid_notes)} invalid note events, {len(est_pedal_events) - len(valid_pedals)} invalid pedal events")

# เขียน MIDI ด้วย events ที่ถูกต้อง
from piano_transcription_inference.utilities import write_events_to_midi
write_events_to_midi(start_time=0, note_events=valid_notes, pedal_events=valid_pedals, midi_path=output_midi_path)
print(f'Write out to {output_midi_path}')

# --- 2. Plot Matching Analysis ไปที่ Output ---
print("กำลังวาดกราฟ Spectrogram Match...")
S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
S_db = librosa.power_to_db(S, ref=np.max)
midi_data = pretty_midi.PrettyMIDI(output_midi_path)

plt.figure(figsize=(15, 8))
# วาด Spectrogram พื้นหลัง
librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, cmap='magma')

# วาดโน้ตที่ AI ทายออกมาทับลงไป
for instrument in midi_data.instruments:
    for note in instrument.notes:
        note_hz = librosa.midi_to_hz(note.pitch)
        plt.hlines(note_hz, note.start, note.end, colors='lime', linewidth=3, alpha=0.7)

plt.colorbar(format='%+2.0f dB')
plt.title(f'Piano Matching: {model_path} vs Audio Input', fontsize=15)
plt.tight_layout()
plt.show()