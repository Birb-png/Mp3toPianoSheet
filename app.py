import os
import uuid
import shutil
import subprocess
from pathlib import Path
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from moviepy.editor import AudioFileClip, VideoFileClip

app = Flask(__name__)
BASE = Path(__file__).parent
UPLOADS = BASE / 'uploads'
OUTPUTS = BASE / 'outputs'
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.ogv', '.mpeg'}


def extract_audio_file(source_path: Path, target_path: Path):
    ext = source_path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        clip = VideoFileClip(str(source_path))
        if clip.audio is None:
            clip.close()
            raise RuntimeError('No audio track found in video file.')
        clip.audio.write_audiofile(str(target_path), fps=16000, verbose=False, logger=None)
        clip.audio.close()
        clip.close()
    else:
        clip = AudioFileClip(str(source_path))
        clip.write_audiofile(str(target_path), fps=16000, verbose=False, logger=None)
        clip.close()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    f = request.files.get('file')
    if not f:
        return redirect(url_for('index'))

    ext = Path(f.filename).suffix.lower()
    uid = uuid.uuid4().hex
    upload_path = UPLOADS / f'{uid}{ext}'
    f.save(upload_path)

    audio_input = upload_path
    if ext in VIDEO_EXTENSIONS:
        audio_input = UPLOADS / f'{uid}.wav'
        extract_audio_file(upload_path, audio_input)

    midi_path = OUTPUTS / f'{uid}.mid'
    png_path = OUTPUTS / f'{uid}.png'
    xml_path = OUTPUTS / f'{uid}.musicxml'

    subprocess.run([
        'python', str(BASE / 'process_file.py'),
        '--input', str(audio_input),
        '--output-dir', str(OUTPUTS)
    ], check=True)

    xml_name = xml_path.name if xml_path.exists() else None
    return render_template('result.html', midi=midi_path.name, xml=xml_name, png=png_path.name)


@app.route('/convert_xml', methods=['POST'])
def convert_xml():
    midi_name = request.form.get('midi')
    if not midi_name:
        return redirect(url_for('index'))

    midi_path = OUTPUTS / midi_name
    if not midi_path.exists():
        return redirect(url_for('index'))

    xml_path = midi_path.with_suffix('.musicxml')
    if not xml_path.exists():
        from converter import midi_to_sheet
        midi_to_sheet(str(midi_path), str(xml_path))

    png_name = midi_path.with_suffix('.png').name
    return render_template('result.html', midi=midi_name, xml=xml_path.name, png=png_name)


@app.route('/outputs/<path:filename>')
def outputs(filename):
    return send_from_directory(OUTPUTS, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
