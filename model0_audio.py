"""
Model 0 — Audio Feature Extractor (Calibrated v6)
StreamBreaker AI

Calibrated to match Spotify's feature scale:
  - Energy:      RMS amplitude / 0.33 (not RMS*10 which saturates)
  - Speechiness: delta-MFCC modulation / 200 (not ZCR which overcounts cymbals)
  - Valence:     weighted mode + brightness + rolloff + tempo (not brightness alone)

Install: pip install librosa soundfile mutagen
"""

import os, re, json, urllib.request, urllib.parse, numpy as np

KEY_NAMES    = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])


def _parse_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r'[\s_]+\d{5,}[\s_]*$', '', name)
    return name.replace('_',' ').replace('-',' ').strip().title(), ""


def get_file_metadata(file_path_or_bytes, filename="audio"):
    title, artist, embedded_lyrics = "", "", None
    try:
        from mutagen import File as MutagenFile
        if hasattr(file_path_or_bytes, "read"):
            import tempfile
            suffix = os.path.splitext(filename)[-1].lower() or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                pos = file_path_or_bytes.tell() if hasattr(file_path_or_bytes,'tell') else None
                tmp.write(file_path_or_bytes.read())
                if pos is not None: file_path_or_bytes.seek(0)
                load_path = tmp.name
            cleanup = True
        else:
            load_path = file_path_or_bytes
            cleanup = False

        audio = MutagenFile(load_path, easy=False)
        if audio and audio.tags:
            tags = audio.tags
            if hasattr(tags, 'getall'):
                t = tags.get('TIT2')
                if t: title = str(t.text[0]) if hasattr(t,'text') else str(t)
                a = tags.get('TPE1')
                if a: artist = str(a.text[0]) if hasattr(a,'text') else str(a)
                for key in tags.keys():
                    if key.startswith('USLT'):
                        raw = tags[key]
                        text = str(raw.text) if hasattr(raw,'text') else str(raw)
                        if text.strip(): embedded_lyrics = text.strip()
                        break
            elif isinstance(tags, dict):
                title  = str(tags.get('\xa9nam',[''])[0]) if '\xa9nam' in tags else ''
                artist = str(tags.get('\xa9ART',[''])[0]) if '\xa9ART' in tags else ''
                lyr = tags.get('\xa9lyr',None)
                if lyr: embedded_lyrics = str(lyr[0]).strip()
            elif hasattr(tags,'get'):
                title  = str(tags.get('title',[''])[0])  if tags.get('title')  else ''
                artist = str(tags.get('artist',[''])[0]) if tags.get('artist') else ''
                lyr = tags.get('lyrics',None) or tags.get('unsyncedlyrics',None)
                if lyr: embedded_lyrics = str(lyr[0]).strip()
        if cleanup:
            try: os.unlink(load_path)
            except: pass
    except Exception:
        pass
    if not title:
        title, artist = _parse_filename(filename)
    return {"title": title, "artist": artist, "embedded_lyrics": embedded_lyrics}


def extract_features(file_path_or_bytes, filename="audio", duration=None):
    try:
        import librosa, tempfile

        if hasattr(file_path_or_bytes, "read"):
            suffix = os.path.splitext(filename)[-1].lower() or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file_path_or_bytes.seek(0)
                tmp.write(file_path_or_bytes.read())
                file_path_or_bytes.seek(0)
                load_path = tmp.name
            cleanup = True
        else:
            load_path = file_path_or_bytes
            cleanup = False

        y, sr = librosa.load(load_path, sr=22050, mono=True, duration=duration)
        if cleanup:
            try: os.unlink(load_path)
            except: pass

        duration_ms = int((len(y)/sr)*1000)

        # Tempo
        tempo_raw, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.asarray(tempo_raw).flat[0])

        # Energy — calibrated: RMS/0.33 (loud mastered track ~0.25 RMS → 0.75)
        rms = librosa.feature.rms(y=y)[0]
        energy = float(np.clip(np.mean(rms) / 0.33, 0.0, 1.0))

        # Loudness
        loudness = float(np.clip(librosa.amplitude_to_db(rms, ref=np.max).mean(), -60.0, 0.0))

        # Danceability
        beat_times = librosa.frames_to_time(beats, sr=sr)
        if len(beat_times) > 2:
            ibi = np.diff(beat_times)
            regularity = float(np.clip(1.0 - np.std(ibi)/(np.mean(ibi)+1e-6), 0.0, 1.0))
        else:
            regularity = 0.5
        tempo_norm = float(np.clip((tempo-60)/140, 0.0, 1.0))
        danceability = round(0.5*regularity + 0.5*tempo_norm, 3)

        # Acousticness
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        acousticness = float(np.clip(1.0 - np.mean(contrast)/40, 0.0, 1.0))

        # Speechiness — calibrated: delta-MFCC/200 (music lands ~0.01-0.05)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfccs)
        speechiness = float(np.clip(np.mean(np.abs(delta_mfcc[1:5])) / 200.0, 0.01, 0.50))

        # Instrumentalness
        vocal_energy = float(np.mean(np.abs(mfccs[1:4])))
        instrumentalness = float(np.clip(1.0 - vocal_energy/30, 0.0, 1.0))

        # Liveness
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_e = float(np.mean(np.abs(y_harm))) + 1e-6
        perc_e = float(np.mean(np.abs(y_perc)))
        liveness = float(np.clip(perc_e/harm_e*0.5, 0.0, 1.0))

        # Valence — calibrated: mode + brightness + rolloff + tempo
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key = int(np.argmax(chroma_mean))
        chroma_norm = chroma_mean / (chroma_mean.sum()+1e-6)
        maj_corr = np.corrcoef(chroma_norm, np.roll(MAJOR_PROFILE/MAJOR_PROFILE.sum(), key))[0,1]
        min_corr = np.corrcoef(chroma_norm, np.roll(MINOR_PROFILE/MINOR_PROFILE.sum(), key))[0,1]
        mode = 1 if maj_corr > min_corr else 0
        mode_val = float(mode)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        brightness = float(np.clip(np.mean(spec_centroid)/(sr/2), 0.0, 1.0))
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        rolloff_norm = float(np.clip(np.mean(rolloff)/(sr/2), 0.0, 1.0))
        valence = float(np.clip(0.35*mode_val + 0.25*brightness + 0.25*rolloff_norm + 0.15*tempo_norm, 0.0, 1.0))

        return {
            "danceability":     round(danceability, 3),
            "energy":           round(energy, 3),
            "key":              key,
            "loudness":         round(loudness, 2),
            "mode":             mode,
            "speechiness":      round(speechiness, 4),
            "acousticness":     round(acousticness, 3),
            "instrumentalness": round(instrumentalness, 3),
            "liveness":         round(liveness, 3),
            "valence":          round(valence, 3),
            "tempo":            round(tempo, 1),
            "duration_ms":      duration_ms,
            "time_signature":   4,
            "_key_name":        KEY_NAMES[key],
            "_mode_name":       "Major" if mode == 1 else "Minor",
        }, None

    except ImportError:
        return None, "librosa not installed. Run: pip install librosa soundfile"
    except Exception as e:
        return None, f"Audio analysis failed: {str(e)}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python model0_audio.py <audio_file>")
        sys.exit(1)
    feats, err = extract_features(sys.argv[1], filename=sys.argv[1])
    if err:
        print(f"Error: {err}")
    else:
        print("\n Audio Features:")
        for k, v in feats.items():
            print(f"  {k:22s}: {v}")
