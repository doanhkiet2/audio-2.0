
2 days ago

v1
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
# =========================
# TRANSCRIBE
# =========================
def process_file(audio):
    output = OUTPUT_DIR / f"{audio.stem}.json"
    if SKIP_DONE and output.exists():
        print(f"SKIP: {audio.name}")
        return
    tmp = TMP_DIR / f"{audio.stem}.wav"
    print(f"\nSTART {audio.name}")
    start = time.time()
    convert_audio(audio, tmp)
    segments, info = model.transcribe(
        str(tmp),
        language="vi",
        vad_filter=True,
        beam_size=BEAM_SIZE,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    result = []
    for s in segments:
        result.append(
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
            }
        )
    with open(
        output,
        "w",
        encoding="utf8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )
    tmp.unlink(missing_ok=True)
    del result
    gc.collect()
    print(f"DONE {audio.name}")
    print(f"TIME: {(time.time()-start)/60:.2f} min")
# =========================
# RUN
# =========================
files = []
for p in INPUT_DIR.rglob("*"):
    if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
        files.append(p)
print(f"\nFound {len(files)} files")
for file in files:
    try:
        process_file(file)
    except Exception as e:
        print(f"ERROR {file.name}")
        print(e)
print("\nALL DONE")