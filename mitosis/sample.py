from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SCRIPT = (
    "I thought we were fine. We just stopped talking about anything real. "
    "Then my wife found this app called Betterly. Every night it gives us one small question to answer together. "
    "Five minutes, no therapist, no awkward homework. By day nine we talked for two hours. "
    "I forgot how much I actually like her. Couples who feel like roommates, try Betterly free."
)

SHOTS = [
    {"top": (38, 44, 66), "bottom": (122, 92, 70), "shirt": (54, 78, 112), "zoom": "in", "prop": "window"},
    {"top": (24, 28, 36), "bottom": (88, 64, 52), "shirt": (54, 78, 112), "zoom": "out", "prop": "lamp"},
    {"top": (60, 70, 82), "bottom": (170, 150, 128), "shirt": (54, 78, 112), "zoom": "in", "prop": "phone"},
    {"top": (30, 34, 52), "bottom": (110, 80, 96), "shirt": (54, 78, 112), "zoom": "left", "prop": "window"},
    {"top": (44, 50, 40), "bottom": (150, 128, 90), "shirt": (54, 78, 112), "zoom": "in", "prop": "lamp"},
]

WIDTH, HEIGHT = 1080, 1920


def run(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True)


def gradient(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), top)
    draw = ImageDraw.Draw(canvas)
    for y in range(HEIGHT):
        mix = y / (HEIGHT - 1)
        color = tuple(round(a + (b - a) * mix) for a, b in zip(top, bottom))
        draw.line([(0, y), (WIDTH, y)], fill=color)
    return canvas


def draw_prop(draw: ImageDraw.ImageDraw, prop: str) -> None:
    if prop == "window":
        draw.rounded_rectangle([90, 260, 470, 900], radius=18, fill=(196, 214, 230), outline=(80, 90, 104), width=14)
        draw.line([(280, 260), (280, 900)], fill=(80, 90, 104), width=10)
    elif prop == "lamp":
        draw.ellipse([760, 420, 1000, 660], fill=(255, 214, 150))
        draw.rectangle([868, 660, 892, 1100], fill=(70, 60, 50))
    else:
        draw.rounded_rectangle([700, 1180, 900, 1560], radius=28, fill=(28, 28, 32))
        draw.rounded_rectangle([716, 1200, 884, 1540], radius=20, fill=(110, 170, 230))


def draw_hero(draw: ImageDraw.ImageDraw, shirt: tuple[int, int, int]) -> None:
    skin, hair = (214, 172, 140), (52, 40, 32)
    draw.ellipse([250, 1180, 830, 2100], fill=shirt)
    draw.rectangle([480, 1010, 600, 1240], fill=(196, 154, 124))
    draw.ellipse([380, 640, 700, 1080], fill=skin)
    draw.chord([370, 600, 710, 900], start=180, end=360, fill=hair)
    draw.ellipse([450, 820, 490, 850], fill=(40, 30, 28))
    draw.ellipse([590, 820, 630, 850], fill=(40, 30, 28))
    draw.chord([470, 930, 610, 1010], start=0, end=180, fill=(150, 90, 80))
    draw.chord([400, 900, 680, 1090], start=20, end=160, fill=hair)


def frame(shot: dict) -> Image.Image:
    canvas = gradient(shot["top"], shot["bottom"])
    draw = ImageDraw.Draw(canvas)
    draw_prop(draw, shot["prop"])
    draw_hero(draw, shot["shirt"])
    return canvas.filter(ImageFilter.GaussianBlur(1.2))


def zoom_filter(kind: str, frames: int) -> str:
    moves = {
        "in": f"z='min(1.0+0.0009*on,1.18)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        "out": f"z='max(1.18-0.0009*on,1.0)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        "left": f"z=1.12:x='(iw-iw/zoom)*(1-on/{frames})':y='ih/2-(ih/zoom/2)'",
    }
    return f"scale=2160:3840,zoompan={moves[kind]}:d={frames}:s={WIDTH}x{HEIGHT}:fps=30,format=yuv420p"


def narrate(text: str, voice: str, out: Path) -> float:
    with tempfile.TemporaryDirectory() as scratch:
        aiff = Path(scratch) / "voice.aiff"
        try:
            run("say", "-v", voice, "-r", "178", "-o", str(aiff), text)
        except subprocess.CalledProcessError:
            run("say", "-r", "178", "-o", str(aiff), text)
        run("ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-af", "adelay=150:all=1,apad=pad_dur=0.4", "-ar", "48000", "-ac", "1", str(out))
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(probe.stdout.strip())


def music_bed(out: Path, seconds: float) -> None:
    expression = (
        "0.10*sin(2*PI*220*t)*(0.65+0.35*sin(2*PI*0.25*t))"
        "+0.07*sin(2*PI*261.63*t)*(0.6+0.4*sin(2*PI*0.2*t+1))"
        "+0.06*sin(2*PI*329.63*t)*(0.6+0.4*sin(2*PI*0.33*t+2))"
        "+0.05*sin(2*PI*110*t)"
    )
    run(
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"aevalsrc='{expression}|{expression}':s=48000:d={seconds + 4:.2f}",
        "-af", "lowpass=f=1800,afade=t=in:d=1.5,volume=0.8",
        "-ar", "48000", str(out),
    )


def build_sample(out: Path, voice: str = "Reed (English (US))") -> dict[str, Path]:
    if not shutil.which("say"):
        raise RuntimeError("the sample narration needs macOS 'say'; bring your own winner.mp4 + script instead")
    out.mkdir(parents=True, exist_ok=True)
    script_path = out / "winner_script.txt"
    script_path.write_text(SCRIPT + "\n")
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        narration = work / "narration.wav"
        seconds = narrate(SCRIPT, voice, narration)
        per_shot = seconds / len(SHOTS)
        clips: list[Path] = []
        for index, shot in enumerate(SHOTS):
            still = work / f"shot_{index}.png"
            frame(shot).save(still)
            clip = work / f"shot_{index}.mp4"
            frames = max(1, round(per_shot * 30))
            run(
                "ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(still),
                "-vf", zoom_filter(shot["zoom"], frames), "-frames:v", str(frames),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", "30", str(clip),
            )
            clips.append(clip)
        listing = work / "clips.txt"
        listing.write_text("".join(f"file '{c}'\n" for c in clips))
        video = out / "winner.mp4"
        run(
            "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(narration),
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", str(video),
        )
    bed = out / "music_bed.wav"
    music_bed(bed, seconds)
    return {"video": video, "script": script_path, "music": bed}
