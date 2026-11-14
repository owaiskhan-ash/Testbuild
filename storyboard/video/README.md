# The Last Durian – storyboard video

`the-last-durian.mp4` is a 31 second, 1920×1080, 30 fps motion-comic cut of the five storyboard
frames in the parent folder. It has no audio.

The stills are not re-drawn or animated by a video model. Each frame is split into depth layers
and filmed with a virtual camera:

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `prep.py` | 2× upscale of each frame, a saliency matte and a depth map (saved in `maps/`) |
| 2 | `layers.py` | Splits the frame into background plate, characters and near foreground, and pre-blurs each layer with a bokeh-shaped kernel |
| 3 | `render.py` | Per shot: parallax camera move, depth of field with focus pulls, character rim light and halo, light wash, dust, sparkles, bloom, colour grade |
| 4 | `assemble.py` | Defocus dissolves between shots, titles, film grain, fades, H.264 encode |

Shot timing, camera moves and focus pulls are the `shot1` … `shot5` functions at the top of
`render.py`; the look of each scene is in `LOOK`.

## Re-rendering

```bash
python -m venv /tmp/venv
/tmp/venv/bin/pip install numpy opencv-python-headless onnxruntime ncnn numexpr pillow imageio-ffmpeg

/tmp/venv/bin/python prep.py                 # only if the storyboard frames change
/tmp/venv/bin/python layers.py
/tmp/venv/bin/python render.py shot 1 2 3 4 5
/tmp/venv/bin/python assemble.py
```

Working files (about 8 GB) go to `$DURIAN_WORK`, default `/tmp/work`. `render.py still 3 2.5 out.jpg`
renders a single frame of shot 3 at 2.5 s for look-dev.

`prep.py` and `assemble.py` expect these files under `$DURIAN_WORK`; they are not in the repo:

| File | Source |
| --- | --- |
| `esr/realesr-animevideov3-x2.{param,bin}` | `models/` inside the `realesrgan-ncnn-py` wheel on PyPI |
| `models/isnet_medium.onnx` | the `/models/medium` chunks listed in `dist/resources.json` of the npm package `@imgly/background-removal-node@1.4.5`, concatenated |
| `models/da2_small_int8.onnx` | `plugin/models/depth_anything_v2_small_int8.onnx` in the GitHub repo `rydersd/ill-tool` |
| `fonts/DMSerifDisplay-Regular.ttf`, `DMSerifDisplay-Italic.ttf`, `Poppins-Medium.ttf` | `ofl/dmserifdisplay` and `ofl/poppins` in the GitHub repo `google/fonts` |
