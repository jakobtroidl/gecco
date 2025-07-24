# Pretty Renderings

```bash
mitsuba --help
ffmpeg --help
```

```bash
python convert.py --input /nrs/turaga/jakob/autoproof_data/flywire_cave_post/test_render --output ./xml # Generate XML files from point clouds
python render.py --xml ./xml --exr ./exr --png ./png # Render the XML files to EXR and PNG images
```
