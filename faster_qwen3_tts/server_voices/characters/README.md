# Character reference library

One directory per character, one subdirectory per emotion, and inside each a set of
`.wav` / `.txt` pairs sharing a filename stem. The `.wav` is the reference recording to
clone from; the `.txt` is its exact transcript. A request for a character and emotion
picks one pair at random.

```
nicole/
  character.yaml        # optional
  neutral/
    calm_intro.wav
    calm_intro.txt
  amused/
    ...
```

Characters are discovered from the directory names, so adding one means adding a
directory — no code or config change. The ten emotion names are fixed:

    neutral  amused  smug  excited  impressed  earnest  deadpan  annoyed  panicked  confused

An emotion with no pairs falls back to `neutral`, so `neutral` is the one directory a
character must have populated — a character without it is skipped at startup.

`character.yaml` is optional and sets `language` (default English) and `temperature`
(default 0.7).

Recordings should be clean, single-speaker, and roughly 5-15 seconds. The transcript must
match the audio exactly; the clone path conditions on it, and a wrong transcript degrades
the result.
