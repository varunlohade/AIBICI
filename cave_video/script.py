"""Single source of truth for the Boesmansgat cave-diving Reel.

Each scene is a list of lines: (voice, text, pause_after_seconds[, speed]).
Every factual line is traceable to RESEARCH.md (sources and confidence levels there).
`python3 build_audio.py` builds narration + cues.json; `python3 script.py` writes SCRIPT.md.
"""

VOICES = {
    "NARR": "bm_george",  # Kokoro fallback; the release narration uses ElevenLabs "George" (JBFqnCBsd6RMkjVDRZzb)
}

SPEAKER_LABELS = {}

N = "NARR"
SCENES = [
    {
        "id": "c01_hook",
        "title": "Hook",
        "visual": "Full cross-section of the sinkhole, surface to black floor. A tiny torch light starts sinking. "
                  "Depth gauge ticks. Date slip: 8 JAN 2005.",
        "lines": [
            (N, "In January 2005, a diver went two hundred and seventy metres down a flooded sinkhole...", 0.3, 0.95),
            (N, "to bring a stranger home.", 0.8, 0.92),
        ],
    },
    {
        "id": "c02_hole",
        "title": "Boesmansgat",
        "visual": "Surface: a small duckweed pool on a dry farm. Push down through the narrow throat into a vast "
                  "flooded chamber. 1994: a second, older light sinks away and goes still on the floor (a soft glow, never a body).",
        "lines": [
            (N, "Boesmansgat, South Africa: a small pool above a flooded chamber nearly three hundred metres deep.", 0.3),
            (N, "In 1994, a twenty-year-old diver named Deon Dreyer sank out of sight in it, and never came back up.", 0.3),
            (N, "His body stayed down there for ten years.", 0.7, 0.95),
        ],
    },
    {
        "id": "c03_shaw",
        "title": "Dave Shaw",
        "visual": "Silhouette of a diver with a rebreather; a pilot's wings pin. 2004: his torch cone sweeps the floor "
                  "and finds the soft glow. He ties his line to it. Then: an offer, two silhouettes at the water's edge.",
        "lines": [
            (N, "In 2004, Dave Shaw, an Australian pilot and record-breaking diver, reached the bottom.", 0.2),
            (N, "His torch found Deon.", 0.4, 0.95),
            (N, "Nobody asked him to go back. He offered.", 0.6, 0.95),
        ],
    },
    {
        "id": "c04_plan",
        "title": "The Plan",
        "visual": "A dive plan drawn on the cross-section: a fast line down, a 6-minute box at the bottom, then a long "
                  "staircase of decompression stops up the side, labelled 12 hours. Support divers' lights at stations.",
        "lines": [
            (N, "The plan: a few minutes at the bottom, then twelve hours climbing slowly back to the light.", 0.6),
        ],
    },
    {
        "id": "c05_wrong",
        "title": "What Went Wrong",
        "visual": "Bottom, -270 m. Gas molecules in the breathing loop drawn packed tight (density). The line, a thin "
                  "bright thread, loops around the torch light. CO2 icon fills; red warning pulses; the torch stops moving.",
        "lines": [
            (N, "At two hundred and seventy metres, the gas you breathe is so dense that every breath is hard work.", 0.25),
            (N, "As Dave worked, the body came loose, and his line tangled around him.", 0.25),
            (N, "Straining against it, he couldn't clear the carbon dioxide from his lungs.", 0.2),
            (N, "It built up, until he passed out.", 0.4, 0.92),
            (N, "He never came back up.", 1.0, 0.9),
        ],
    },
    {
        "id": "c06_shirley",
        "title": "Don Shirley",
        "visual": "Above, a second light (Shirley) sees the still light below and descends. At -250 m a crack/warning on "
                  "his rebreather unit; he turns back. A slate. Then a very long climb up the decompression staircase, "
                  "the world tilting (vertigo).",
        "lines": [
            (N, "His friend Don Shirley saw Dave's light below, not moving, and went down after him.", 0.25),
            (N, "At two hundred and fifty metres, the pressure crushed his rebreather's controls. He had to turn back.", 0.25),
            (N, "It took him twelve and a half hours to get back to the surface.", 0.7),
        ],
    },
    {
        "id": "c07_aftermath",
        "title": "Aftermath",
        "visual": "Four days later: the line is lifted; two soft lights rise together through the water toward the "
                  "surface caustics. The pool, the plaque. End on the line 'Dave Shaw did bring Deon home.' and loop.",
        "lines": [
            (N, "Four days later, as divers lifted the line, Dave and Deon rose to the surface together.", 0.4),
            (N, "Dave Shaw did bring Deon home.", 2.5, 0.92),
        ],
    },
]

# One Instagram Reel (9:16).
PARTS = [
    {"id": "reel", "title": "270 Metres Down", "scenes": [s["id"] for s in SCENES]},
]

SOURCES = [
    "Zimmermann, T. (2005). Raising the Dead. Outside, August 2005.",
    "Mitchell, S.J., Cronje, F.J. et al. (2007). Fatal respiratory failure during a technical rebreather dive at extreme pressure. "
    "Aviat Space Environ Med 78(2):81-86. PMID 17310877.",
    "IOL (Independent Online, South Africa), 10, 13 and 14 January 2005; Mail & Guardian, 9 January 2005; The Globe and Mail, January 2005.",
    "Finch, P. (2008). Raising the Dead (US: Diving into Darkness). HarperCollins.",
    "Divers Alert Network (DAN): articles on breathing-gas density and work of breathing at depth.",
    "Full fact table, confidence levels and myths avoided: RESEARCH.md.",
]


def write_markdown(path="SCRIPT.md"):
    out = ["# 270 Metres Down: the Boesmansgat dive", "",
           "_An animated true-incident Reel for Instagram (9:16). Narration + visual direction._", "",
           "> Real people: told factually and respectfully. No footage of the death, no graphic detail, no invented "
           "drama. Every line is sourced in RESEARCH.md.", ""]
    for n, sc in enumerate(SCENES, 1):
        out += [f"## Scene {n}: {sc['title']}", "", f"**Visual:** {sc['visual']}", ""]
        for voice, text, *_ in sc["lines"]:
            out.append(f"**NARRATOR:** {text}  ")
        out.append("")
    out += ["## Sources", ""] + [f"- {s}" for s in SOURCES] + [""]
    with open(path, "w") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    import os
    write_markdown(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SCRIPT.md"))
