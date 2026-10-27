"""Single source of truth for the MSG explainer script.

Each scene is a list of lines: (voice, text, pause_after_seconds[, speed]).
The optional speed multiplies the voice's base pace, e.g. 0.92 to slow down a wry aside.
Voice "NARR" is the narrator; other keys are the cold-open "talking heads".
`python3 build_audio.py` turns this into narration WAVs + cues.json,
and `python3 script.py` writes SCRIPT.md.
"""

VOICES = {
    "NARR": "bm_george*0.5+am_michael*0.5",  # client pick "voice 9": blend of two Kokoro voices
    "TVDOC": "am_onyx",      # "Daytime TV Doctor"
    "CHEF": "bm_george",     # "Celebrity Chef"
    "GURU": "af_bella",      # "Wellness Influencer"
    "HOST": "am_eric",       # "Talk Show Host"
    "AUNT": "af_sarah",      # "Facebook Aunt"
}

SPEAKER_LABELS = {
    "TVDOC": "Daytime TV Doctor",
    "CHEF": "Celebrity Chef",
    "GURU": "Wellness Influencer",
    "HOST": "Talk Show Host",
    "AUNT": "Your Aunt, on Facebook",
}

SCENES = [{'id': 's01_coldopen',
  'title': 'Cold Open: The Claim',
  'visual': 'Two real, attributed clips of Dr. Russell Blaylock (retired neurosurgeon, lecture video posted '
            '2012) play inside a paper TV on a wall of TVs, with a name and credit tag. Pull back to the TV '
            "wall ('50+ years'), the NO MSG sign over restaurant tables, then an envelope lands on a 1968 "
            'desk and hands into the rumour story.',
  'lines': [('CLIP:blaylock_hungry',
             "That's why they put so much MSG in food because it makes you hungry and you eat more of their "
             'food.',
             0.35),
            ('CLIP:blaylock_brain', 'Excitotoxicity destroys brain cells.', 0.5),
            ('NARR',
             'For over fifty years, people have been scared of MSG. And it all started with one letter, in '
             '1968.',
             0.4)]},
 {'id': 's06_origin',
  'title': 'How the Rumour Started',
  'visual': '1968: a typewritten letter flutters onto a desk, addressed to the New England Journal of '
            "Medicine. Question marks over wine, salt, MSG. A headline stamp slams down: 'CHINESE-RESTAURANT "
            "SYNDROME'. Newspapers multiply and spin. 1969: lab mouse + giant syringe. Then a split screen: "
            'tomatoes/parmesan (no alarm) vs takeout box (alarm bells).',
  'lines': [('NARR', 'So how did a seaweed seasoning become a villain?', 0.25),
            ('NARR',
             'In 1968, a doctor wrote to a medical journal: after Chinese food, he felt numb and weak.',
             0.2),
            ('NARR', 'Wine? Salt? MSG? He was guessing.', 0.2),
            ('NARR', 'But it ran under a catchy headline: Chinese Restaurant Syndrome.', 0.3),
            ('NARR',
             'Then mice injected with huge doses got brain damage. But a huge dose in a baby mouse is '
             'nothing like eating dinner.',
             0.25),
            ('NARR',
             'And nobody panicked about parmesan. The fear stuck to one cuisine, and the name said it out '
             'loud.',
             0.4),
            ('NARR', 'But was he right? Is MSG actually bad for you?', 0.5)]},
 {'id': 's03_ikeda',
  'title': '1908: The Discovery of Umami',
  'visual': 'Tokyo 1908. Professor Ikeda at a table with a steaming bowl of kombu dashi. Four taste icons '
            '(sweet, sour, salty, bitter) appear, then a mysterious fifth. Lab montage: giant pile of '
            "seaweed boiled down to a few crystals. Word 'UMAMI' appears.",
  'lines': [('NARR',
             'In 1908, a Japanese chemist named Kikunae Ikeda tasted something odd in his seaweed soup.',
             0.25),
            ('NARR',
             'Not sweet, sour, salty or bitter. He called it umami, and turned it into a seasoning.',
             0.2),
            ('NARR', 'Monosodium glutamate.', 0.4, 0.92)]},
 {'id': 's04_everywhere',
  'title': 'Glutamate Is Everywhere',
  'visual': 'The glutamate molecule zooms out to reveal it inside tomatoes, mushrooms, soy sauce, parmesan, '
            'and a baby bottle / breast milk. Bar chart of free glutamate. Then two identical molecules side '
            "by side: 'from tomato' vs 'from MSG' — an equals sign.",
  'lines': [('NARR', 'Which sounds alarmingly chemical.', 0.25, 0.92),
            ('NARR', 'But glutamate is everywhere: tomatoes, parmesan, even breast milk.', 0.2),
            ('NARR', 'To your body, glutamate from a tomato and from MSG is the exact same molecule.', 0.4)]},
 {'id': 's05_body',
  'title': 'What Happens When You Eat It',
  'visual': 'Cutaway of a cute flat-style human body. Glutamate particles land on tongue receptors '
            '(click/glow). Travel down to the gut where little round intestine-cell characters gobble them '
            'up as fuel. 95% counter. Tiny remainder reaches blood, meets the blood-brain barrier as a '
            'castle wall with guards.',
  'lines': [('NARR', 'So what happens when you eat it?', 0.2),
            ('NARR',
             'The cells lining your gut burn glutamate as fuel. Around ninety-five percent never even '
             'reaches your blood.',
             0.25),
            ('NARR',
             'And the rest hits the blood-brain barrier, one of the best-guarded borders in your body.',
             0.4)]},
 {'id': 's07_tests',
  'title': 'Putting MSG to the Test',
  'visual': 'Clean lab. Two identical capsules, labeled A and B, shuffled under cups. Volunteers with '
            "thought bubbles. Results chart: MSG vs placebo bars nearly equal. FDA-style 'GRAS' stamp lands "
            'next to salt, sugar, baking powder icons.',
  'lines': [('NARR',
             "In double-blind studies, people who blamed MSG couldn't reliably tell it from a placebo.",
             0.25),
            ('NARR', 'The FDA calls it generally recognized as safe. Just like salt.', 0.4)]},
 {'id': 's08_twist',
  'title': 'Plot Twist: MSG vs Salt',
  'visual': 'Two shakers face off on a scale: SALT vs MSG. Sodium atoms pop out: salt has lots, MSG about a '
            'third. A heart icon smiles.',
  'lines': [('NARR',
             'Plot twist: it has a third of the sodium of table salt. Big flavor, less sodium.',
             0.4)]},
 {'id': 's09_outro',
  'title': 'So... Is MSG Bad for You?',
  'visual': "Callback to the wall of TVs, now switching off one by one. The 'NO MSG' sign flickers and the "
            "'NO' burns out. Final shot: a cozy bowl of ramen with an MSG shaker, camera pulls back into "
            'space. End card.',
  'lines': [('NARR', 'So, is MSG bad for you?', 0.3),
            ('NARR', 'Not really.', 0.35, 0.9),
            ('NARR', 'A guess, a catchy name, and a little prejudice outlived decades of science.', 0.25),
            ('NARR', 'So next time you see a No MSG sign... reach for the shaker.', 1.2)]}]

# One Instagram Reel (9:16), about 2 minutes. The earlier 3-part cut is in git history.
PARTS = [
    {"id": "reel", "title": "MSG: The Most Misunderstood Molecule", "scenes": ["s01_coldopen", "s06_origin", "s03_ikeda", "s04_everywhere", "s05_body", "s07_tests", "s08_twist", "s09_outro"]},
]

SOURCES = ['Kwok, R.H.M. (1968). Chinese-Restaurant Syndrome. N Engl J Med 278:796.',
 'Olney, J.W. (1969). Brain lesions, obesity, and other disturbances in mice treated with monosodium '
 'glutamate. Science 164:719-721.',
 'Geha, R.S. et al. (2000). Review of alleged reaction to monosodium glutamate and outcome of a multicenter '
 'double-blind placebo-controlled study. J Nutr 130(4S):1058S-1062S.',
 'Reeds, P.J. et al. (2000). Intestinal glutamate metabolism. J Nutr 130(4S):978S-982S.',
 'Ikeda, K. (1909/2002 transl.). New seasonings. Chem Senses 27:847-849.',
 'US FDA. Questions and Answers on Monosodium Glutamate (MSG).',
 "JECFA (1988). L-glutamic acid and its salts: ADI 'not specified'.",
 'Agostoni, C. et al. (2000). Free glutamine and glutamic acid increase in human milk through a three-month '
 'lactation period. J Pediatr Gastroenterol Nutr 31:508-512.',
 'Hawkins, R.A. (2009). The blood-brain barrier and glutamate. Am J Clin Nutr 90(3):867S-874S.',
 "Mosby, I. (2009). 'That won-ton soup headache': the Chinese restaurant syndrome, MSG and the making of "
 'American food, 1968-1980. Soc Hist Med 22(1):133-151. https://doi.org/10.1093/shm/hkn098',
 'US Code of Federal Regulations, 21 CFR 182.1(a): salt, pepper, vinegar, baking powder and monosodium '
 'glutamate listed as GRAS.',
 'EFSA Panel on Food Additives (2017). Re-evaluation of glutamic acid and glutamates as food additives. EFSA '
 'Journal 15(7):4910.',
 'FoodNavigator (2019-11-08). Study looks at reducing sodium intake through MSG substitution in salty food categories. https://www.foodnavigator.com/Article/2019/11/08/Study-looks-at-reducing-sodium-intake-through-MSG-substitution-in-salty-food-categories/',
 'Foods / J Food Sci (see FACTCHECK.md for the exact citation used).',
 'Independent fact-check of every claim in this Reel: FACTCHECK.md.']


def write_markdown(path="SCRIPT.md"):
    out = ["# MSG: The Most Misunderstood Molecule", "",
           "_An animated explainer Reel for Instagram (9:16, about 2 minutes). Narration + visual direction._", "",
           "> The cold open uses two real, attributed clips of Dr. Russell Blaylock from a 2012 lecture "
           "(archived at https://archive.org/details/youtube-Y0kvaulUIfc), quoted verbatim and checked against "
           "two independent machine transcripts (small and medium Whisper models agree word for word; a final human listen is still recommended). See clips/clips.json and clips/CANDIDATES.md.", ""]
    by_id = {sc["id"]: sc for sc in SCENES}
    n = 0
    for part in PARTS:
      out += [f"# {part['title']}", ""]
      for sid in part["scenes"]:
        sc = by_id[sid]
        n += 1
        out += [f"## Scene {n}: {sc['title']}", "", f"**Visual:** {sc['visual']}", ""]
        for voice, text, *_ in sc["lines"]:
            who = ("NARRATOR" if voice == "NARR" else f"[REAL CLIP {voice[5:]}]" if voice.startswith("CLIP:")
                   else SPEAKER_LABELS[voice].upper())
            out.append(f"**{who}:** {text}  ")
        out.append("")
    out += ["## Sources", ""] + [f"- {s}" for s in SOURCES] + [""]
    with open(path, "w") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    import os
    write_markdown(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SCRIPT.md"))
