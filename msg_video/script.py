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
  'title': 'Cold Open: The Accusations',
  'visual': "Wall of retro TV screens. Each lights up with a flat-vector silhouette 'celebrity' "
            "(dramatization) making a claim. Glitchy channel-switch between them. Ends with a neon 'NO MSG' "
            'restaurant sign buzzing on, then a record scratch.',
  'lines': [('TVDOC', 'MSG is an excitotoxin. It overexcites your brain cells.', 0.25),
            ('CHEF', "Never in my kitchen. It's poison, darling.", 0.25),
            ('GURU', "Every time I eat it, I get headaches. It's the MSG, you guys!", 0.6),
            ('NARR', 'For fifty years, three little letters have terrified dinner tables.', 0.3),
            ('NARR', 'But is MSG actually bad for you?', 0.35),
            ('NARR', "And if it isn't...", 0.15, 0.92),
            ('NARR', 'why does everyone think it is?', 0.8)]},
 {'id': 's02_title',
  'title': 'Title Card',
  'visual': "Kurzgesagt-style title: a glowing glutamate crystal spins in a starfield. Title 'MSG: The Most "
            "Misunderstood Molecule'.",
  'lines': [('NARR', 'MSG. The most misunderstood molecule in your kitchen.', 0.5)]},
 {'id': 's03_ikeda',
  'title': '1908: The Discovery of Umami',
  'visual': 'Tokyo 1908. Professor Ikeda at a table with a steaming bowl of kombu dashi. Four taste icons '
            '(sweet, sour, salty, bitter) appear, then a mysterious fifth. Lab montage: giant pile of '
            "seaweed boiled down to a few crystals. Word 'UMAMI' appears.",
  'lines': [('NARR',
             'In 1908, a Japanese chemist named Kikunae Ikeda noticed something odd about his seaweed soup.',
             0.25),
            ('NARR', "It wasn't sweet, sour, salty or bitter. It was something else.", 0.25),
            ('NARR', 'So he boiled down a mountain of seaweed, and found the culprit: glutamic acid.', 0.25),
            ('NARR', 'He named the taste umami, and paired it with sodium to make a seasoning.', 0.2),
            ('NARR', 'Monosodium glutamate.', 0.5, 0.92)]},
 {'id': 's04_everywhere',
  'title': 'Glutamate Is Everywhere',
  'visual': 'The glutamate molecule zooms out to reveal it inside tomatoes, mushrooms, soy sauce, parmesan, '
            'and a baby bottle / breast milk. Bar chart of free glutamate. Then two identical molecules side '
            "by side: 'from tomato' vs 'from MSG' — an equals sign.",
  'lines': [('NARR', 'Which sounds alarmingly chemical.', 0.3, 0.92),
            ('NARR', 'But glutamate is one of the most common amino acids on Earth.', 0.2),
            ('NARR', "It's in tomatoes, mushrooms, parmesan, even human breast milk.", 0.25),
            ('NARR',
             'And to your body, glutamate from a tomato and from MSG is exactly the same molecule.',
             0.5)]},
 {'id': 's05_body',
  'title': 'What Happens When You Eat It',
  'visual': 'Cutaway of a cute flat-style human body. Glutamate particles land on tongue receptors '
            '(click/glow). Travel down to the gut where little round intestine-cell characters gobble them '
            'up as fuel. 95% counter. Tiny remainder reaches blood, meets the blood-brain barrier as a '
            'castle wall with guards.',
  'lines': [('NARR', 'So what happens when you eat it?', 0.25),
            ('NARR', 'Your tongue tastes it, and tells your brain: protein!', 0.2),
            ('NARR', 'Then it reaches your gut... and gets eaten.', 0.25),
            ('NARR',
             'The cells lining your intestines burn glutamate as fuel. Around ninety-five percent never even '
             'makes it into your blood.',
             0.25),
            ('NARR',
             'And the rest hits the blood-brain barrier, one of the best-guarded borders in your body.',
             0.5)]},
 {'id': 's07_tests',
  'title': 'Putting MSG to the Test',
  'visual': 'Clean lab. Two identical capsules, labeled A and B, shuffled under cups. Volunteers with '
            "thought bubbles. Results chart: MSG vs placebo bars nearly equal. FDA-style 'GRAS' stamp lands "
            'next to salt, sugar, baking powder icons.',
  'lines': [('NARR',
             'Scientists have tested it again and again, in double-blind studies where nobody knows who got '
             'the real thing.',
             0.25),
            ('NARR', 'People who said they were sensitive to MSG reacted just as often to the fake.', 0.25),
            ('NARR',
             'Today, the FDA classifies MSG as generally recognized as safe. Just like salt and sugar.',
             0.5)]},
 {'id': 's08_twist',
  'title': 'Plot Twist: MSG vs Salt',
  'visual': 'Two shakers face off on a scale: SALT vs MSG. Sodium atoms pop out: salt has lots, MSG about a '
            'third. A heart icon smiles.',
  'lines': [('NARR',
             'Plot twist: MSG has about a third of the sodium of table salt. Swap some in: same flavor, less '
             'sodium.',
             0.5)]},
 {'id': 's06_origin',
  'title': 'How the Rumour Started',
  'visual': '1968: a typewritten letter flutters onto a desk, addressed to the New England Journal of '
            "Medicine. Question marks over wine, salt, MSG. A headline stamp slams down: 'CHINESE-RESTAURANT "
            "SYNDROME'. Newspapers multiply and spin. 1969: lab mouse + giant syringe. Then a split screen: "
            'tomatoes/parmesan (no alarm) vs takeout box (alarm bells).',
  'lines': [('NARR', 'So how did a seaweed seasoning become a villain?', 0.3),
            ('NARR',
             'In 1968, a doctor wrote to a medical journal. After eating at Chinese restaurants, he felt '
             'numb and weak.',
             0.2),
            ('NARR', 'Was it the wine? The salt? The MSG? It was a guess.', 0.25),
            ('NARR', 'But the journal gave it a catchy headline: Chinese Restaurant Syndrome.', 0.3),
            ('NARR', 'Then scientists injected huge doses into newborn mice, and found brain damage.', 0.2),
            ('NARR',
             'Scary. Until you remember that a baby mouse injection is nothing like dinner.',
             0.3,
             0.95),
            ('NARR',
             'And nobody panicked about tomato sauce or parmesan. The fear stuck to one cuisine, and the '
             'name said it out loud.',
             0.5)]},
 {'id': 's09_outro',
  'title': 'So... Is MSG Bad for You?',
  'visual': "Callback to the wall of TVs, now switching off one by one. The 'NO MSG' sign flickers and the "
            "'NO' burns out. Final shot: a cozy bowl of ramen with an MSG shaker, camera pulls back into "
            'space. End card.',
  'lines': [('NARR', 'So, is MSG bad for you?', 0.35),
            ('NARR', 'Not really.', 0.4, 0.9),
            ('NARR',
             "It's a story about how one guess, a catchy name, and a little prejudice outlived decades of "
             'science.',
             0.3),
            ('NARR', 'So next time you see a No MSG sign...', 0.2),
            ('NARR', 'maybe reach for the shaker.', 1.5)]}]

# One Instagram Reel (9:16), under 3 minutes. The earlier 3-part cut is in git history.
PARTS = [
    {"id": "reel", "title": "MSG: The Most Misunderstood Molecule", "scenes": ['s01_coldopen', 's02_title', 's03_ikeda', 's04_everywhere', 's05_body', 's07_tests', 's08_twist', 's06_origin', 's09_outro']},
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
 'lactation period. J Pediatr Gastroenterol Nutr 31:508-512.']


def write_markdown(path="SCRIPT.md"):
    out = ["# MSG: The Most Misunderstood Molecule", "",
           "_An animated explainer series for Instagram Reels (9:16, three parts). Narration + visual direction._", "",
           "> The cold-open 'celebrities' are fictional, silhouetted composite characters "
           "(dramatization) voiced by synthetic voices, representing claims commonly made on TV "
           "and social media. No real person is depicted or quoted.", ""]
    by_id = {sc["id"]: sc for sc in SCENES}
    n = 0
    for part in PARTS:
      out += [f"# {part['title']}", ""]
      for sid in part["scenes"]:
        sc = by_id[sid]
        n += 1
        out += [f"## Scene {n}: {sc['title']}", "", f"**Visual:** {sc['visual']}", ""]
        for voice, text, *_ in sc["lines"]:
            who = "NARRATOR" if voice == "NARR" else SPEAKER_LABELS[voice].upper()
            out.append(f"**{who}:** {text}  ")
        out.append("")
    out += ["## Sources", ""] + [f"- {s}" for s in SOURCES] + [""]
    with open(path, "w") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    import os
    write_markdown(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SCRIPT.md"))
