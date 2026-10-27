"""Single source of truth for the MSG explainer script.

Each scene is a list of lines: (voice, text, pause_after_seconds).
Voice "NARR" is the narrator; other keys are the cold-open "talking heads".
`python3 build_audio.py` turns this into narration WAVs + cues.json,
and `python3 script.py` writes SCRIPT.md.
"""

VOICES = {
    "NARR": "am_michael",
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

SCENES = [
    {
        "id": "s01_coldopen",
        "title": "Cold Open: The Accusations",
        "visual": "Wall of retro TV screens. Each lights up with a flat-vector silhouette "
                  "'celebrity' (dramatization) making a claim. Glitchy channel-switch between them. "
                  "Ends with a neon 'NO MSG' restaurant sign buzzing on, then a record scratch.",
        "lines": [
            ("HOST", "Coming up after the break: the hidden chemical in your takeout that could be making you sick!", 0.35),
            ("TVDOC", "MSG is an excitotoxin. It overexcites your brain cells. Literally to death.", 0.35),
            ("CHEF", "We never use MSG in my kitchen. Never! It's poison, darling.", 0.35),
            ("GURU", "Every time I eat Chinese food I get headaches, my heart races... it's the MSG, you guys!", 0.35),
            ("AUNT", "Sharing this for everyone. MSG is basically brain poison! Please read!!", 0.9),
            ("NARR", "For more than fifty years, three little letters have terrified diners around the world.", 0.3),
            ("NARR", "Restaurants hang signs promising they don't use it. Food labels proudly brag about leaving it out.", 0.3),
            ("NARR", "But what actually is MSG? Is it really bad for you?", 0.3),
            ("NARR", "And if it isn't... why does almost everyone think it is?", 1.2),
        ],
    },
    {
        "id": "s02_title",
        "title": "Title Card",
        "visual": "Kurzgesagt-style title: a glowing glutamate crystal spins in a starfield. "
                  "Title 'MSG: The Most Misunderstood Molecule'.",
        "lines": [
            ("NARR", "MSG. The most misunderstood molecule in your kitchen.", 1.6),
        ],
    },
    {
        "id": "s03_ikeda",
        "title": "1908: The Discovery of Umami",
        "visual": "Tokyo 1908. Professor Ikeda at a table with a steaming bowl of kombu dashi. "
                  "Four taste icons (sweet, sour, salty, bitter) appear, then a mysterious fifth. "
                  "Lab montage: giant pile of seaweed boiled down to a few crystals. Word 'UMAMI' appears.",
        "lines": [
            ("NARR", "Our story begins in Tokyo, in 1908.", 0.3),
            ("NARR", "A chemist named Kikunae Ikeda was eating a bowl of soup made with kombu, a kind of seaweed.", 0.3),
            ("NARR", "He noticed a taste that wasn't sweet, or sour, or salty, or bitter. It was something else. Savory. Meaty. Deep.", 0.4),
            ("NARR", "So he went to his lab, boiled down huge amounts of seaweed, and found the source: a substance called glutamic acid.", 0.3),
            ("NARR", "He named the taste umami. Roughly: delicious taste.", 0.3),
            ("NARR", "To turn it into a seasoning, he paired it with sodium, making a stable, salt-like crystal. Monosodium glutamate.", 0.3),
            ("NARR", "Within a year, it was on sale.", 1.0),
        ],
    },
    {
        "id": "s04_everywhere",
        "title": "Glutamate Is Everywhere",
        "visual": "The glutamate molecule zooms out to reveal it inside tomatoes, mushrooms, "
                  "soy sauce, parmesan, and a baby bottle / breast milk. Bar chart of free glutamate. "
                  "Then two identical molecules side by side: 'from tomato' vs 'from MSG' — an equals sign.",
        "lines": [
            ("NARR", "Here's the thing. Glutamate isn't some strange lab-made chemical.", 0.3),
            ("NARR", "It's one of the most common amino acids on Earth, a building block of the proteins in every living thing.", 0.3),
            ("NARR", "Ripe tomatoes are full of it. So are mushrooms, soy sauce, and parmesan cheese, one of the richest sources of glutamate there is.", 0.3),
            ("NARR", "Human breast milk contains around ten times more free glutamate than cow's milk. Our very first meal is umami.", 0.4),
            ("NARR", "Your body even makes its own glutamate, every single day.", 0.3),
            ("NARR", "And to your body, glutamate from a tomato and glutamate from a shaker of MSG are exactly the same molecule.", 0.5),
            ("NARR", "So what actually happens when you eat it? That's part two.", 1.5),
        ],
    },
    {
        "id": "s05_body",
        "title": "What Happens When You Eat It",
        "visual": "Cutaway of a cute flat-style human body. Glutamate particles land on tongue receptors "
                  "(click/glow). Travel down to the gut where little round intestine-cell characters gobble "
                  "them up as fuel. 95% counter. Tiny remainder reaches blood, meets the blood-brain barrier "
                  "as a castle wall with guards.",
        "lines": [
            ("NARR", "Almost all the MSG you eat never even makes it into your blood. Here's why.", 0.5),
            ("NARR", "So what happens when you eat it?", 0.3),
            ("NARR", "First, glutamate docks onto special receptors on your tongue, and sends your brain a message: this food has protein in it. Yum.", 0.3),
            ("NARR", "Then it reaches your gut. And this is where the scary story starts to fall apart.", 0.3),
            ("NARR", "The cells lining your intestines love glutamate. They burn it as fuel.", 0.3),
            ("NARR", "Almost all the glutamate you eat, around ninety-five percent, is used up right there in your gut, before it ever reaches your bloodstream.", 0.4),
            ("NARR", "And the little that slips through faces another wall: the blood-brain barrier, a tightly guarded border that controls what gets into your brain.", 0.3),
            ("NARR", "The glutamate on your dinner plate doesn't just stroll into your neurons.", 1.0),
        ],
    },
    {
        "id": "s07_tests",
        "title": "Putting MSG to the Test",
        "visual": "Clean lab. Two identical capsules, labeled A and B, shuffled under cups. Volunteers with "
                  "thought bubbles. Results chart: MSG vs placebo bars nearly equal. FDA-style 'GRAS' stamp "
                  "lands next to salt, sugar, baking powder icons.",
        "lines": [
            ("NARR", "Scientists took the claims seriously, and tested them. Again, and again.", 0.3),
            ("NARR", "The gold standard is a double-blind study: people get either MSG or a placebo, and nobody, not even the researchers, knows who got which.", 0.3),
            ("NARR", "Over and over, people who said they were sensitive to MSG reacted just as often to the fake.", 0.3),
            ("NARR", "Huge doses on an empty stomach caused mild, short-lived symptoms in some people. But the results weren't consistent, and nothing like that showed up with normal amounts in normal food.", 0.4),
            ("NARR", "Today, the US Food and Drug Administration lists MSG as generally recognized as safe, the same category as salt, sugar, and baking powder.", 1.0),
        ],
    },
    {
        "id": "s08_twist",
        "title": "Plot Twist: MSG vs Salt",
        "visual": "Two shakers face off on a scale: SALT vs MSG. Sodium atoms pop out: salt has lots, "
                  "MSG about a third. A heart icon smiles.",
        "lines": [
            ("NARR", "Here's a plot twist. MSG might actually help.", 0.3),
            ("NARR", "Gram for gram, it has only about a third of the sodium of table salt.", 0.3),
            ("NARR", "Swap some of the salt in a recipe for MSG, and you can cut the sodium in a dish noticeably, while keeping all the flavor.", 0.5),
            ("NARR", "So if the science is this clear, how did MSG end up with such a terrible reputation? That's part three.", 1.5),
        ],
    },
    {
        "id": "s06_origin",
        "title": "How the Rumour Started",
        "visual": "1968: a typewritten letter flutters onto a desk, addressed to the New England Journal of "
                  "Medicine. Question marks over wine, salt, MSG. A headline stamp slams down: "
                  "'CHINESE-RESTAURANT SYNDROME'. Newspapers multiply and spin. 1969: lab mouse + giant syringe. "
                  "Then a split screen: tomatoes/parmesan (no alarm) vs takeout box (alarm bells).",
        "lines": [
            ("NARR", "Fifty years of fear around MSG started with a single letter.", 0.5),
            ("NARR", "So how did a seaweed seasoning become public enemy number one?", 0.3),
            ("NARR", "It started in 1968, with a letter.", 0.3),
            ("NARR", "A doctor wrote to the New England Journal of Medicine, describing numbness, weakness and palpitations after eating at Chinese restaurants in the United States.", 0.3),
            ("NARR", "He wondered what might be causing it. The cooking wine? The salt? Maybe the MSG? He didn't know. It was a guess.", 0.3),
            ("NARR", "But the journal printed it under a catchy headline: Chinese Restaurant Syndrome.", 0.3),
            ("NARR", "Letters poured in. Newspapers ran with it. Almost overnight, a guess had become a diagnosis.", 0.4),
            ("NARR", "A year later, a scientist injected huge doses of MSG under the skin of newborn mice, and found brain damage. The headlines got even scarier.", 0.3),
            ("NARR", "But injecting a massive dose into a baby mouse is nothing like eating a seasoned meal. Remember: your gut eats almost all of it first.", 0.4),
            ("NARR", "And there was something else going on. In 1960s America, Chinese food was seen as foreign and suspicious.", 0.3),
            ("NARR", "Nobody blamed the tomato sauce. Nobody worried about Italian Restaurant Syndrome. The fear stuck to one cuisine, and the name said it out loud.", 1.0),
        ],
    },
    {
        "id": "s09_outro",
        "title": "So... Is MSG Bad for You?",
        "visual": "Callback to the wall of TVs, now switching off one by one. The 'NO MSG' sign flickers "
                  "and the 'NO' burns out. Final shot: a cozy bowl of ramen with an MSG shaker, camera pulls "
                  "back into space. End card.",
        "lines": [
            ("NARR", "So, is MSG bad for you?", 0.3),
            ("NARR", "In absurd amounts, almost anything is. Even water. But in food, the evidence says it's about as dangerous as the tomato on your pizza.", 0.4),
            ("NARR", "The story of MSG is really a story about how one guess, a catchy name, and a little bit of prejudice can outlive decades of science.", 0.4),
            ("NARR", "So the next time you see a No MSG sign, you'll know the real story.", 0.3),
            ("NARR", "And maybe... you'll reach for the shaker.", 2.5),
        ],
    },
]

# Instagram Reel series: 3 standalone parts (9:16), each under 3 minutes.
PARTS = [
    {"id": "part1", "title": "Part 1: The Accusation", "scenes": ["s01_coldopen", "s02_title", "s03_ikeda", "s04_everywhere"]},
    {"id": "part2", "title": "Part 2: The Science", "scenes": ["s05_body", "s07_tests", "s08_twist"]},
    {"id": "part3", "title": "Part 3: The Rumour", "scenes": ["s06_origin", "s09_outro"]},
]

SOURCES = [
    "Kwok, R.H.M. (1968). Chinese-Restaurant Syndrome. N Engl J Med 278:796.",
    "Olney, J.W. (1969). Brain lesions, obesity, and other disturbances in mice treated with monosodium glutamate. Science 164:719-721.",
    "Geha, R.S. et al. (2000). Review of alleged reaction to monosodium glutamate and outcome of a multicenter double-blind placebo-controlled study. J Nutr 130(4S):1058S-1062S.",
    "Reeds, P.J. et al. (2000). Intestinal glutamate metabolism. J Nutr 130(4S):978S-982S.",
    "Ikeda, K. (1909/2002 transl.). New seasonings. Chem Senses 27:847-849.",
    "US FDA. Questions and Answers on Monosodium Glutamate (MSG).",
    "JECFA (1988). L-glutamic acid and its salts: ADI 'not specified'.",
    "Agostoni, C. et al. (2000). Free glutamine and glutamic acid increase in human milk through a three-month lactation period. J Pediatr Gastroenterol Nutr 31:508-512.",
]


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
        for voice, text, _ in sc["lines"]:
            who = "NARRATOR" if voice == "NARR" else SPEAKER_LABELS[voice].upper()
            out.append(f"**{who}:** {text}  ")
        out.append("")
    out += ["## Sources", ""] + [f"- {s}" for s in SOURCES] + [""]
    with open(path, "w") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    import os
    write_markdown(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SCRIPT.md"))
