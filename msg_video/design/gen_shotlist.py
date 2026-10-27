#!/usr/bin/env python3
"""Shot-list generator for the MSG Reel (one 9:16 Instagram Reel, Kombu Paper style).

Single source for design/shotlist.json and design/SHOTLIST.md:
    python3 design/gen_shotlist.py
All times come from cues.json (line starts + per-word "words" timings) via L()/W()/B(). After a narration
re-record, re-run this script: it fails loudly if a referenced word disappeared, shots stop tiling, a label
breaks the one-label rule, or a forbidden (dull-colour) word sneaks into a description.
Edit the shot data below, never the generated files.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUES = json.load(open(os.path.join(ROOT, "cues.json")))
sys.dont_write_bytecode = True
sys.path.insert(0, ROOT)
import script  # noqa: E402  (PARTS)

ALLOWED = set("""whoosh whoosh_long swoosh_up swoosh_down pop pop_small bubble blip click tick ding chime sparkle
impact_soft impact_big thud stamp boing riser record_scratch tv_on tv_off channel_switch static glitch
neon_buzz neon_flicker typewriter paper heartbeat alarm crunch slurp chomp sizzle bubbling_pot cash error success""".split())
DULL = re.compile(r"\b(gr[ae]y|beige|cream|sepia|muted|desaturat\w*|drab|muddy|brown)\b", re.I)

FRAME = {
    "size": [1080, 1920],
    "ui_top": [0, 220], "ui_bottom": [1480, 1920], "right_gutter_x": [940, 1080], "left_margin_x": [0, 40],
    "caption_band": [1300, 1480],
    "key_box": {"x": [40, 940], "y": [220, 1300], "center_x": 490},
    "zones": {"TOP": [220, 520], "MID": [520, 1100], "LOW": [1100, 1300]},
}

SCENES, CUR = {}, None


def scene(sid):
    global CUR
    CUR = sid


def L(n):
    return CUES[CUR]["lines"][n - 1]["start"]


def _norm(w):
    return re.sub(r"[^a-z0-9\-']", "", w.lower())


def W(n, word, k=1):
    hits = [w for w in CUES[CUR]["lines"][n - 1]["words"] if _norm(w["w"]) == _norm(word)]
    if len(hits) < k:
        raise SystemExit(f"{CUR} L{n}: word '{word}' #{k} not found")
    return round(hits[k - 1]["start"], 3)


def B(n, word, k=1):
    return round(W(n, word, k) - 0.1, 3)


D = lambda: CUES[CUR]["duration"]  # noqa: E731


def shot(start, end, size, action, top, mid, low, camera, labels, t_in, t_out, sfx, props=(), hero=None,
         notes=None, drama=False):
    SCENES.setdefault(CUR, []).append(dict(
        start=round(start, 3), end=round(end, 3), shot_size=size, on_screen=action,
        layout={"TOP": top, "MID": mid, "LOW": low}, camera=camera,
        labels=[{"text": a, "in": round(b, 3), "out": round(c if c is not None else end, 3)} for a, b, c in labels],
        dramatization_tag=drama, transition_in=t_in, transition_out=t_out,
        sfx=[dict(t=round(a, 3), name=b, gain=c) for a, b, c in sfx], props=list(props), hero=hero, notes=notes))


SLIDE_IN, SLIDE_OUT = "Paper slide (sheet slides over, shadow leading)", "Paper slide"
RIPPLE_OUT = "Gold Ripple wipe (act change)"
DRAMA = "torn-paper lower third: role + 'DRAMATIZATION - fictional character'"

# ================================================================ s01 cold open (HOOK)
scene("s01_coldopen")
shot(0.0, L(2), "MEDIUM (style frame A)",
     "HOOK, lit on frame 0 = style frame A: violet TV wall, 8 dim grape sets + the turquoise hero TV spanning two "
     "columns with a lemon screen. The TV Doctor shadow puppet (head mirror, stethoscope, orange stick) jabs a finger "
     "at a pink paper brain; tomato-red zap bolts burst from it on 'overexcites'.",
     "Dim TV sets", "Hero TV: TV Doctor puppet + zapping brain", DRAMA,
     "Slow multiplane push-in (3%).",
     [("TV doctor", 0.0, None)], "NONE: lit on frame 0 (hook; also the loop landing point)", "Channel-switch flicker",
     [(0.0, "tv_on", 0.5), (W(1, "excitotoxin"), "static", 0.25), (W(1, "overexcites"), "glitch", 0.35), (W(1, "brain"), "blip", 0.3)],
     ["TVWALL", "PUPPETS", "DRAMA_LT", "CAPTION"], hero="Hook: TV Doctor puppet + zapping brain inside 0.3 s", drama=True,
     notes="Frame 0 must match the last frame of s09 (same TV wall layout) so the Reel loops.")
shot(L(2), L(3), "MEDIUM",
     "Channel flick: the hero screen now shows the Celebrity Chef puppet (tall toque, ladle) stirring a pot; on "
     "'poison' the ladle flicks the MSG shaker off the counter and a tomato-red X stamp lands on it.",
     "Dim TV sets", "Hero TV: Chef puppet + shaker", DRAMA,
     "Locked, multiplane drift.",
     [("Celebrity chef", L(2), None)], "Channel-switch flicker", "Channel-switch flicker",
     [(L(2) - 0.05, "channel_switch", 0.55), (W(2, "never"), "pop_small", 0.3), (W(2, "poison,"), "stamp", 0.4)],
     ["TVWALL", "PUPPETS", "DRAMA_LT", "SHAKER"], drama=True)
shot(L(3), L(4), "MEDIUM",
     "Channel flick: the Wellness Influencer puppet (ring-light halo, topknot, phone) clutches its head; tomato-red "
     "headache zig-zags on 'headaches'; the MSG shaker pops into the phone frame on 'MSG'.",
     "Dim TV sets", "Hero TV: Influencer puppet inside a phone shape", DRAMA,
     "Locked, multiplane drift.",
     [("Wellness influencer", L(3), None)], "Channel-switch flicker", "Fast pull-back",
     [(L(3) - 0.05, "channel_switch", 0.55), (W(3, "headaches."), "pop_small", 0.35), (W(3, "msg,"), "pop", 0.35)],
     ["TVWALL", "PUPPETS", "DRAMA_LT", "SHAKER"], drama=True)
shot(L(4), B(4, "terrified"), "EXTREME WIDE",
     "Pull back: every set on the wall lights up with the three puppets repeating their claims; the wall keeps "
     "receding (it may bleed into the UI zones as decor).",
     "Upper TV rows", "Full TV wall, all screens lit lemon", "Lower TV rows",
     "Pull-back, multiplane layers separating.",
     [("50+ years", W(4, "fifty"), None)], "Fast pull-back", "Paper slide",
     [(L(4), "whoosh", 0.45), (W(4, "fifty"), "pop", 0.4), (W(4, "letters"), "tv_on", 0.3)],
     ["TVWALL", "PUPPETS", "DRAMA_LT"], drama=True, notes="Keep the DRAMATIZATION tag on screen while puppets are readable.")
shot(B(4, "terrified"), L(5), "MEDIUM",
     "Dinner tables around the world: three small tables stacked down the frame (different vivid cloths), each with "
     "a folded NO MSG card; at top, a restaurant window with the paper 'neon' NO MSG sign switching on.",
     "Restaurant window + NO MSG sign", "Table 1 / table 2", "Table 3",
     "Tilt down the stack.",
     [("No MSG", W(4, "terrified"), None)], "Paper slide", "Cut",
     [(W(4, "terrified"), "neon_flicker", 0.45), (W(4, "dinner"), "pop_small", 0.3), (W(4, "world."), "pop_small", 0.3)],
     ["SIGN"], notes="First appearance of the NO MSG sign (paid off in s09). Paper 'neon': flat colour tubes, no glow.")
shot(L(5), B(5, "and"), "CLOSE",
     "The MSG shaker alone on a turquoise table under a cut-paper spotlight; a big ink '?' cut-out drops above it on 'bad'.",
     "'?' cut-out", "Shaker", "Table edge",
     "Very slow push-in.",
     [], "Cut", "Freeze on record scratch",
     [(W(5, "msg"), "pop", 0.4), (W(5, "bad"), "boing", 0.3)],
     ["SHAKER"])
shot(B(5, "and"), D(), "CLOSE -> WIDE",
     "Record scratch after 'isn't...': freeze 0.3 s (boil stops). On 'why' pull back: the shaker sits in front of the "
     "TV wall, all three puppets pointing at it.",
     "TV wall", "Puppets pointing", "Shaker",
     "Freeze, then pull-back.",
     [], "Freeze (no cut)", "Paper slide, the sheet carries a falling gold drop into s02",
     [(W(5, "why") - 0.35, "record_scratch", 0.65), (W(5, "why"), "pop", 0.4), (D() - 0.6, "whoosh", 0.4)],
     ["SHAKER", "TVWALL", "PUPPETS", "DRAMA_LT"], drama=True, hero="Record scratch: 'why does everyone think it is?'")

# ================================================================ s02 title
scene("s02_title")
shot(0.0, 1.2, "MEDIUM",
     "A single gold drop falls into a broth surface on 'MSG.'; the Fifth Ripple spreads (4 white rings, the fifth "
     "gold and wider); Goldie pops up out of the centre.",
     "Drop falling", "Fifth Ripple + Goldie", "Title slip",
     "Locked.",
     [("MSG", W(1, "msg."), None)], "Paper slide", "Continuous",
     [(0.05, "swoosh_down", 0.4), (W(1, "msg."), "bubble", 0.5), (W(1, "msg.") + 0.1, "impact_soft", 0.6)],
     ["GOLDIE", "RIPPLE"], notes="Replace the old 16:9 scenes/s02_title.js (rejected style).")
shot(1.2, D(), "MEDIUM",
     "Goldie bobs in the gold ring and smiles at camera (canonical Goldie introduction); rings keep spreading.",
     "—", "Goldie in the ripple", "'MSG' title slip (Young Serif)",
     "Slow push-in.",
     [("MSG", 1.2, None)], "Continuous", RIPPLE_OUT,
     [(W(1, "misunderstood"), "sparkle", 0.5), (W(1, "kitchen."), "pop", 0.4)],
     ["GOLDIE", "RIPPLE"], hero="Title: Goldie and the Fifth Ripple")

# ================================================================ s03 Ikeda
scene("s03_ikeda")
shot(0.0, B(1, "noticed"), "MEDIUM",
     "1908: a bright period laboratory-kitchen; Kikunae Ikeda (respectful paper doll: high collar, round glasses, "
     "moustache) carries a steaming bowl of seaweed soup to his desk.",
     "Date slip", "Ikeda + bowl", "Desk",
     "Lateral truck with Ikeda.",
     [("1908", W(1, "1908,"), None)], "Gold Ripple reveal", "Continuous",
     [(W(1, "1908,"), "pop", 0.4), (W(1, "chemist"), "paper", 0.3), (W(1, "ikeda"), "pop_small", 0.3)],
     ["IKEDA"])
shot(B(1, "noticed"), L(2), "MEDIUM-CLOSE",
     "Ikeda sips; a kombu-green strip floats in the bowl; his eyebrows lift and a '?' cut-out pops on 'odd'.",
     "'?' cut-out", "Ikeda sipping", "Bowl with kombu strip",
     "Push-in.",
     [], "Continuous", "Push into the bowl",
     [(W(1, "noticed"), "slurp", 0.35), (W(1, "odd"), "pop_small", 0.35)],
     ["IKEDA"])
shot(L(2), B(2, "it", 2), "CLOSE (overhead on bowl)",
     "Fifth Ripple motif: rings spread from the bowl, each carrying a taste icon that pops on its word (candy = "
     "sweet, lemon = sour, salt crystal = salty, coffee bean = bitter).",
     "Sweet / sour icons", "Bowl + rings", "Salty / bitter icons",
     "Locked overhead.",
     [], "Push-in", "Continuous",
     [(W(2, "sweet,"), "pop", 0.35), (W(2, "sour,"), "pop", 0.35), (W(2, "salty"), "pop", 0.35), (W(2, "bitter."), "pop", 0.35)],
     ["RIPPLE", "TASTE_ICONS"], notes="Icons only, no words: captions say them.")
shot(B(2, "it", 2), L(3), "CLOSE",
     "The fifth ring blooms wide and gold; a tiny Goldie rises from the centre.",
     "—", "Fifth gold ring + Goldie", "—",
     "Slight push-in.",
     [("Savory", W(2, "savory."), None)], "Continuous", "Paper slide",
     [(W(2, "else."), "sparkle", 0.45), (W(2, "savory."), "chime", 0.45)],
     ["RIPPLE", "GOLDIE", "TASTE_ICONS"])
shot(L(3), B(3, "and"), "WIDE (tall)",
     "A mountain of kombu-green seaweed strips tumbles from the top of frame into a giant pot; Ikeda is tiny beside it.",
     "Seaweed falling", "Seaweed mountain + pot", "Ikeda on a stool",
     "Tilt down with the falling seaweed.",
     [], "Paper slide", "Time-lapse",
     [(W(3, "boiled"), "thud", 0.45), (W(3, "mountain"), "bubbling_pot", 0.35)],
     ["IKEDA"])
shot(B(3, "and"), L(4), "MEDIUM",
     "Time-lapse: the pile boils down in paper steam puffs to a dish of crystals; Goldie peeks out and a flat cut-paper "
     "skeletal diagram of glutamic acid slides in beside it.",
     "Steam puffs", "Dish + Goldie + skeletal diagram", "—",
     "Tilt down, push-in.",
     [("Glutamic acid", W(3, "glutamic"), None)], "Time-lapse", "Cut",
     [(W(3, "and"), "swoosh_down", 0.35), (W(3, "culprit:"), "sparkle", 0.4), (W(3, "glutamic"), "pop", 0.4)],
     ["GOLDIE", "GLU_DIAGRAM"])
shot(L(4), B(4, "monosodium"), "MEDIUM",
     "Ikeda holds up the crystal dish; the fifth gold ring returns behind it. Then Goldie meets a small blue sodium "
     "ion and they link mittens on 'sodium'.",
     "Umami slip (Young Serif)", "Goldie + sodium ion", "Ikeda's hands",
     "Slow push-in.",
     [("Umami", W(4, "umami,"), W(4, "paired"))], "Cut", "Continuous",
     [(W(4, "umami,"), "impact_soft", 0.6), (W(4, "umami,") + 0.15, "chime", 0.4), (W(4, "sodium"), "click", 0.45)],
     ["IKEDA", "GOLDIE", "NA", "RIPPLE"], hero="Umami is named")
shot(B(4, "monosodium"), D(), "MEDIUM",
     "Goldie and the sodium ion hop into the MSG shaker (first canonical appearance); its cap snaps on.",
     "—", "Shaker", "'MSG' slip",
     "Locked.",
     [("MSG", W(4, "monosodium"), None)], "Continuous", SLIDE_OUT,
     [(W(4, "monosodium"), "pop", 0.4), (W(4, "glutamate."), "click", 0.4), (D() - 0.4, "whoosh", 0.35)],
     ["GOLDIE", "NA", "SHAKER"])

# ================================================================ s04 everywhere
scene("s04_everywhere")
shot(0.0, B(1, "but"), "MEDIUM",
     "'Alarmingly chemical': the shaker sprouts bubbling lab flasks; Goldie looks nervous; a tomato-red warning card flaps.",
     "Flasks", "Shaker + nervous Goldie", "Warning card",
     "Small shake on 'chemical'.",
     [("Chemical?", W(1, "chemical."), None)], SLIDE_IN, "Zoom-out",
     [(W(1, "alarmingly"), "bubble", 0.35), (W(1, "chemical."), "error", 0.3)],
     ["SHAKER", "GOLDIE"])
shot(B(1, "but"), L(2), "MEDIUM -> WIDE",
     "Flasks fall away; zoom out: Goldie is one bead in a long chain of vivid amino-acid beads that wraps around a "
     "paper Earth.",
     "Bead chain", "Paper Earth wrapped in chains", "—",
     "Zoom-out.",
     [("Amino acid", W(1, "amino"), None)], "Continuous", "Cut",
     [(W(1, "but"), "swoosh_up", 0.4), (W(1, "amino"), "pop_small", 0.35), (W(1, "earth."), "chime", 0.35)],
     ["GOLDIE"])
shot(L(2), L(3), "MEDIUM (2x2 grid)",
     "A 2x2 grid fills cell by cell, each food with a mini Goldie peeking out: tomato, mushrooms, parmesan, a "
     "simple milk-drop / baby-bottle icon.",
     "Tomato | mushrooms", "Parmesan | milk drop", "—",
     "Locked, boil only.",
     [], "Cut", "Split-wipe",
     [(W(2, "tomatoes,"), "pop", 0.4), (W(2, "mushrooms,"), "pop", 0.4), (W(2, "parmesan,"), "pop", 0.4), (W(2, "milk."), "pop", 0.4)],
     ["TOMATO", "GOLDIE"], notes="Icon-level milk; no nudity. No amounts, no ranking.")
shot(L(3), B(3, "are"), "WIDE (top/bottom split)",
     "Top half: a tomato releases a Goldie. Bottom half: the MSG shaker shakes out a Goldie. Each stands beside the "
     "same skeletal diagram.",
     "Tomato -> Goldie (top half)", "Divider", "Shaker -> Goldie (bottom half)",
     "Locked split.",
     [], "Split-wipe", "Continuous",
     [(W(3, "and"), "whoosh", 0.35), (W(3, "tomato"), "pop", 0.4), (W(3, "msg"), "pop", 0.4)],
     ["TOMATO", "SHAKER", "GOLDIE", "GLU_DIAGRAM"])
shot(B(3, "are"), D(), "MEDIUM",
     "Divider tears away; the two Goldies and their diagrams slide together and overlap perfectly; an '=' cut-out.",
     "'=' cut-out", "Two identical Goldies merging", "Same molecule slip",
     "Slow push-in.",
     [("Same molecule", W(3, "same"), None)], "Continuous", RIPPLE_OUT,
     [(W(3, "exactly"), "impact_soft", 0.55), (W(3, "same"), "success", 0.4)],
     ["GOLDIE", "GLU_DIAGRAM"], hero="Tomato glutamate = MSG glutamate (copies identical)")

# ================================================================ s05 body (style frame B)
scene("s05_body")
shot(0.0, L(2), "MEDIUM",
     "A fork lifts a bite sprinkled with tiny Goldies up to a paper-doll mouth in profile.",
     "Mouth (profile)", "Fork rising", "Plate",
     "Tilt up with the fork.",
     [], "Gold Ripple reveal", "Match-cut into the tongue",
     [(W(1, "happens"), "whoosh", 0.35), (W(1, "eat"), "chomp", 0.45)],
     ["GOLDIE"])
shot(L(2), B(2, "then"), "CLOSE -> MEDIUM",
     "Tongue landscape: Goldies land on receptor cups, each dock sends a small gold Fifth Ripple; a signal zips up a "
     "brush-stroke nerve to a pink paper brain that pops a speech slip.",
     "Brain + speech slip", "Nerve", "Tongue receptors",
     "Tilt up following the signal.",
     [("Protein!", W(2, "protein!"), None)], "Match-cut", "Dive",
     [(W(2, "tongue"), "click", 0.4), (W(2, "tastes"), "click", 0.35), (W(2, "tells"), "swoosh_up", 0.35), (W(2, "protein!"), "ding", 0.45)],
     ["BODY", "GOLDIE", "RIPPLE"])
shot(B(2, "then"), L(3), "WIDE (vertical dive)",
     "The camera dives down through layered pink paper into the gut; on 'eaten' the first gut cell chomps a Goldie.",
     "Pink paper layers rushing up", "Gut opening", "First gut cell",
     "Fast downward dive, ease-out.",
     [], "Dive", "Continuous",
     [(W(2, "then"), "whoosh_long", 0.45), (W(2, "eaten."), "chomp", 0.45)],
     ["BODY", "GUTCELL", "GOLDIE"])
shot(L(3), B(3, "around"), "MEDIUM (style frame B)",
     "Style frame B: turquoise sky over pink villi; gut cells with white bibs on two tiers; a stream of Goldies "
     "drifts in; the front cell chomps one with happy arc eyes, the next one wears a gold flame (burning fuel).",
     "'Fuel!' slip (taped)", "Back tier of 3 cells + Goldie stream", "Front eater + burning cell",
     "Slow lateral truck.",
     [("Fuel!", W(3, "fuel."), None)], "Continuous", "Pull-back",
     [(W(3, "cells"), "chomp", 0.4), (W(3, "intestines"), "chomp", 0.35), (W(3, "burn"), "sizzle", 0.3), (W(3, "fuel."), "pop_small", 0.35)],
     ["GUTCELL", "GOLDIE", "CAPTION"], hero="Gut cells eat glutamate as fuel (style frame B)")
shot(B(3, "around"), L(4), "MEDIUM",
     "Pull back: the gut wall as a pink border; a big Young Serif number rolls up and lands on '95%'; below the wall a "
     "coral blood vessel gets only a trickle of one or two Goldies on 'blood'.",
     "95% big number", "Gut wall eating the stream", "Blood vessel with a trickle",
     "Pull-back, then lock.",
     [("95%", W(3, "ninety-five"), None)], "Pull-back", "Track up the vessel",
     [(W(3, "around"), "tick", 0.2), (W(3, "around") + 0.15, "tick", 0.2), (W(3, "ninety-five"), "ding", 0.5), (W(3, "blood."), "bubble", 0.3)],
     ["GUTCELL", "GOLDIE", "BADGE95"], hero="95% lands on 'ninety-five'", notes="Never show '5%'.")
shot(L(4), B(4, "one"), "WIDE (vertical travel)",
     "Track up the vessel with the few Goldies to the head; a cobalt cut-paper castle wall rises across the frame.",
     "Castle wall", "Goldies rising", "Vessel",
     "Track upward.",
     [("Blood-brain barrier", W(4, "blood-brain"), None)], "Track", "Continuous",
     [(L(4), "whoosh", 0.4), (W(4, "blood-brain"), "thud", 0.45)],
     ["BBB", "GOLDIE"])
shot(B(4, "one"), D(), "MEDIUM",
     "Paper-doll guards (mitten hands, friendly) at the gate; a Goldie approaches, a guard raises a mitten, Goldie "
     "bounces back and shrugs.",
     "Gate", "Guard + Goldie", "Bouncing Goldie",
     "Push-in to the gate.",
     [], "Continuous", SLIDE_OUT,
     [(W(4, "best-guarded"), "click", 0.35), (W(4, "body."), "boing", 0.4), (D() - 0.4, "whoosh", 0.3)],
     ["BBB", "GOLDIE"])

# ================================================================ s07 tests
scene("s07_tests")
shot(0.0, B(1, "in"), "WIDE",
     "Bright lab on a cobalt graph-paper grid; scientists in all three skin tones run tests; clipboards flip with "
     "ticks on each 'again'.",
     "Graph-paper wall", "Scientists at the bench", "Clipboards flipping",
     "Slow lateral truck.",
     [], SLIDE_IN, "Cut",
     [(W(1, "tested"), "pop", 0.35), (W(1, "again"), "click", 0.35), (W(1, "again,"), "click", 0.35)],
     ["LAB"], notes="No study count on screen.")
shot(B(1, "in"), L(2), "MEDIUM",
     "Two identical capsules drop under two cups and shuffle; scientist and volunteer both wear paper blindfolds "
     "(nobody knows).",
     "Scientist (blindfolded)", "Shuffling cups", "Volunteer (blindfolded)",
     "Locked.",
     [("Double-blind", W(1, "double-blind"), None)], "Cut", "Cut",
     [(W(1, "studies,"), "whoosh", 0.3), (W(1, "where"), "whoosh", 0.3), (W(1, "nobody"), "pop_small", 0.3), (W(1, "thing."), "click", 0.35)],
     ["CAPSULES"], hero="Double-blind shuffle")
shot(L(2), B(2, "reacted"), "WIDE",
     "Two tiers of volunteers (all three skin tones) with worried thought bubbles take their cups.",
     "Thought bubbles", "Volunteer tier 1", "Volunteer tier 2",
     "Slow tilt down.",
     [], "Cut", "Continuous",
     [(W(2, "people"), "bubble", 0.35), (W(2, "sensitive"), "bubble", 0.3)],
     ["VOLUNTEERS"])
shot(B(2, "reacted"), L(3), "MEDIUM (tall chart)",
     "A flattened Fifth Ripple becomes the chart baseline; a gold bar (Goldie sticker = MSG) and a cobalt bar "
     "(placebo) rise to nearly the same height.",
     "Bar tops", "Gold bar | cobalt bar", "Ripple baseline",
     "Locked.",
     [("Placebo", W(2, "often"), None)], "Continuous", "Cut",
     [(W(2, "reacted"), "swoosh_up", 0.4), (W(2, "fake."), "ding", 0.35)],
     ["RIPPLE", "GOLDIE"], hero="MSG vs placebo nearly equal", notes="No axis numbers; label only the placebo bar (gold = MSG by colour).")
shot(L(3), B(3, "just"), "MEDIUM",
     "A generic official document slides in (no seal, no logo); the distressed rubber stamp slams on.",
     "Stamp handle lifting", "Document + GRAS stamp", "—",
     "Push-in; 4-frame shake on the stamp.",
     [("GRAS", W(3, "generally"), None)], "Cut", "Pull-back",
     [(W(3, "today,"), "paper", 0.35), (W(3, "generally"), "stamp", 0.8), (W(3, "generally") + 0.05, "impact_soft", 0.4)],
     ["STAMP"], hero="GRAS stamp")
shot(B(3, "just"), D(), "MEDIUM",
     "The MSG shaker hops onto a shelf next to a salt shaker and a sugar bowl.",
     "Shelf", "Salt | MSG | sugar", "—",
     "Pull-back, hold.",
     [], "Pull-back", SLIDE_OUT,
     [(W(3, "salt"), "pop", 0.4), (W(3, "sugar."), "pop", 0.4)],
     ["SHAKER", "SALT"])

# ================================================================ s08 twist
scene("s08_twist")
shot(0.0, B(1, "swap"), "MEDIUM",
     "The page curls (plot twist) to reveal a balance scale: salt shaker and MSG shaker level. Cobalt sodium ions pop "
     "out: three over salt, one over MSG.",
     "Sodium ions: 3 | 1", "Level scale with both shakers", "—",
     "Locked, then push-in on the ions.",
     [("Plot twist", W(1, "plot"), W(1, "has")), ("1/3 sodium", W(1, "third"), None)], "Page curl", "Cut",
     [(W(1, "plot"), "swoosh_up", 0.45), (W(1, "msg"), "thud", 0.4), (W(1, "third") - 0.3, "pop_small", 0.3),
      (W(1, "third") - 0.15, "pop_small", 0.3), (W(1, "third"), "pop_small", 0.3), (W(1, "sodium"), "ding", 0.45)],
     ["SALT", "SHAKER", "NA"], hero="About a third of the sodium (ions exactly 3:1)")
shot(B(1, "swap"), D(), "MEDIUM",
     "A cooking pot: part of a salt spoon is swapped for an MSG spoon; Goldie bobs happily in the broth (flavour "
     "kept); a sodium meter beside the pot drops.",
     "Spoons swapping", "Pot with happy Goldie", "Sodium meter dropping",
     "Locked.",
     [("Less sodium", W(1, "less"), None)], "Cut", "Paper slide to the cobalt desk",
     [(W(1, "swap"), "swoosh_down", 0.4), (W(1, "in,"), "bubbling_pot", 0.25), (W(1, "flavor"), "pop", 0.35), (W(1, "sodium.", 2), "success", 0.4)],
     ["SALT", "SHAKER", "NA", "GOLDIE"], notes="No % reduction on screen.")

# ================================================================ s06 origin (style frame C)
scene("s06_origin")
shot(0.0, L(2), "MEDIUM",
     "A kombu strip curls into the MSG shaker; a tomato-red spotlight snaps on and a cut-paper villain moustache drops onto it.",
     "Spotlight", "Shaker with moustache", "—",
     "Push-in.",
     [("Villain?", W(1, "villain?"), None)], SLIDE_IN, "Paper slide",
     [(W(1, "seaweed"), "whoosh", 0.35), (W(1, "villain?"), "impact_soft", 0.45)],
     ["SHAKER"])
shot(L(2), B(2, "after"), "MEDIUM (style frame C desk)",
     "Cobalt desk: a lemon letter on a hot-pink underlay slides in and writes itself in squiggle lines (no legible "
     "text); it folds into an envelope addressed in squiggles on 'journal'.",
     "Date slip", "Lemon letter writing itself", "Envelope",
     "Overhead, slow push.",
     [("1968", W(2, "1968,"), None)], "Paper slide", "Continuous",
     [(W(2, "1968,"), "pop", 0.4), (W(2, "wrote"), "typewriter", 0.35), (W(2, "journal."), "paper", 0.4)],
     ["LETTER"], notes="Author not named or drawn; journal not named.")
shot(B(2, "after"), L(3), "MEDIUM",
     "Two small cut-paper symptom icons pin onto the letter: a mitten hand with pins-and-needles dots (numb) on "
     "'numb', a drooping arm (weak) on 'weak'.",
     "—", "Letter + symptom icons", "—",
     "Locked.",
     [], "Continuous", "Continuous",
     [(W(2, "numb"), "pop_small", 0.35), (W(2, "weak."), "pop_small", 0.35)],
     ["LETTER"], notes="SENSITIVE: no restaurant, no people, no 'oriental' signifiers - paper and objects only.")
shot(L(3), B(3, "but"), "MEDIUM",
     "The turquoise card from style frame C: wine bottle, salt cellar, MSG shaker; an ink '?' drops over each on its "
     "word, then the three merge into one wobbly '?'.",
     "Wobbly '?'", "Turquoise card: bottle | cellar | shaker", "—",
     "Locked.",
     [("A guess", W(3, "guess."), None)], "Continuous", "Whip to the stamp",
     [(W(3, "wine?"), "blip", 0.4), (W(3, "salt?"), "blip", 0.4), (W(3, "msg?"), "blip", 0.45), (W(3, "guess."), "boing", 0.35)],
     ["LETTER", "SHAKER", "SALT"])
shot(B(3, "but"), L(4), "MEDIUM (style frame C)",
     "Style frame C: the stamp handle slams the quoted headline across the letter; paper flecks fly; coral "
     "newspapers start sliding out from the stamp like a tomato-red anti-ripple.",
     "Stamp handle lifting", "Stamped quoted headline", "Turquoise card + newspapers sliding",
     "Whip-in, 6-frame shake, hold.",
     [("\"Chinese-Restaurant Syndrome\"", W(3, "chinese"), None)], "Whip", "Continuous",
     [(W(3, "catchy"), "swoosh_down", 0.35), (W(3, "chinese"), "stamp", 0.8), (W(3, "chinese") + 0.05, "impact_big", 0.45),
      (W(3, "syndrome."), "paper", 0.35)],
     ["HEADLINE", "STAMP", "LETTER", "RIPPLE"], hero="Headline stamp (style frame C)",
     notes="Young Serif, hyphenated, in quotes; framed as the thing being criticised.")
shot(L(4), B(4, "scary,"), "MEDIUM",
     "A bright lab: pink newborn mouse pups in a nest; a comically giant syringe lowers from the top (never touches "
     "them). On 'brain damage' the coral newspapers sliding around the edges grow bigger tomato-red headline bars.",
     "Giant syringe", "Mouse pups in nest", "Newspapers at the edges",
     "Slow push-in.",
     [("Huge doses", W(4, "huge"), None)], "Cut", "Split-wipe",
     [(W(4, "injected"), "swoosh_down", 0.4), (W(4, "huge"), "impact_soft", 0.4), (W(4, "damage."), "alarm", 0.25)],
     ["MOUSE", "SYRINGE"], notes="No gore, no needle contact.")
shot(B(4, "scary,"), L(5), "WIDE (top/bottom split)",
     "Top: giant syringe over a tiny mouse pup. Bottom: a dinner bowl with a spoon and Goldies, and a tiny gut cell "
     "with a bib chomping one (callback to s05). A big '≠' cut-out on the divider.",
     "Syringe + pup (top half)", "'≠' on the divider", "Dinner bowl + gut-cell cameo (bottom half)",
     "Locked split.",
     [], "Split-wipe", "Split-wipe",
     [(W(4, "scary,"), "whoosh", 0.35), (W(4, "nothing"), "impact_soft", 0.5), (W(4, "dinner."), "chomp", 0.35)],
     ["MOUSE", "SYRINGE", "GOLDIE", "GUTCELL"], hero="Injection is not dinner")
shot(L(5), B(5, "the"), "WIDE (top/bottom split)",
     "Top: pasta with tomato sauce and parmesan under a silent tomato-red bell. Bottom: a plain white takeout carton "
     "under the same bell, ringing. Identical Goldies float over both.",
     "Pasta + silent bell (top half)", "Divider", "Carton + ringing bell (bottom half)",
     "Locked split.",
     [], "Split-wipe", "Collapse",
     [(W(5, "tomato"), "pop_small", 0.3), (W(5, "parmesan."), "alarm", 0.3)],
     ["TOMATO", "TAKEOUT", "GOLDIE"], hero="Double standard split")
shot(B(5, "the"), D(), "CLOSE",
     "The split collapses; the stamped headline slides back in; push in on it on 'name'.",
     "—", "Stamped quoted headline", "—",
     "Slow push-in.",
     [("\"Chinese-Restaurant Syndrome\"", W(5, "the"), None)], "Collapse", SLIDE_OUT + " back to the TV wall",
     [(W(5, "fear"), "swoosh_down", 0.3), (W(5, "name"), "impact_soft", 0.4)],
     ["HEADLINE", "STAMP"])

# ================================================================ s09 outro (LOOP)
scene("s09_outro")
shot(0.0, L(2), "WIDE",
     "TV wall callback: the three puppets still mouth their claims; the shaker sits in front with a '?'; on 'Not "
     "really' the '?' flips and Goldie pops out of the shaker smiling.",
     "TV wall", "Puppets on their screens", "Shaker + Goldie",
     "Slow push-in.",
     [("Not really", W(1, "not"), None)], SLIDE_IN, "Continuous",
     [(W(1, "msg"), "pop", 0.35), (W(1, "not"), "boing", 0.35), (W(1, "really."), "sparkle", 0.35)],
     ["TVWALL", "PUPPETS", "DRAMA_LT", "SHAKER", "GOLDIE"], drama=True)
shot(L(2), B(2, "one"), "WIDE",
     "The TV wall loops the claims; camera eases back so the wall fills MID.",
     "Upper TVs", "TV wall", "Shaker",
     "Slow pull-back.",
     [], "Continuous", "Continuous",
     [(W(2, "story"), "static", 0.2), (W(2, "about"), "channel_switch", 0.3)],
     ["TVWALL", "PUPPETS", "DRAMA_LT", "SHAKER"], drama=True)
shot(B(2, "one"), B(2, "can"), "MEDIUM",
     "Three objects drop onto the shaker like weights, each on its phrase: an ink '?' (guess), a mini tomato-red stamp "
     "impression (catchy name), a tomato-red tag on a string (prejudice).",
     "Falling object", "Shaker under the weights", "—",
     "Locked.",
     [], "Continuous", "Pull-back",
     [(W(2, "guess,"), "thud", 0.35), (W(2, "catchy"), "stamp", 0.35), (W(2, "prejudice"), "thud", 0.4)],
     ["SHAKER", "STAMP"], notes="Pictures only; the caption carries the words.")
shot(B(2, "can"), L(3), "WIDE",
     "TVs switch off one by one while a tall stack of vivid research papers rises beside them; the weights slide off.",
     "Paper stack top", "TV wall switching off", "Stack base",
     "Pull-back.",
     [], "Pull-back", "Paper slide",
     [(W(2, "outlive"), "tv_off", 0.45), (W(2, "decades"), "tv_off", 0.45), (W(2, "of", 2), "tv_off", 0.45), (W(2, "science."), "paper", 0.4)],
     ["TVWALL", "PUPPETS", "DRAMA_LT"], drama=True, hero="TVs switch off")
shot(L(3), B(3, "maybe"), "MEDIUM",
     "The NO MSG sign from s01 in its window; the 'NO' tube flickers and goes out on 'sign...', leaving 'MSG' lit gold.",
     "Window top", "Sign", "Window sill",
     "Slow push-in.",
     [("No MSG", L(3), W(3, "sign...")), ("MSG", W(3, "sign..."), None)], "Paper slide", "Dissolve through the window",
     [(W(3, "no"), "neon_flicker", 0.45), (W(3, "sign..."), "click", 0.45), (W(3, "sign...") + 0.3, "sparkle", 0.35)],
     ["SIGN"], hero="'NO' burns out")
shot(B(3, "maybe"), D(), "CLOSE -> WIDE (LOOP)",
     "Lamp-lit kitchen table: ramen bowl; a mitten hand reaches for the MSG shaker and shakes; one gold drop hits the "
     "broth and makes the Fifth Ripple (end card: 'MSG' slip on the ripple). Then pull back: the bowl is on the hero "
     "TV's lemon screen in the TV wall, framed exactly like s01 frame 0; a channel flicker hands over to the TV Doctor.",
     "TV wall (end)", "Ramen ripple -> hero TV", "'MSG' end-card slip",
     "Push-in, then pull-back to the s01 framing.",
     [("MSG", W(3, "shaker."), None)], "Dissolve", "LOOP: last frame = s01 frame 0 layout; flicker on the final frames",
     [(W(3, "reach"), "pop_small", 0.35), (W(3, "shaker."), "pop_small", 0.35), (W(3, "shaker.") + 0.4, "bubble", 0.4),
      (W(3, "shaker.") + 0.6, "chime", 0.4), (D() - 0.15, "channel_switch", 0.45)],
     ["SHAKER", "GOLDIE", "RIPPLE", "TVWALL", "END_CARD"], hero="End card -> loop into the hook",
     notes="Hero TV size/position and wall layout identical to s01.01 frame 0. No fade to black.")

# ================================================================ continuity, guardrails
CONTINUITY = {
    "GOLDIE": ("Goldie, the broth droplet", "KP.goldie: gold #FFB000 droplet, dot eyes, pink cheeks, steam curl. Gold is "
               "reserved for glutamate. In streams, many small Goldies. Friendly, never menacing; s04 copies identical."),
    "GLU_DIAGRAM": ("Glutamic-acid skeletal diagram", "Flat cut-paper five-carbon zigzag beside Goldie when chemistry "
                    "matters (s03, s04), so the droplet is never read as the literal molecule."),
    "SHAKER": ("MSG shaker", "One prop. Identity without lettering: gold crystals + Goldie sticker; text only when it is "
               "the shot's one label."),
    "SALT": ("Salt shaker", "Same silhouette, white crystals and ink-dot cap; no lettering."),
    "TVWALL": ("TV wall (style frame A)", "Violet wall, grape dim sets, turquoise hero TV spanning two columns with a "
               "lemon screen, layout exactly as frame A. s01.01 frame 0 and the last frame of s09 use the identical layout."),
    "PUPPETS": ("Shadow-puppet talking heads", "Ink silhouettes on orange sticks, identity from props only: TV Doctor "
                "(head mirror, stethoscope; KP.puppetDoctor), Celebrity Chef (toque, ladle), Wellness Influencer "
                "(ring-light halo, topknot, phone). Fictional; no likeness to real people. Host and Aunt are cut."),
    "DRAMA_LT": ("Torn-paper lower third", "Blue torn slip: role (the shot's one label) + 'DRAMATIZATION - fictional "
                 "character' on every frame where a puppet is readable."),
    "SIGN": ("NO MSG sign", "Paper 'neon': separate NO and MSG tubes, flat colour, no glow; same window and angle in s01 and s09."),
    "GUTCELL": ("Gut cells with bibs", "KP.cell: pink cells with white bibs, two tiers (3 behind, eater + burning cell in "
                "front), chomp and gold-flame states. Canonical in s05.04; cameo in s06."),
    "BADGE95": ("95% number", "Young Serif big number; only percentage in the film."),
    "BODY": ("Body interior", "Layered pink paper tints; tongue, nerve, brain and vessel as cut paper."),
    "BBB": ("Blood-brain barrier", "Cobalt cut-paper castle wall with friendly paper-doll guards."),
    "NA": ("Sodium ion", "Small cobalt 'Na'-free paper dot (no lettering), same in s03 and s08; s08 count exactly 3:1."),
    "TOMATO": ("Tomato", "Same ripe tomato paper cut in s04 and s06."),
    "RIPPLE": ("The Fifth Ripple", "KP.ripple: drop + 5 rings, the fifth gold and wider. Used in s02 (title), s03 "
               "(taste rings), s05 (receptor docks), s06 (newspaper anti-ripple), s07 (chart baseline), s09 (ramen, end card)."),
    "HEADLINE": ("Stamped headline", "KP.stamp, Young Serif, '\"CHINESE-RESTAURANT SYNDROME\"' over two lines, tomato red, distressed."),
    "STAMP": ("Rubber stamp", "Ink handle with tomato-red pad; same prop for the headline, GRAS and the s09 mini stamp."),
    "LETTER": ("1968 letter", "Lemon letter on hot-pink underlay, squiggle lines only (KP.squiggle), no legible words."),
    "TASTE_ICONS": ("Four taste icons", "Candy, lemon, salt crystal, coffee bean; riding the first four rings."),
    "IKEDA": ("Kikunae Ikeda", "Respectful paper doll, high collar, round glasses, moustache; never comedic."),
    "CAPTION": ("Caption strip", "KP.caption: torn white strip, Nunito Black 64 px, spoken words ink, upcoming lavender, "
                "current word on a raised chip (pink cold open, gold on glutamate, tomato red in s06)."),
    "END_CARD": ("End card", "'MSG' slip on the ramen Fifth Ripple, then the pull-back into the TV wall that loops to s01."),
}
INCIDENTAL = {"TAKEOUT", "MOUSE", "SYRINGE", "LAB", "CAPSULES", "VOLUNTEERS"}

GUARDRAILS = {
    "allowed_numbers_and_claims": [
        "s01: the three puppets' claims (fictional, DRAMATIZATION); 'more than fifty years' -> label '50+ years'",
        "s03: 1908; Japanese chemist Kikunae Ikeda; seaweed soup; not sweet/sour/salty/bitter; savory; boiled down seaweed; glutamic acid; named umami; paired with sodium -> monosodium glutamate",
        "s04: glutamate is one of the most common amino acids on Earth; in tomatoes, mushrooms, parmesan, human breast milk; tomato glutamate = MSG glutamate",
        "s05: tongue tastes it -> brain: protein; gut cells burn it as fuel; ~95% never reaches the blood; blood-brain barrier is one of the best-guarded borders",
        "s07: double-blind studies; MSG-sensitive people reacted as often to placebo; FDA: generally recognized as safe (GRAS), like salt and sugar",
        "s08: ~1/3 the sodium of table salt; swapping some in keeps flavour with less sodium",
        "s06: 1968 letter from a doctor to a medical journal; felt numb and weak after eating at Chinese restaurants; wine/salt/MSG was a guess; the journal's headline 'Chinese Restaurant Syndrome'; scientists injected huge doses into newborn mice and found brain damage; a mouse injection is nothing like dinner; nobody panicked about tomato sauce or parmesan",
        "s09: 'Not really'; one guess, a catchy name and a little prejudice outlived decades of science",
    ],
    "forbidden": [
        "Anything the rewrite cut: Tokyo label, 'within a year', soy sauce, '~10x' milk, 'body makes its own', palpitations, the journal's name, 1969, 'Italian Restaurant Syndrome', baking powder, 'even water', 'tomato on your pizza', huge doses on an empty stomach.",
        "Any number, %, axis value, study count, dose or date not listed above; never '5%'.",
        "Health claims beyond the script; real logos, seals, mastheads, networks, platforms or brands; recognisable real people.",
        "Gore: no needle touching an animal.",
        "Dull colours (grey, beige, cream, muddy brown, desaturated); glows and gradients (Kombu Paper is flat paper).",
        "More than one 1-3 word label at a time (the DRAMATIZATION line in the lower third is the only extra text).",
    ],
    "dramatization": "Every frame with a readable puppet (s01.01-04 and s01.07, s09.01, s09.02, s09.04) carries the torn-paper lower third with the role and 'DRAMATIZATION - fictional character'.",
    "sensitive_1960s_section": [
        "s06 shows only paper, type, a bottle, a salt cellar, a shaker, mice, food and a plain white carton: no restaurants, no staff, no diners.",
        "No faux-Asian fonts or signifiers (pagodas, lanterns, dragons, chopsticks as decoration, conical hats); the carton is plain white.",
        "The headline is quoted, hyphenated and stamped as the thing being criticised; never a joke or a brand.",
        "The prejudice beat is a bell over one food and silence over another; no people are shown being othered.",
        "Ikeda is drawn respectfully; the letter author and the mouse researchers are never named or drawn.",
    ],
}

ORDER = ["s01_coldopen", "s05_body", "s06_origin", "s02_title", "s03_ikeda", "s04_everywhere", "s07_tests", "s08_twist", "s09_outro"]
ORDER_WHY = {
    "s01_coldopen": "Closest to style frame A; builds TV wall, puppets, lower third, sign, shaker, caption usage. The hook.",
    "s05_body": "Closest to style frame B; builds Goldie streams, gut cells, body layers, 95%, BBB.",
    "s06_origin": "Closest to style frame C; builds letter, stamp, headline, newspapers; the sensitive scene, reviewed early.",
    "s02_title": "Short; establishes Goldie + the Fifth Ripple as the title beat.",
    "s03_ikeda": "Ikeda, taste rings, skeletal diagram, sodium ion, shaker birth.",
    "s04_everywhere": "Reuses Goldie, diagram, shaker; food grid.",
    "s07_tests": "Lab, capsules, chart on the ripple baseline, GRAS with the existing stamp.",
    "s08_twist": "Reuses shakers and sodium ions.",
    "s09_outro": "Last: all callbacks exist by now, and its final frame must match s01 frame 0 for the loop.",
}

# ================================================================ build + validate
errors = []
reel = script.PARTS[0]
assert len(script.PARTS) == 1, "expected a single Reel in script.PARTS"
offset, scenes_out, all_shots = 0.0, [], {}
for sid in reel["scenes"]:
    meta, shots, Dd = CUES[sid], SCENES[sid], CUES[sid]["duration"]
    if shots[0]["start"] != 0: errors.append(f"{sid}: first shot not at 0")
    if abs(shots[-1]["end"] - Dd) > 1e-3: errors.append(f"{sid}: last shot ends {shots[-1]['end']} != {Dd}")
    for j, sh in enumerate(shots):
        sh["id"] = f"{sid[:3]}.{j + 1:02d}"
        if j and abs(shots[j - 1]["end"] - sh["start"]) > 1e-6: errors.append(f"{sh['id']}: gap/overlap")
        dur = sh["end"] - sh["start"]
        if (dur < 2.0 and sid != "s02_title") or dur > 6.5: errors.append(f"{sh['id']}: duration {dur:.2f}")
        for s in sh["sfx"]:
            if s["name"] not in ALLOWED: errors.append(f"{sh['id']}: bad sfx {s['name']}")
            if not (sh["start"] - 0.15 <= s["t"] <= sh["end"] + 0.05) or not (0 <= s["t"] <= Dd): errors.append(f"{sh['id']}: sfx {s['t']} outside")
            if not 0 <= s["gain"] <= 1: errors.append(f"{sh['id']}: gain")
        labs = sorted(sh["labels"], key=lambda l: l["in"])
        for k, lab in enumerate(labs):
            words = len(re.sub(r"[\"?!]", "", lab["text"]).replace("-", " ").split())
            if words > 3: errors.append(f"{sh['id']}: label '{lab['text']}' has {words} words")
            if not (sh["start"] - 0.05 <= lab["in"] < lab["out"] <= sh["end"] + 1e-6): errors.append(f"{sh['id']}: label timing {lab}")
            if k and lab["in"] < labs[k - 1]["out"] - 1e-6: errors.append(f"{sh['id']}: two labels overlap")
        blob = " ".join([sh["on_screen"], *sh["layout"].values(), sh["camera"]])
        if DULL.search(blob): errors.append(f"{sh['id']}: dull-colour word '{DULL.search(blob).group(0)}'")
        if "PUPPETS" in sh["props"] and not sh["dramatization_tag"]: errors.append(f"{sh['id']}: puppet without DRAMATIZATION")
        for pr in sh["props"]:
            if pr not in CONTINUITY and pr not in INCIDENTAL: errors.append(f"{sh['id']}: undefined prop {pr}")
        sh["lines"] = []
        for n, ln in enumerate(meta["lines"], 1):
            ov = min(sh["end"], ln["end"]) - max(sh["start"], ln["start"])
            if ov > 0.2:
                full = sh["start"] <= ln["start"] + 0.05 and sh["end"] >= ln["end"] - 0.05
                sh["lines"].append({"n": n, "voice": ln["voice"], "start": ln["start"], "end": ln["end"],
                                    "coverage": "full" if full else "partial", "text": ln["text"]})
        sh["reel_time"] = [round(offset + sh["start"], 3), round(offset + sh["end"], 3)]
        all_shots[sh["id"]] = sh
    scenes_out.append({"id": sid, "title": meta["title"], "duration": Dd, "reel_offset": round(offset, 3),
                       "build_order": ORDER.index(sid) + 1,
                       "shots": [{k: sh[k] for k in ["id", "start", "end", "reel_time", "lines", "shot_size", "on_screen",
                                                      "layout", "camera", "labels", "dramatization_tag", "transition_in",
                                                      "transition_out", "sfx", "props", "hero", "notes"]} for sh in shots]})
    offset += Dd
# global label budget across consecutive shots is per-shot; check hook + loop
first, last = all_shots["s01.01"], all_shots[f"{reel['scenes'][-1][:3]}.{len(SCENES[reel['scenes'][-1]]):02d}"]
if not first["labels"] or first["labels"][0]["in"] > 0.3: errors.append("hook: label must be up by 0.3 s")
if "TVWALL" not in last["props"] or "END_CARD" not in last["props"]: errors.append("loop: last shot must end on the TV wall end card")
if offset >= 180: errors.append(f"Reel is {offset:.1f}s (must be under 3 min)")
if errors:
    print("\n".join(errors)); sys.exit(1)

continuity = []
for cid, (name, rules) in CONTINUITY.items():
    used = [s["id"] for s in all_shots.values() if cid in s["props"]]
    continuity.append({"id": cid, "name": name, "rules": rules, "appears_in": used})
heroes = [{"shot": s["id"], "start": s["start"], "end": s["end"], "reel_time": s["reel_time"], "beat": s["hero"]}
          for s in all_shots.values() if s["hero"]]
out = {"meta": {"project": "MSG: The Most Misunderstood Molecule", "format": "single Instagram Reel, 9:16, loops",
                "duration": round(offset, 3), "fps": 30, "resolution": [1080, 1920], "frame": FRAME,
                "style": "Kombu Paper (design/APPROVED.md; style frames design/frames/kombu-paper/A,B,C; kit window.KP in frames.js)",
                "time_base": "seconds, scene-relative (the t passed to draw); reel_time = seconds from the start of the Reel",
                "label_rule": "at most one 1-3 word label at a time besides captions (+ the DRAMATIZATION line under puppets)",
                "source_of_truth": "cues.json (line + word timings), script.py (PARTS)",
                "generator": "design/gen_shotlist.py (edit there, not here)"},
       "build_order": [{"scene": s, "why": ORDER_WHY[s]} for s in ORDER],
       "scenes": scenes_out, "hero_moments": heroes, "continuity": continuity, "guardrails": GUARDRAILS}
json.dump(out, open(os.path.join(HERE, "shotlist.json"), "w"), indent=1, ensure_ascii=False)

# ================================================================ markdown
f2 = lambda x: f"{x:.2f}"  # noqa: E731
n_shots = len(all_shots)
md = [f"""# MSG — Shot List (one Instagram Reel, 9:16, Kombu Paper)

_Generated by `design/gen_shotlist.py` from `cues.json` + `script.py`. Edit the generator, then run
`python3 design/gen_shotlist.py`. Machine-readable twin: `design/shotlist.json`._

**Totals:** 9 scenes · {n_shots} shots · {len(heroes)} hero moments · {offset:.1f} s · loops (last frame = first frame layout)

**Build order (one animator):** """ + " → ".join(ORDER) + """

## Frame map (1080x1920) — applies to every shot

```
y    0 ┌──────────────────────────────┬────┐
       │ UI TOP — decor only          │    │
y  220 ├──────────────────────────────┤ R  │  TOP  220-520   the one label, secondary props
       │ TOP                          │ I  │
y  520 ├──────────────────────────────┤ G  │  MID  520-1100  main subject (centre x = 490, not 540)
       │ MID                          │ H  │
y 1100 ├──────────────────────────────┤ T  │  LOW  1100-1300 secondary subject, torn-paper lower third
       │ LOW                          │    │
y 1300 ├══════════════════════════════┤ 140│  CAPTION BAND 1300-1480: KP.caption strip only
       │ CAPTIONS                     │ px │
y 1480 ├──────────────────────────────┤    │
       │ UI BOTTOM — decor only       │    │
y 1920 └──────────────────────────────┴────┘
       x=40                       x=940  1080
```

- **Key content box: x 40-940, y 220-1300.** Decor (sideboard, gut layers, sliding newspapers, as in the style
  frames) may fill the UI zones, the right gutter and the area behind the caption band.
- **One-label rule:** besides captions, each moment shows at most one label of 1-3 words, printed on paper (torn
  slip, stamp, sign, lower third). Each shot lists its label(s) with in/out times; they never overlap. Puppet shots
  add only the required 'DRAMATIZATION - fictional character' line inside the same lower third. Props carry no
  lettering unless the lettering is that shot's label. Everything else is told with pictures.
- **Kombu Paper vocabulary:** cut paper with hard offset shadows and riso misregistration, boil on twos, no
  outlines, no glows, no gradients, vivid riso stock only, gold reserved for glutamate (Goldie). Use the `KP` kit
  (`design/frames/kombu-paper/frames.js`).
- **Transitions:** paper slide (default), Gold Ripple wipe on act changes (s02→s03, s04→s05), channel flicker on
  the TV wall. No dips to black anywhere: the Reel opens lit and loops.
- **Timing:** shot boundaries are line starts or `word start - 0.1 s`; labels and SFX land on real word starts.
  Times are scene-relative; `reel_time` in the JSON is the Reel clock.
- SFX: `time name gain`, BRIEF.md names only. ★ = hero moment.
"""]
for sc in scenes_out:
    md += [f"## {sc['id']} — {sc['title']}  ·  {sc['duration']:.3f} s  ·  {len(sc['shots'])} shots  ·  "
           f"Reel {sc['reel_offset']:.1f}–{sc['reel_offset'] + sc['duration']:.1f} s  ·  build #{sc['build_order']}", ""]
    for sh in sc["shots"]:
        hero = f"  ★ {sh['hero']}" if sh["hero"] else ""
        md += [f"### {sh['id']}  `{f2(sh['start'])} → {f2(sh['end'])}` ({sh['end'] - sh['start']:.1f} s) · {sh['shot_size']}{hero}", ""]
        for ln in sh["lines"]:
            md.append(f"> L{ln['n']} {ln['voice']} [{f2(ln['start'])}–{f2(ln['end'])}{', partial' if ln['coverage'] == 'partial' else ''}]: {ln['text']}  ")
        if not sh["lines"]:
            md.append("> (no narration)")
        lay = sh["layout"]
        lab = "; ".join(f"**{l['text']}** {f2(l['in'])}–{f2(l['out'])}" for l in sh["labels"]) or "none (pictures only)"
        if sh["dramatization_tag"]:
            lab += " + DRAMATIZATION line"
        md += ["",
               f"- **Action:** {sh['on_screen']}",
               f"- **TOP:** {lay['TOP']}  \n  **MID:** {lay['MID']}  \n  **LOW:** {lay['LOW']}",
               f"- **Label:** {lab}",
               f"- **Camera:** {sh['camera']}",
               f"- **Transition:** in: {sh['transition_in']} / out: {sh['transition_out']}",
               "- **SFX:** " + ", ".join(f"`{s['t']:.2f} {s['name']} {s['gain']}`" for s in sh["sfx"])]
        if sh["props"]: md.append(f"- **Continuity:** {', '.join(sh['props'])}")
        if sh["notes"]: md.append(f"- **Note:** {sh['notes']}")
        md.append("")
md += ["---", "", "## Hero moments", "", "| # | Shot | Scene time | Reel time | Beat |", "|---|---|---|---|---|"]
for i, h in enumerate(heroes, 1):
    md.append(f"| {i} | {h['shot']} | {f2(h['start'])}–{f2(h['end'])} | {f2(h['reel_time'][0])} | {h['beat']} |")
md += ["", "## Continuity bible (Kombu Paper)", "", "\"Appears in\" is computed from the shots above.", ""]
for c in continuity:
    md += [f"### {c['id']} — {c['name']}", f"- **Rules:** {c['rules']}",
           f"- **Appears in:** {', '.join(c['appears_in']) or '—'}", ""]
md += ["## Factual guardrails", "", "### Allowed (only what the new script says)", ""]
md += [f"- {x}" for x in GUARDRAILS["allowed_numbers_and_claims"]]
md += ["", "### Forbidden", ""] + [f"- {x}" for x in GUARDRAILS["forbidden"]]
md += ["", "### DRAMATIZATION rule", "", GUARDRAILS["dramatization"], "", "### 1960s section / cultural sensitivity", ""]
md += [f"- {x}" for x in GUARDRAILS["sensitive_1960s_section"]]
open(os.path.join(HERE, "SHOTLIST.md"), "w").write("\n".join(md) + "\n")

print({s["id"]: len(s["shots"]) for s in scenes_out}, n_shots, "shots,", len(heroes), "heroes,", f"{offset:.1f}s")
