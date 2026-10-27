#!/usr/bin/env python3
"""Shot-list generator for the MSG Reel (one 9:16 Instagram Reel, Kombu Paper style, viewer-test cut).

Single source for design/shotlist.json and design/SHOTLIST.md:
    python3 design/gen_shotlist.py
All times come from cues.json (line starts + per-word "words" timings) via L()/W()/B(). After a narration
re-record, re-run this script. It fails loudly if:
- a referenced word disappeared
- shots stop tiling
- a label breaks the one-label rule
- a shot holds still for more than 2 s (no sfx/label/beat)
- a real clip lacks its name + credit lower third
- a dull-colour word sneaks into a description
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
MAX_STILL = 2.0   # viewer-test rule: no shot holds without a visible action for longer than this

FRAME = {
    "size": [1080, 1920],
    "ui_top": [0, 220], "ui_bottom": [1480, 1920], "right_gutter_x": [940, 1080], "left_margin_x": [0, 40],
    "caption_band": [1300, 1480],
    "key_box": {"x": [40, 940], "y": [220, 1300], "center_x": 490},
    "edge_margin": 40,
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
         notes=None, credit=False, beats=(), flash=False):
    """labels: [(text, in, out|None)]; beats: [(t, secondary action)] that keep the frame alive between events."""
    SCENES.setdefault(CUR, []).append(dict(
        start=round(start, 3), end=round(end, 3), shot_size=size, on_screen=action,
        layout={"TOP": top, "MID": mid, "LOW": low}, camera=camera,
        labels=[{"text": a, "in": round(b, 3), "out": round(c if c is not None else end, 3)} for a, b, c in labels],
        credit_lower_third=credit, flash=flash, transition_in=t_in, transition_out=t_out,
        beats=[{"t": round(a, 3), "action": b} for a, b in beats],
        sfx=[dict(t=round(a, 3), name=b, gain=c) for a, b, c in sfx], props=list(props), hero=hero, notes=notes))


SLIDE_IN, SLIDE_OUT = "Paper slide in, already carrying this shot's art", "Paper slide carrying the next shot's art (never a blank sheet)"
RIPPLE_OUT = "Gold Ripple wipe (the ring reveals the next scene inside it)"
CREDIT = "torn-paper lower third: real name (the label) + credit line (outlet, date)"
CLIP_NOTE = ("Real interview clip (delivered): draw it with CLIPS.draw(ctx, id, tc, ...) inside the hero "
             "screen (lib/clips.js; tc = scene time - CLIPS.lineStart) and take name/credit from CLIPS.info(id); until "
             "its frames load, CLIPS.draw returns false: show a placeholder rectangle. The clip's own audio plays on its cue line.")

# ================================================================ s01 cold open (HOOK)
scene("s01_coldopen")
shot(0.0, L(2), "CLOSE (pushed in on the hero TV)",
     "HOOK, lit on frame 0: the camera is pushed in so the hero TV's screen fills most of the frame (screen about "
     "860 px wide across TOP+MID, paper TV bezel at the edges). It plays clip blaylock_hungry. In LOW, beside/below the "
     "screen, a paper MSG shaker tips and pours gold crystals onto a plate on 'MSG'; the plate grows on 'hungry' and "
     "swells again, piled high, on 'eat more'.",
     "Hero TV screen (top part)", "Clip blaylock_hungry, large", CREDIT + "; shaker + growing plate beside it",
     "Locked close, slight multiplane drift.",
     [("Russell Blaylock", 0.0, None)], "NONE: lit on frame 0 (hook; also the loop landing point)", "Jump-cut flicker (same speaker, next clip)",
     [(0.0, "tv_on", 0.45), (W(1, "msg"), "pop", 0.35), (W(1, "hungry"), "bubble", 0.35), (W(1, "eat"), "chomp", 0.35)],
     ["TVWALL", "CLIPS", "NAME_LT", "SHAKER", "CAPTION"], hero="Hook: big real clip + MSG pouring onto a growing plate",
     credit=True, beats=[(W(1, "food.", 2), "crystals keep pouring")],
     notes=CLIP_NOTE + " Credit: 'Neurosurgeon, lecture on MSG, 2012' (CLIPS.info). The clip must read big at a glance "
     "(client). Frame 0 must equal the last frame of s09 so the Reel loops.")
shot(L(2), L(3), "CLOSE (pushed in on the hero TV)",
     "Jump-cut flicker: same big framing, the screen plays clip blaylock_brain. Beside the screen, a pink paper brain "
     "cut-out whose tomato-red zap bolts burst on 'destroys' and crackle on 'brain'.",
     "Hero TV screen (top part)", "Clip blaylock_brain, large", CREDIT + "; brain cut-out beside it",
     "Locked close, slight drift.",
     [("Russell Blaylock", L(2), None)], "Jump-cut flicker", "Fast pull-back",
     [(L(2) - 0.05, "channel_switch", 0.45), (W(2, "excitotoxicity"), "static", 0.25), (W(2, "destroys"), "glitch", 0.4), (W(2, "brain"), "blip", 0.3)],
     ["TVWALL", "CLIPS", "NAME_LT"], credit=True, beats=[(W(2, "excitotoxicity") + 0.8, "brain cut-out trembles")],
     notes=CLIP_NOTE + " Cut-outs sit beside the screen, never over the speaker.")
shot(L(3), B(3, "of"), "CLOSE -> EXTREME WIDE",
     "Fast pull-back from the big screen: every set on the wall lights up (paused frames of the two clips plus lemon "
     "screens); the wall recedes (it may bleed into the UI zones as decor).",
     "Upper TV rows", "Full TV wall", "Lower TV rows",
     "Fast pull-back, multiplane layers separating.",
     [("50 years", W(3, "fifty"), None)], "Fast pull-back", "Paper slide carrying the restaurant window",
     [(L(3), "whoosh", 0.45), (W(3, "fifty"), "pop", 0.4), (W(3, "everyone"), "tv_on", 0.3)],
     ["TVWALL"], notes="Clip frames here are small and unreadable; no lower third needed. If a face reads, add its credit.")
shot(B(3, "of"), B(3, "one"), "MEDIUM",
     "Everyone scared of MSG: a restaurant window with the paper 'neon' NO MSG sign switching on, three small dinner "
     "tables stacked below with folded NO MSG cards. On 'it all started' a lemon envelope slides onto the top table.",
     "Restaurant window + NO MSG sign", "Table 1 / table 2", "Table 3 + envelope arriving",
     "Tilt down the stack.",
     [("No MSG", W(3, "of"), None)], "Paper slide carrying the restaurant window", "Push into the envelope",
     [(W(3, "of"), "neon_flicker", 0.45), (W(3, "msg."), "pop_small", 0.3), (W(3, "started"), "paper", 0.35)],
     ["SIGN", "LETTER"], beats=[(W(3, "and"), "envelope slides in")],
     notes="First appearance of the NO MSG sign (paid off in s09). Paper 'neon': flat colour tubes, no glow.")
shot(B(3, "one"), D(), "CLOSE",
     "HANDOFF TO s06: the sealed lemon envelope lifts off the table toward camera on 'one letter' and lands on the "
     "cobalt desk of s06 (hot-pink underlay, turquoise card at the edge). A '1968' date slip is pinned to it on '1968'. "
     "The last frame is s06.01's first frame (same desk, same envelope).",
     "Date slip", "Sealed envelope on the cobalt desk", "Turquoise card edge",
     "Push-in following the envelope, then lock.",
     [("1968", W(3, "1968."), None)], "Push into the envelope", "Match cut into s06.01 (same desk; no blank frame)",
     [(W(3, "one"), "whoosh", 0.4), (W(3, "letter,"), "paper", 0.4), (W(3, "1968."), "pop", 0.45)],
     ["LETTER"], beats=[(W(3, "1968.") + 1.0, "the envelope's flap lifts a little")])

# ================================================================ s03 Ikeda
scene("s03_ikeda")
shot(0.0, B(1, "tasted"), "MEDIUM",
     "Opens on the gold drop from s06.10 landing in a steaming soup bowl: a small Fifth Ripple spreads on the broth. "
     "Pull back: 1908, a bright period laboratory-kitchen; Kikunae Ikeda (respectful paper doll: high collar, round "
     "glasses, moustache) is carrying that bowl to his desk.",
     "Date slip", "Ikeda + bowl (ripple on the broth)", "Desk",
     "Start close on the bowl, pull back, lateral truck with Ikeda.",
     [("1908", W(1, "1908,"), None)], "Match: the gold drop from s06.10 lands in Ikeda's soup bowl (no blank frame)", "Continuous",
     [(0.05, "bubble", 0.45), (W(1, "1908,"), "pop", 0.4), (W(1, "chemist"), "paper", 0.3), (W(1, "ikeda"), "pop_small", 0.3)],
     ["IKEDA", "RIPPLE"], beats=[(W(1, "japanese"), "steam curls off the bowl")])
shot(B(1, "tasted"), L(2), "MEDIUM-CLOSE",
     "Ikeda sips. A long ruffled kombu blade (kelp ribbon with wavy edges and a holdfast) hangs over the bowl's rim, "
     "unmistakably seaweed; his eyebrows lift and a '?' pops on 'odd'.",
     "'?' cut-out", "Ikeda sipping", "Bowl + kombu blade",
     "Push-in.",
     [("Seaweed", W(1, "seaweed"), None)], "Continuous", "Push into the bowl",
     [(W(1, "tasted"), "slurp", 0.35), (W(1, "odd"), "pop_small", 0.35), (W(1, "seaweed"), "bubble", 0.3)],
     ["IKEDA", "KOMBU"], notes="Seaweed must read as seaweed at thumbnail size (check a 50% still).")
shot(L(2), B(2, "he"), "CLOSE (overhead on bowl)",
     "Fifth Ripple motif: rings spread from the bowl, each carrying a taste icon that pops on its word (candy, lemon, "
     "salt crystal, coffee bean).",
     "Sweet / sour icons", "Bowl + rings", "Salty / bitter icons",
     "Locked overhead.",
     [], "Push-in", "Continuous",
     [(W(2, "sweet,"), "pop", 0.35), (W(2, "sour,"), "pop", 0.35), (W(2, "salty"), "pop", 0.35), (W(2, "bitter."), "pop", 0.35)],
     ["RIPPLE", "TASTE_ICONS", "KOMBU"])
shot(B(2, "he"), L(3), "CLOSE -> MEDIUM",
     "The fifth ring blooms wide and gold; Goldie rises from the centre. On 'seasoning' gold crystals pour from the "
     "ring into the MSG shaker.",
     "Umami slip (Young Serif)", "Fifth gold ring + Goldie", "Shaker filling",
     "Slight push-in, then tilt down to the shaker.",
     [("Umami", W(2, "umami,"), None)], "Continuous", "Continuous",
     [(W(2, "umami,"), "impact_soft", 0.6), (W(2, "umami,") + 0.15, "chime", 0.4), (W(2, "turned"), "swoosh_down", 0.35), (W(2, "seasoning."), "pop", 0.35)],
     ["RIPPLE", "GOLDIE", "SHAKER"], hero="Umami is named")
shot(L(3), D(), "MEDIUM",
     "The shaker's cap snaps on; Goldie sticker presses onto it; the shaker does a little hop.",
     "—", "Shaker", "'MSG' slip",
     "Locked.",
     [("MSG", W(3, "monosodium"), None)], "Continuous", SLIDE_OUT,
     [(W(3, "monosodium"), "click", 0.4), (W(3, "glutamate."), "pop", 0.4), (D() - 0.4, "whoosh", 0.35)],
     ["GOLDIE", "SHAKER"])

# ================================================================ s04 everywhere
scene("s04_everywhere")
shot(0.0, round(W(1, "chemical.") + 0.9, 3), "MEDIUM",
     "'Alarmingly chemical': the shaker sprouts bubbling lab flasks; Goldie looks nervous; a tomato-red warning card flaps.",
     "Flasks", "Shaker + nervous Goldie", "Warning card",
     "Small shake on 'chemical'.",
     [("Chemical?", W(1, "chemical."), None)], SLIDE_IN, "Zoom-out",
     [(0.2, "bubble", 0.3), (W(1, "alarmingly"), "bubble", 0.35), (W(1, "chemical."), "error", 0.3)],
     ["SHAKER", "GOLDIE"])
shot(round(W(1, "chemical.") + 0.9, 3), B(2, "tomatoes,"), "WIDE",
     "Flasks fall away; zoom out to a paper Earth where little Goldies pop up all over its surface.",
     "Goldies popping", "Paper Earth", "—",
     "Zoom-out.",
     [], "Continuous", "Cut",
     [(W(2, "but"), "swoosh_up", 0.4), (W(2, "glutamate"), "pop_small", 0.3), (W(2, "everywhere:"), "sparkle", 0.4)],
     ["GOLDIE"])
shot(B(2, "tomatoes,"), L(3), "MEDIUM (3 stacked)",
     "Three foods pop in a vertical stack, each with a mini Goldie peeking out: tomato, parmesan wedge, a simple "
     "milk-drop / baby-bottle icon.",
     "Tomato", "Parmesan", "Milk drop",
     "Tilt down the stack.",
     [], "Cut", "Split-wipe",
     [(W(2, "tomatoes,"), "pop", 0.4), (W(2, "parmesan,"), "pop", 0.4), (W(2, "milk."), "pop", 0.4)],
     ["TOMATO", "GOLDIE"], notes="Icon-level milk; no nudity. No amounts.")
shot(L(3), B(3, "is"), "WIDE (top/bottom split)",
     "Top half: a tomato releases a Goldie. Bottom half: the MSG shaker shakes out a Goldie. Each stands beside the "
     "same skeletal diagram.",
     "Tomato -> Goldie (top half)", "Divider", "Shaker -> Goldie (bottom half)",
     "Locked split.",
     [], "Split-wipe", "Continuous",
     [(W(3, "to"), "whoosh", 0.35), (W(3, "tomato"), "pop", 0.4), (W(3, "msg"), "pop", 0.4)],
     ["TOMATO", "SHAKER", "GOLDIE", "GLU_DIAGRAM"], beats=[(W(3, "glutamate"), "diagrams slide in")])
shot(B(3, "is"), D(), "MEDIUM",
     "Divider tears away; the two Goldies and their diagrams slide together and overlap perfectly; an '=' cut-out.",
     "'=' cut-out", "Two identical Goldies merging", "Same molecule slip",
     "Slow push-in.",
     [("Same molecule", W(3, "same"), None)], "Continuous", RIPPLE_OUT,
     [(W(3, "exact"), "impact_soft", 0.55), (W(3, "same"), "success", 0.4)],
     ["GOLDIE", "GLU_DIAGRAM"], hero="Tomato glutamate = MSG glutamate (copies identical)",
     beats=[(W(3, "molecule.") + 0.6, "the merged Goldie winks")])

# ================================================================ s05 body (style frame B)
scene("s05_body")
shot(0.0, L(2), "MEDIUM -> WIDE (dive)",
     "A fork lifts a bite sprinkled with tiny Goldies to a paper-doll mouth; on the bite the camera dives straight "
     "down through layered pink paper into the gut (no tongue beat).",
     "Mouth (profile)", "Fork rising -> pink layers rushing up", "Plate",
     "Tilt up with the fork, then fast downward dive.",
     [], "Gold Ripple reveal", "Dive lands in the gut",
     [(W(1, "happens"), "whoosh", 0.35), (W(1, "eat"), "chomp", 0.45), (W(1, "it?"), "whoosh_long", 0.45)],
     ["GOLDIE", "BODY"])
shot(L(2), B(2, "around"), "MEDIUM (style frame B)",
     "Style frame B: turquoise sky over pink villi; gut cells with white bibs on two tiers; a stream of Goldies "
     "drifts in; the front cell chomps one with happy arc eyes, the next one wears a gold flame (burning fuel).",
     "'Fuel!' slip (taped)", "Back tier of 3 cells + Goldie stream", "Front eater + burning cell",
     "Slow lateral truck.",
     [("Fuel!", W(2, "fuel."), None)], "Dive lands", "Pull-back",
     [(W(2, "cells"), "chomp", 0.4), (W(2, "gut"), "chomp", 0.35), (W(2, "burn"), "sizzle", 0.3), (W(2, "fuel."), "pop_small", 0.35)],
     ["GUTCELL", "GOLDIE", "CAPTION"], hero="Gut cells eat glutamate as fuel (style frame B)")
shot(B(2, "around"), L(3), "MEDIUM",
     "Pull back: the gut wall as a pink border; a big Young Serif number rolls up and lands on '95%'; below the wall a "
     "coral blood vessel gets only a trickle of one or two Goldies on 'blood'.",
     "95% big number", "Gut wall eating the stream", "Blood vessel with a trickle",
     "Pull-back, then lock.",
     [("95%", W(2, "ninety-five"), None)], "Pull-back", "Track up the vessel",
     [(W(2, "around"), "tick", 0.2), (W(2, "around") + 0.15, "tick", 0.2), (W(2, "ninety-five"), "ding", 0.5), (W(2, "blood."), "bubble", 0.3)],
     ["GUTCELL", "GOLDIE", "BADGE95"], hero="95% lands on 'ninety-five'",
     beats=[(W(2, "never"), "a gut cell burps a tiny flame")], notes="Never show '5%'.")
shot(L(3), B(3, "of"), "WIDE (vertical travel)",
     "Track up the vessel with the few remaining Goldies to the head; a cobalt cut-paper castle wall rises across the frame.",
     "Castle wall", "Goldies rising", "Vessel",
     "Track upward.",
     [("Blood-brain barrier", W(3, "blood-brain"), None)], "Track", "Continuous",
     [(L(3), "whoosh", 0.4), (W(3, "blood-brain"), "thud", 0.45)],
     ["BBB", "GOLDIE"])
shot(B(3, "of"), D(), "MEDIUM",
     "Paper-doll guards (mitten hands, friendly) at the gate; a Goldie approaches, a guard raises a mitten, Goldie "
     "bounces back and shrugs.",
     "Gate", "Guard + Goldie", "Bouncing Goldie",
     "Push-in to the gate.",
     [], "Continuous", SLIDE_OUT,
     [(W(3, "best-guarded"), "click", 0.35), (W(3, "body."), "boing", 0.4), (D() - 0.4, "whoosh", 0.3)],
     ["BBB", "GOLDIE"])

# ================================================================ s07 tests
scene("s07_tests")
shot(0.0, B(1, "reacted"), "MEDIUM",
     "Cobalt graph-paper lab: two identical capsules drop under two cups and shuffle; scientist and volunteer both "
     "wear paper blindfolds. On 'MSG made them sick' a row of volunteers (all three skin tones) shows worried thought "
     "bubbles.",
     "Scientist (blindfolded)", "Shuffling cups", "Volunteers with thought bubbles",
     "Locked, then tilt down to the volunteers.",
     [("Double-blind", W(1, "double-blind"), None)], SLIDE_IN, "Continuous",
     [(W(1, "in"), "whoosh", 0.3), (W(1, "studies,"), "whoosh", 0.3), (W(1, "people"), "bubble", 0.35), (W(1, "sick"), "bubble", 0.3)],
     ["CAPSULES", "VOLUNTEERS"], hero="Double-blind shuffle", beats=[(W(1, "made"), "a volunteer clutches its tummy in a thought bubble")])
shot(B(1, "reacted"), L(2), "MEDIUM (tall chart)",
     "A flattened Fifth Ripple becomes the chart baseline; a gold bar (Goldie sticker = MSG) and a cobalt bar "
     "(placebo) rise to nearly the same height.",
     "Bar tops", "Gold bar | cobalt bar", "Ripple baseline",
     "Locked.",
     [("Placebo", W(1, "placebo."), None)], "Continuous", "Cut",
     [(W(1, "reacted"), "swoosh_up", 0.4), (W(1, "often"), "pop_small", 0.3), (W(1, "placebo."), "ding", 0.35)],
     ["RIPPLE", "GOLDIE"], hero="MSG vs placebo nearly equal", notes="No axis numbers; only the placebo bar is labelled.")
shot(L(2), D(), "MEDIUM",
     "A generic official document slides in (no seal, no logo); the rubber stamp slams GRAS on; then the MSG shaker "
     "hops onto a shelf beside a salt shaker.",
     "Stamp handle lifting", "Document + GRAS stamp", "Shelf: salt | MSG",
     "Push-in; 4-frame shake on the stamp; pull-back to the shelf.",
     [("GRAS", W(2, "generally"), None)], "Cut", SLIDE_OUT,
     [(W(2, "fda"), "paper", 0.35), (W(2, "generally"), "stamp", 0.8), (W(2, "generally") + 0.05, "impact_soft", 0.4),
      (W(2, "safe."), "sparkle", 0.3), (W(2, "salt."), "pop", 0.4)],
     ["STAMP", "SHAKER", "SALT"], hero="GRAS stamp")

# ================================================================ s08 twist
scene("s08_twist")
shot(0.0, B(1, "same"), "MEDIUM",
     "The page curls (plot twist) to reveal a balance scale: salt shaker and MSG shaker level. Cobalt sodium ions pop "
     "out: three over salt, one over MSG.",
     "Sodium ions: 3 | 1", "Level scale with both shakers", "—",
     "Locked, then push-in on the ions.",
     [("Plot twist", W(1, "plot"), W(1, "third") - 0.05), ("1/3 sodium", W(1, "third"), None)], "Page curl", "Cut",
     [(W(1, "plot"), "swoosh_up", 0.45), (W(1, "has"), "thud", 0.4), (W(1, "third") - 0.3, "pop_small", 0.3),
      (W(1, "third") - 0.15, "pop_small", 0.3), (W(1, "third"), "pop_small", 0.3), (W(1, "sodium"), "ding", 0.45)],
     ["SALT", "SHAKER", "NA"], hero="A third of the sodium (ions exactly 3:1)", beats=[(W(1, "table"), "scale wobbles level")])
shot(B(1, "same"), D(), "MEDIUM",
     "A cooking pot: Goldie bobs happily in the broth (same flavour); a sodium meter beside the pot drops.",
     "Steam", "Pot with happy Goldie", "Sodium meter dropping",
     "Locked.",
     [("Less sodium", W(1, "less"), None)], "Cut", "Paper slide carrying s09's TV wall",
     [(W(1, "same"), "bubbling_pot", 0.3), (W(1, "flavor,"), "pop", 0.35), (W(1, "less"), "swoosh_down", 0.35), (W(1, "sodium.", 2), "success", 0.4)],
     ["SHAKER", "NA", "GOLDIE"], notes="No % reduction on screen.")

# ================================================================ s06 origin (style frame C)
scene("s06_origin")
shot(0.0, L(2), "MEDIUM (cobalt desk)",
     "Same cobalt desk as the end of s01, the sealed envelope beside it: a kombu blade curls into the MSG shaker; a "
     "tomato-red spotlight snaps on and a cut-paper villain moustache drops onto it (viewer favourite: keep).",
     "Spotlight", "Shaker with moustache", "Envelope on the desk",
     "Push-in.",
     [("Villain?", W(1, "villain?"), None)], "Match cut from s01.05 (same desk)", "Pan to the envelope (continuous)",
     [(0.2, "whoosh", 0.3), (W(1, "seaweed"), "swoosh_down", 0.3), (W(1, "villain?"), "impact_soft", 0.45)],
     ["SHAKER", "KOMBU", "LETTER"])
shot(L(2), B(2, "after"), "MEDIUM (style frame C desk)",
     "The envelope opens; a lemon letter on a hot-pink underlay slides out and writes itself in squiggle lines (no "
     "legible text); on 'journal' it is addressed in squiggles.",
     "Date slip", "Lemon letter writing itself", "Envelope",
     "Overhead, slow push.",
     [("1968", W(2, "1968,"), None)], "Pan (continuous)", "Continuous",
     [(W(2, "1968,"), "pop", 0.4), (W(2, "doctor"), "typewriter", 0.35), (W(2, "journal:"), "paper", 0.4)],
     ["LETTER"], beats=[(W(2, "1968,") + 1.1, "squiggle line 2 writes")], notes="Author not named or drawn; journal not named.")
shot(B(2, "after"), L(3), "MEDIUM",
     "Two small cut-paper symptom icons pin onto the letter: a mitten hand with pins-and-needles dots (numb), a "
     "drooping arm (weak).",
     "—", "Letter + symptom icons", "—",
     "Locked.",
     [], "Continuous", "Continuous",
     [(W(2, "after"), "paper", 0.25), (W(2, "numb"), "pop_small", 0.35), (W(2, "weak."), "pop_small", 0.35)],
     ["LETTER"], beats=[(W(2, "food,"), "letter flutters")],
     notes="SENSITIVE: no restaurant, no people, no faux-Asian signifiers - paper and objects only.")
shot(L(3), L(4), "MEDIUM",
     "The turquoise card from style frame C: wine bottle, salt cellar, MSG shaker; an ink '?' drops over each on its "
     "word, then the three merge into one wobbly '?'.",
     "Wobbly '?'", "Turquoise card: bottle | cellar | shaker", "—",
     "Locked.",
     [("Guessing", W(3, "guessing."), None)], "Continuous", "Whip to the stamp",
     [(W(3, "wine?"), "blip", 0.4), (W(3, "salt?"), "blip", 0.4), (W(3, "msg?"), "blip", 0.45), (W(3, "guessing."), "boing", 0.35)],
     ["LETTER", "SHAKER", "SALT"], beats=[(W(3, "he"), "the three '?' drift together")])
shot(L(4), L(5), "MEDIUM (style frame C)",
     "Style frame C: the stamp handle slams the quoted headline across the letter; paper flecks fly; coral "
     "newspapers start sliding out from the stamp like a tomato-red anti-ripple.",
     "Stamp handle lifting", "Stamped quoted headline", "Turquoise card + newspapers sliding",
     "Whip-in, 6-frame shake, hold with drift.",
     [("\"Chinese-Restaurant Syndrome\"", W(4, "chinese"), None)], "Whip", "Cut",
     [(W(4, "journal"), "swoosh_down", 0.3), (W(4, "catchy"), "swoosh_down", 0.35), (W(4, "chinese"), "stamp", 0.8),
      (W(4, "chinese") + 0.05, "impact_big", 0.45), (W(4, "syndrome."), "paper", 0.35)],
     ["HEADLINE", "STAMP", "LETTER", "RIPPLE"], hero="Headline stamp (style frame C)",
     notes="Young Serif, hyphenated, in quotes; framed as the thing being criticised.")
shot(L(5), B(5, "but"), "MEDIUM",
     "Mouse pups on a cobalt LAB TRAY (tray rim and a lab label tab, never a plate); a comically giant syringe lowers "
     "from the top (never touches them). On 'brain damage' the coral newspapers at the edges grow bigger tomato-red "
     "headline bars.",
     "Giant syringe", "Mouse pups on the lab tray", "Newspapers at the edges",
     "Slow push-in.",
     [("Huge doses", W(5, "huge"), None)], "Cut", "Split-wipe",
     [(W(5, "mice"), "pop_small", 0.3), (W(5, "injected"), "swoosh_down", 0.35), (W(5, "huge"), "impact_soft", 0.4), (W(5, "damage."), "alarm", 0.25)],
     ["MOUSE", "SYRINGE"], notes="Lab tray, never a plate (viewer test). No gore, no needle contact.")
shot(B(5, "but"), L(6), "WIDE (top/bottom split)",
     "Top: giant syringe over a tiny pup on its lab tray. Bottom: a dinner bowl with a spoon and Goldies. A big '≠' "
     "cut-out on the divider.",
     "Syringe + pup on tray (top half)", "'≠' on the divider", "Dinner bowl + Goldies (bottom half)",
     "Locked split.",
     [], "Split-wipe", "Split-wipe",
     [(W(5, "but"), "whoosh", 0.35), (W(5, "mouse"), "pop_small", 0.3), (W(5, "nothing"), "impact_soft", 0.5), (W(5, "dinner."), "chomp", 0.35)],
     ["MOUSE", "SYRINGE", "GOLDIE"], hero="Injection is not dinner",
     notes="s06 now plays before s05, so no gut-cell cameo here (the gut cells have not been introduced yet).")
shot(L(6), B(6, "the"), "WIDE (top/bottom split)",
     "Top: a parmesan wedge under a tomato-red bell that stays DEAD STILL (no boil on the bell, no wobble). Bottom: a "
     "plain white takeout carton under a HUGE bell that rings hard: big swings, motion marks, the bottom half shakes. "
     "Identical Goldies float over both.",
     "Parmesan + dead-still bell (top half)", "Divider", "Carton + big ringing bell (bottom half)",
     "Locked split; only the bottom half shakes.",
     [], "Split-wipe", "Collapse onto the carton",
     [(W(6, "nobody"), "pop_small", 0.3), (W(6, "panicked"), "alarm", 0.5), (W(6, "parmesan."), "alarm", 0.45)],
     ["TAKEOUT", "GOLDIE"], hero="Double standard: big bell on the carton, silent bell on the parmesan",
     notes="Viewer test: the contrast must be obvious at a glance.")
shot(B(6, "the"), L(7), "MEDIUM -> CLOSE",
     "THE FEAR STICKS (strong, full frame): the split collapses onto the plain white carton, now big in MID. The stamp "
     "handle slams the quoted headline straight onto the carton's side on 'stuck', and it stays stuck: the coral "
     "newspapers swirl round the carton in a tight tomato-red anti-ripple and three ringing bells crowd in around it. "
     "Push in on the stamped name on 'name'.",
     "Ringing bells crowding in", "Carton with the headline stamped on it", "Newspapers swirling round its base",
     "Push-in, 4-frame shake on the stamp, slow push on 'name'.",
     [("\"Chinese-Restaurant Syndrome\"", W(6, "stuck"), None)], "Collapse", "Continuous",
     [(W(6, "fear"), "swoosh_down", 0.3), (W(6, "stuck"), "stamp", 0.7), (W(6, "cuisine,"), "alarm", 0.35), (W(6, "name"), "impact_soft", 0.45)],
     ["HEADLINE", "STAMP", "TAKEOUT", "RIPPLE"], hero="The fear stuck to one cuisine (strong visual, no empty frame)",
     beats=[(W(6, "loud."), "a newspaper wraps round the carton")],
     notes="The stamp on the carton shows the label sticking to one cuisine, framed as the injustice. Plain carton, quoted "
     "headline, no people, no faux-Asian signifiers.")
shot(L(7), D(), "MEDIUM",
     "BRIDGE TO THE SCIENCE: a giant ink '?' cut-out drops onto the stamped carton on 'right?'. Then the villain shaker "
     "steps into the foreground and turns to look straight at camera on 'Is MSG'; its moustache wobbles loose on 'bad'. "
     "On 'you?' a single gold drop falls from the top of frame: that drop lands in Ikeda's soup bowl at the start of s03.",
     "Gold drop falling (end)", "Shaker looking at camera", "Carton with '?' behind",
     "Slow push-in on the shaker.",
     [], "Continuous", "Match: the gold drop falls into Ikeda's bowl in s03 (no blank frame)",
     [(W(7, "right?"), "thud", 0.45), (W(7, "msg"), "boing", 0.3), (W(7, "bad"), "pop_small", 0.35), (W(7, "you?"), "swoosh_down", 0.4)],
     ["SHAKER", "TAKEOUT", "HEADLINE"], hero="Bridge: 'Is MSG actually bad for you?' -> the gold drop into Ikeda's bowl",
     beats=[(W(7, "you?") + 0.15, "the drop enters frame")])

# ================================================================ s09 outro (LOOP)
scene("s09_outro")
shot(0.0, L(3), "WIDE",
     "TV wall callback: every screen shows vivid riso channel static (no faces); the shaker sits in front with a "
     "'?'; on 'Not really' the '?' flips and Goldie pops out of the shaker smiling.",
     "TV wall (static)", "Shaker + '?'", "Goldie",
     "Slow push-in.",
     [("Not really", W(2, "not"), None)], SLIDE_IN, "Continuous",
     [(0.2, "static", 0.25), (W(1, "msg"), "pop", 0.35), (W(2, "not"), "boing", 0.35), (W(2, "really."), "sparkle", 0.35)],
     ["TVWALL", "SHAKER", "GOLDIE"], notes="No clip faces here (avoids re-crediting and a second label).")
shot(L(3), round(W(3, "outlived") - 0.15, 3), "MEDIUM",
     "Three objects drop onto the shaker like weights, each on its phrase: an ink '?' (guess), a mini tomato-red stamp "
     "impression (catchy name), a tomato-red tag on a string (prejudice).",
     "Falling object", "Shaker under the weights", "—",
     "Locked.",
     [], "Continuous", "Pull-back",
     [(W(3, "guess,"), "thud", 0.35), (W(3, "catchy"), "stamp", 0.35), (W(3, "prejudice"), "thud", 0.4)],
     ["SHAKER", "STAMP"], notes="Pictures only; the caption carries the words.")
shot(round(W(3, "outlived") - 0.15, 3), L(4), "WIDE",
     "TVs switch off one by one while a tall stack of vivid research papers rises beside them; the weights slide off.",
     "Paper stack top", "TV wall switching off", "Stack base",
     "Pull-back.",
     [], "Pull-back", "Paper slide carrying the NO MSG window",
     [(W(3, "outlived"), "tv_off", 0.45), (W(3, "decades"), "tv_off", 0.45), (W(3, "of"), "tv_off", 0.45), (W(3, "science."), "paper", 0.4)],
     ["TVWALL"], hero="TVs switch off")
shot(L(4), B(4, "reach"), "MEDIUM",
     "The NO MSG sign from s01 in its window; the 'NO' tube flickers and goes out on 'sign...', leaving 'MSG' lit gold.",
     "Window top", "Sign", "Window sill",
     "Slow push-in.",
     [("No MSG", L(4), W(4, "sign...")), ("MSG", W(4, "sign..."), None)], "Paper slide carrying the NO MSG window", "Dissolve through the window",
     [(L(4) + 0.1, "neon_buzz", 0.3), (W(4, "no"), "neon_flicker", 0.45), (W(4, "sign..."), "click", 0.45), (W(4, "sign...") + 0.3, "sparkle", 0.35)],
     ["SIGN"], hero="'NO' burns out")
shot(B(4, "reach"), D(), "CLOSE -> WIDE (LOOP)",
     "Lamp-lit kitchen table: ramen bowl; a mitten hand reaches for the MSG shaker and shakes; one gold drop hits the "
     "broth and makes the Fifth Ripple (end card: 'MSG' slip on the ripple). Then pull back: the bowl is on the hero "
     "TV's screen; push in until the screen fills the frame exactly like s01 frame 0, and on the last frames the screen "
     "flickers to clip blaylock_hungry's first frame with the name lower third, so the last frame equals s01 frame 0.",
     "TV wall (end)", "Ramen ripple -> hero TV", "'MSG' end-card slip",
     "Push-in on the ramen, pull-back to the hero TV, push-in to the s01.01 framing.",
     [("MSG", W(4, "shaker."), None)], "Dissolve", "LOOP: last frame = s01 frame 0 layout; flicker on the final frames",
     [(W(4, "reach"), "pop_small", 0.35), (W(4, "shaker."), "pop_small", 0.35), (W(4, "shaker.") + 0.4, "bubble", 0.4),
      (W(4, "shaker.") + 0.6, "chime", 0.4), (D() - 0.15, "channel_switch", 0.45)],
     ["SHAKER", "GOLDIE", "RIPPLE", "TVWALL", "END_CARD"], hero="End card -> loop into the hook",
     notes="Last frame = s01 frame 0: big hero screen showing blaylock_hungry frame 0 + lower third. No fade, no blank frame.")

# ================================================================ continuity, guardrails
CONTINUITY = {
    "GOLDIE": ("Goldie, the broth droplet", "KP.goldie: gold #FFB000 droplet, dot eyes, pink cheeks, steam curl. Gold is "
               "reserved for glutamate. Friendly, never menacing; s04 copies identical."),
    "GLU_DIAGRAM": ("Glutamate skeletal diagram", "Flat cut-paper five-carbon zigzag beside Goldie in s04, so the droplet "
                    "is never read as the literal molecule."),
    "SHAKER": ("MSG shaker", "One prop. Identity without lettering: gold crystals + Goldie sticker; text only when it is "
               "the shot's one label."),
    "SALT": ("Salt shaker", "Same silhouette, white crystals and ink-dot cap; no lettering."),
    "TVWALL": ("TV wall (style frame A layout)", "Violet wall, grape dim sets, turquoise hero TV spanning two columns. "
               "s01.01-02 are pushed in so the hero screen fills most of the frame (client: clip big); s01.01 frame 0 and the "
               "last frame of s09 are identical (blaylock_hungry frame 0 at that framing)."),
    "CLIPS": ("Real interview clips", "Two real clips of Dr. Russell Blaylock (blaylock_brain, blaylock_hungry; lib/clips.js) play only inside the hero "
              "TV screen, unaltered in content, with paper cut-outs (brain, shaker, zig-zags) placed beside the screen, never on "
              "the person. Placeholder rectangle until delivered."),
    "NAME_LT": ("Name + credit lower third", "Blue torn-paper slip: the person's real name (the shot's one label) + a "
                "smaller credit line (outlet, date). On every frame a clip face is readable. Replaces the old "
                "DRAMATIZATION tag."),
    "SIGN": ("NO MSG sign", "Paper 'neon': separate NO and MSG tubes, flat colour, no glow; same window and angle in s01 and s09."),
    "GUTCELL": ("Gut cells with bibs", "KP.cell: pink cells with white bibs, two tiers, chomp and gold-flame states. "
                "Canonical in s05.02."),
    "BADGE95": ("95% number", "Young Serif big number; the only percentage in the film."),
    "BODY": ("Body interior", "Layered pink paper tints for the dive in s05.01."),
    "BBB": ("Blood-brain barrier", "Cobalt cut-paper castle wall with friendly paper-doll guards."),
    "NA": ("Sodium ion", "Small cobalt paper dot, no lettering; s08 count exactly 3:1."),
    "TOMATO": ("Tomato", "Same ripe tomato paper cut throughout s04."),
    "KOMBU": ("Kombu seaweed", "A long ruffled kelp blade: wavy edges, a midrib, a small holdfast, kombu green with "
              "turquoise under-print and a few bubbles. Must read as seaweed at thumbnail size (s03, s06)."),
    "RIPPLE": ("The Fifth Ripple", "KP.ripple: drop + 5 rings, the fifth gold and wider: s03 (drop into the bowl, taste rings), s06 (anti-ripple), "
               "s07 (chart baseline), s09 (ramen, end card)."),
    "HEADLINE": ("Stamped headline", "KP.stamp, Young Serif, '\"CHINESE-RESTAURANT SYNDROME\"' over two lines, tomato "
                 "red, distressed. One render, used in s06.05 and stamped onto the carton in s06.09."),
    "STAMP": ("Rubber stamp", "Ink handle with tomato-red pad; same prop for the headline, GRAS and the s09 mini stamp."),
    "LETTER": ("1968 letter", "Lemon letter on hot-pink underlay, squiggle lines only (KP.squiggle), no legible words; "
               "the sealed envelope arrives in s01.04-05 and carries the match cut into s06."),
    "TASTE_ICONS": ("Four taste icons", "Candy, lemon, salt crystal, coffee bean; riding the first four rings."),
    "IKEDA": ("Kikunae Ikeda", "Respectful paper doll, high collar, round glasses, moustache; never comedic."),
    "CAPTION": ("Caption strip", "KP.caption: torn white strip, Nunito Black, spoken words ink, upcoming lavender, "
                "current word on a raised chip (pink cold open, gold on glutamate, tomato red in s06)."),
    "END_CARD": ("End card", "'MSG' slip on the ramen Fifth Ripple, then the pull-back into the TV wall that loops to s01."),
}
INCIDENTAL = {"TAKEOUT", "MOUSE", "SYRINGE", "CAPSULES", "VOLUNTEERS"}

GUARDRAILS = {
    "allowed_numbers_and_claims": [
        "s01: the two Blaylock clips' own words, unaltered and in context ('They put so much MSG in food because it makes you hungry...'; 'Excitotoxicity destroys brain cells.'), credited 'Neurosurgeon, lecture on MSG, 2012'; narrator: 'for fifty years' (label '50 years'), everyone has been scared of MSG, it started with one letter in 1968",
        "s03: 1908; Japanese chemist Kikunae Ikeda tasted something odd in his seaweed soup; not sweet, sour, salty or bitter; called it umami; turned it into a seasoning: monosodium glutamate",
        "s04: glutamate is everywhere: tomatoes, parmesan, even breast milk; tomato glutamate and MSG glutamate are the exact same molecule",
        "s05: gut cells burn glutamate as fuel; ~95% never reaches the blood; the rest hits the blood-brain barrier, one of the best-guarded borders",
        "s07: in double-blind studies, people who said MSG made them sick reacted as often to a placebo; the FDA calls it generally recognized as safe (GRAS), like salt",
        "s08: a third of the sodium of table salt; same flavour, less sodium",
        "s06: 1968 doctor's letter to a medical journal; after Chinese food he felt numb and weak; wine, salt, MSG: he was guessing; the journal's headline 'Chinese Restaurant Syndrome'; mice injected with huge doses got brain damage; injecting a baby mouse is nothing like eating dinner; nobody panicked about parmesan; the fear stuck to one cuisine; 'But was he right? Is MSG actually bad for you?'",
        "s09: 'Not really'; a guess, a catchy name and a little prejudice outlived decades of science; No MSG sign; reach for the shaker",
    ],
    "forbidden": [
        "Anything this cut removed: tongue/receptors/'protein', amino acid, mushrooms, glutamic acid, sodium pairing in s03, the seaweed mountain, sugar, tomato sauce in s06, 'savory', 'Tokyo', '~10x', 1969, the journal's name, 'Italian Restaurant Syndrome', 'even water'.",
        "Any number, %, axis value, study count, dose or date not listed above; never '5%'.",
        "Health claims beyond the script; real logos, seals, mastheads or brands (except as they appear inside a licensed clip); real people other than the credited Blaylock clips and Ikeda.",
        "Editing, overlaying or re-timing a real clip so it says something the person did not say; cut-outs sit beside the screen, never on the person.",
        "Gore: no needle touching an animal. Mice on a lab tray, never a plate.",
        "Dull colours (grey, beige, cream, muddy brown, desaturated); glows and gradients.",
        "More than one 1-3 word label at a time (the credit line under a real name is the only extra text).",
    ],
    "real_clips": "Every frame with a readable clip face (s01.01-02) carries the name + credit lower third. Clip rights, exact attribution and context must be cleared by the client before publishing; captions for those lines must match the clip audio.",
    "sensitive_1960s_section": [
        "s06 shows only paper, type, a bottle, a salt cellar, a shaker, mice on a lab tray, parmesan, a dinner bowl and a plain white carton: no restaurants, staff or diners.",
        "No faux-Asian fonts or signifiers (pagodas, lanterns, dragons, chopsticks as decoration, conical hats); the carton is plain white.",
        "The headline is quoted, hyphenated and stamped as the thing being criticised; never a joke or a brand; stamped onto the carton in s06.09 as the injustice.",
        "The prejudice beat is a big ringing bell over the carton and a dead-still bell over the parmesan; no people are shown being othered.",
        "Ikeda is drawn respectfully; the letter author and the mouse researchers are never named or drawn.",
    ],
}

ORDER = ["s01_coldopen", "s06_origin", "s05_body", "s03_ikeda", "s04_everywhere", "s07_tests", "s08_twist", "s09_outro"]
ORDER_WHY = {
    "s01_coldopen": "BUILT - rework: clip order swapped (hungry first), camera pushed in so the clip is big, flash-forward removed, last shot hands the envelope over to s06 by match cut.",
    "s06_origin": "BUILT - rework: now 2nd in the Reel; opens on the s01 desk; gut-cell cameo removed; s06.09 'fear stuck' becomes a full-frame stamp-on-carton beat; new s06.10 bridge ('Is MSG actually bad for you?') drops the gold drop into Ikeda's bowl (s03).",
    "s05_body": "BUILT - re-time only (still between s04 and s07).",
    "s03_ikeda": "scenes/s03_ikeda.js exists (previous cut) - rework to 5 shots; now opens on the s06.10 gold drop landing in the bowl, and is where Goldie is first seen (s03.04), since the title card is gone.",
    "s04_everywhere": "To build: Earth of Goldies, 3-food stack, equality split.",
    "s07_tests": "To build: 3 shots (shuffle + volunteers, chart, GRAS + shelf).",
    "s08_twist": "To build: scale + ions, pot + meter; now hands straight into s09.",
    "s09_outro": "To build last: its final frame must equal s01 frame 0 (big blaylock_hungry frame 0).",
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
        if (dur < 2.0 and sid != "s02_title" and not sh["flash"]) or dur > 6.5: errors.append(f"{sh['id']}: duration {dur:.2f}")
        if sh["flash"] and not 1.2 <= dur <= 2.0: errors.append(f"{sh['id']}: flash-forward should be ~1.5 s, is {dur:.2f}")
        for tr in (sh["transition_in"], sh["transition_out"]):  # viewer test: no blank paper slides
            if "slide" in tr.lower() and "carrying" not in tr.lower(): errors.append(f"{sh['id']}: slide transition must carry content: '{tr}'")
            if re.search(r"\b(black|blank|empty)\b", tr, re.I) and not re.search(r"\b(no|never)\b", tr, re.I): errors.append(f"{sh['id']}: empty transition '{tr}'")
        ev = sorted([sh["start"], sh["end"], *[x["t"] for x in sh["sfx"]], *[x["in"] for x in sh["labels"]],
                     *[x["t"] for x in sh["beats"]]])
        ev = [min(max(e, sh["start"]), sh["end"]) for e in ev]
        still = max(b - a for a, b in zip(ev, ev[1:]))
        if still > MAX_STILL + 1e-6: errors.append(f"{sh['id']}: holds {still:.2f}s without an action (add a beat)")
        sh["max_still"] = round(still, 2)
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
        if "CLIPS" in sh["props"] and not (sh["credit_lower_third"] and "NAME_LT" in sh["props"]): errors.append(f"{sh['id']}: real clip without name + credit")
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
                                                      "layout", "camera", "labels", "credit_lower_third", "flash", "beats",
                                                      "max_still", "transition_in", "transition_out", "sfx", "props",
                                                      "hero", "notes"]} for sh in shots]})
    offset += Dd
# global label budget across consecutive shots is per-shot; check hook + loop
first, last = all_shots["s01.01"], all_shots[f"{reel['scenes'][-1][:3]}.{len(SCENES[reel['scenes'][-1]]):02d}"]
if not first["labels"] or first["labels"][0]["in"] > 0.3: errors.append("hook: label must be up by 0.3 s")
if first["transition_in"].split(":")[0] != "NONE": errors.append("hook: s01 must open lit on frame 0")
order = reel["scenes"]
for x, y in zip(order, order[1:]):  # every scene join must hand content across (no blank frame)
    out_tr = SCENES[x][-1]["transition_out"]; in_tr = SCENES[y][0]["transition_in"]
    if not re.search(r"carrying|match|ripple|continu|cut", out_tr + " " + in_tr, re.I): errors.append(f"join {x}->{y}: no content carried")
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
out = {"meta": {"project": "MSG: The Most Misunderstood Molecule", "format": "single Instagram Reel, 9:16, loops (viewer-test cut)",
                "viewer_test_rules": ["no shot holds still > 2 s (validated from sfx/labels/beats)", "mice on a lab tray, never a plate",
                                      "bell BIG on the carton, dead still on the parmesan", "seaweed must read as seaweed",
                                      "nothing important cropped at frame edges (40 px inside the key box)", "hook within 2 s"],
                "duration": round(offset, 3), "fps": 30, "resolution": [1080, 1920], "frame": FRAME,
                "style": "Kombu Paper (design/APPROVED.md; style frames design/frames/kombu-paper/A,B,C; kit window.KP in frames.js)",
                "time_base": "seconds, scene-relative (the t passed to draw); reel_time = seconds from the start of the Reel",
                "label_rule": "at most one 1-3 word label at a time besides captions (+ the credit line under a real name)",
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
  slip, stamp, sign, lower third). Each shot lists its label(s) with in/out times; they never overlap. Real-clip
  shots add only the credit line under the person's name in the same lower third. Props carry no lettering unless
  the lettering is that shot's label. Everything else is told with pictures.
- **Viewer-test rules (validated where possible):** no shot holds still for more than 2 s: every shot lists SFX,
  label and secondary-action **beats** so the gap between visible events is at most 2 s (`max_still` in the JSON);
  hook inside 2 s; nothing important within 40 px of the key-box edges; seaweed reads as seaweed; mice on a lab
  tray, never a plate; the s06 bell is big on the carton and dead still on the parmesan.
- **Real clips (s01):** the talking heads are real interview clips, still pending from the client. Build around a
  placeholder rectangle inside the hero TV; cut-outs sit beside the screen, never on the person.
- **Kombu Paper vocabulary:** cut paper with hard offset shadows and riso misregistration, boil on twos, no
  outlines, no glows, no gradients, vivid riso stock only, gold reserved for glutamate (Goldie). Use the `KP` kit
  (`design/frames/kombu-paper/frames.js`).
- **Transitions:** every paper slide carries the next shot's art on the sheet (viewer test: two slides flashed a
  blank colour; never again). Gold Ripple wipe on act changes, match cuts at s01→s06 (envelope) and s06→s03 (gold
  drop into Ikeda's bowl), channel flicker on
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
        if sh["credit_lower_third"]:
            lab += " + credit line"
        if sh["flash"]:
            lab += " (flash-forward)"
        md += ["",
               f"- **Action:** {sh['on_screen']}",
               f"- **TOP:** {lay['TOP']}  \n  **MID:** {lay['MID']}  \n  **LOW:** {lay['LOW']}",
               f"- **Label:** {lab}",
               f"- **Camera:** {sh['camera']}",
               *([f"- **Secondary beats:** " + "; ".join(f"{b['t']:.2f} {b['action']}" for b in sh["beats"])] if sh["beats"] else []),
               f"- **Longest still:** {sh['max_still']:.2f} s",
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
md += ["## Factual guardrails", "", "### Allowed (only what the viewer-test script says)", ""]
md += [f"- {x}" for x in GUARDRAILS["allowed_numbers_and_claims"]]
md += ["", "### Forbidden", ""] + [f"- {x}" for x in GUARDRAILS["forbidden"]]
md += ["", "### Real clips", "", GUARDRAILS["real_clips"], "", "### 1960s section / cultural sensitivity", ""]
md += [f"- {x}" for x in GUARDRAILS["sensitive_1960s_section"]]
open(os.path.join(HERE, "SHOTLIST.md"), "w").write("\n".join(md) + "\n")

print({s["id"]: len(s["shots"]) for s in scenes_out}, n_shots, "shots,", len(heroes), "heroes,", f"{offset:.1f}s")
