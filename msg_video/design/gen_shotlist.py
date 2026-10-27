#!/usr/bin/env python3
"""Shot-list generator for the MSG Reel series (9:16, 3 parts).

Single source for design/shotlist.json and design/SHOTLIST.md:
    python3 design/gen_shotlist.py
Timings come from cues.json (line starts + per-word "words" timings) via L()/W(), so after a narration
re-record just re-run this script; it fails loudly if a referenced word disappeared or shots stop tiling.
Edit the shot data below, never the generated files.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUES = json.load(open(os.path.join(ROOT, "cues.json")))
sys.dont_write_bytecode = True
sys.path.insert(0, ROOT)
import script  # noqa: E402  (PARTS, SCENES)

ALLOWED = set("""whoosh whoosh_long swoosh_up swoosh_down pop pop_small bubble blip click tick ding chime sparkle
impact_soft impact_big thud stamp boing riser record_scratch tv_on tv_off channel_switch static glitch
neon_buzz neon_flicker typewriter paper heartbeat alarm crunch slurp chomp sizzle bubbling_pot cash error success""".split())

# ---------------------------------------------------------------- frame geometry (1080x1920)
FRAME = {
    "size": [1080, 1920],
    "ui_top": [0, 220],            # Instagram header: background only
    "ui_bottom": [1480, 1920],     # Instagram caption/buttons: background only
    "right_gutter_x": [940, 1080], # Instagram action buttons: no key content
    "left_margin_x": [0, 40],
    "caption_band": [1300, 1480],  # burned-in animated captions; no key content
    "key_box": {"x": [40, 940], "y": [220, 1300], "center_x": 490},
    "zones": {"TOP": [220, 520], "MID": [520, 1100], "LOW": [1100, 1300]},
}

SCENES, CUR = {}, None
_used_words = []


def scene(sid):
    global CUR
    CUR = sid


def L(n):
    """Start of narration line n (1-based) in the current scene."""
    return CUES[CUR]["lines"][n - 1]["start"]


def _norm(w):
    return re.sub(r"[^a-z0-9\-']", "", w.lower())


def W(n, word, k=1):
    """Start time of the k-th occurrence of `word` in line n of the current scene."""
    hits = [w for w in CUES[CUR]["lines"][n - 1]["words"] if _norm(w["w"]) == _norm(word)]
    if len(hits) < k:
        raise SystemExit(f"{CUR} L{n}: word '{word}' #{k} not found")
    _used_words.append((CUR, n, word))
    return round(hits[k - 1]["start"], 3)


def B(n, word, k=1):
    """Shot boundary 0.1 s before a word (visual leads the word slightly)."""
    return round(W(n, word, k) - 0.1, 3)


def shot(start, end, size, action, top, mid, low, camera, text, t_in, t_out, sfx, props=(), hero=None, notes=None):
    SCENES.setdefault(CUR, []).append(dict(
        start=round(start, 3), end=round(end, 3), shot_size=size, on_screen=action,
        layout={"TOP": top, "MID": mid, "LOW": low}, camera=camera, text=list(text),
        transition_in=t_in, transition_out=t_out,
        sfx=[dict(t=round(a, 3), name=b, gain=c) for a, b, c in sfx], props=list(props), hero=hero, notes=notes))


D = lambda: CUES[CUR]["duration"]  # noqa: E731
DIP_IN, DIP_OUT = "Dip from black (0.3 s, internal join)", "Dip to black (0.3 s, internal join)"
HOOK_IN = "NONE - fully lit on frame 0 (part opener / hook)"
HOLD_OUT = "Hold end card to last frame (no fade; Reel loops)"

# ================================================================ PART 1 — s01 cold open
scene("s01_coldopen")
shot(0.0, L(2), "CLOSE -> WIDE",
     "HOOK: opens already lit on TV B2 (fills MID): TALK SHOW HOST silhouette mid-gesture. Keyword slams into TOP at "
     "frame ~5. On 'chemical' the camera pulls back to reveal the 3x3 TV wall (other TVs show static).",
     "PART 1/3 · THE ACCUSATION tag (left) + DRAMATIZATION pill; keyword HIDDEN CHEMICAL?",
     "TV B2 close, then the full 3x3 wall (grid ~860 px wide, centered on x=490)",
     "Role lower-third: TALK SHOW HOST + 'DRAMATIZATION - fictional character'",
     "Start close on B2; pull back from W('chemical') to full wall.",
     ["PART 1/3", "DRAMATIZATION", "TALK SHOW HOST", "HIDDEN CHEMICAL?"],
     HOOK_IN, "Channel-switch glitch",
     [(0.0, "tv_on", 0.6), (0.17, "impact_soft", 0.5), (0.4, "static", 0.2), (W(1, "chemical"), "whoosh", 0.35)],
     ["TVWALL", "DRAMA_TAG", "PART_TAG"], hero="Part 1 hook: lit TV + keyword inside 0.2 s")
shot(L(2), L(3), "MEDIUM (single TV)",
     "TV A1 fills MID: DAYTIME TV DOCTOR silhouette (coat, stethoscope). Inset brain icon in the lower part of the "
     "screen with jittery zig-zags that fizzle out on 'Literally to death' (no gore, no skulls).",
     "Keyword EXCITOTOXIN?! (on the word) + DRAMATIZATION pill",
     "TV A1, speaker upper half of screen, brain inset lower half",
     "Lower-third: DAYTIME TV DOCTOR + DRAMATIZATION tag",
     "Diagonal whip up-left B2 -> A1, then slow drift down.",
     ["DRAMATIZATION", "DAYTIME TV DOCTOR", "EXCITOTOXIN?!"],
     "Channel-switch glitch (2-4 frames)", "Channel-switch glitch",
     [(L(2) - 0.05, "channel_switch", 0.6), (L(2) + 0.05, "glitch", 0.3), (W(2, "excitotoxin"), "pop_small", 0.35),
      (W(2, "literally"), "blip", 0.3)],
     ["TVWALL", "DRAMA_TAG"])
shot(L(3), L(4), "MEDIUM (single TV)",
     "TV C1: CELEBRITY CHEF silhouette (toque), arms crossed. On 'Never!' a warning X slams over an MSG shaker "
     "on the chef's counter.",
     "Keyword POISON! (on the word) + DRAMATIZATION pill",
     "TV C1: chef upper half, crossed-out shaker lower half",
     "Lower-third: CELEBRITY CHEF + DRAMATIZATION tag",
     "Diagonal whip A1 -> C1; 5% punch-in on 'Never!'.",
     ["DRAMATIZATION", "CELEBRITY CHEF", "POISON!"],
     "Channel-switch glitch", "Channel-switch glitch",
     [(L(3) - 0.05, "channel_switch", 0.6), (W(3, "never!"), "stamp", 0.45), (W(3, "poison"), "pop_small", 0.3)],
     ["TVWALL", "DRAMA_TAG", "SHAKER"])
shot(L(4), L(5), "MEDIUM (phone frame inside TV)",
     "TV A3 shows a vertical phone video: WELLNESS INFLUENCER silhouette with ring-light halo. Headache zig-zag "
     "and racing heart pop beside the head. A PLAIN generic white takeout carton icon (no print, no lettering).",
     "Keyword HEADACHES? + DRAMATIZATION pill",
     "Phone frame (vertical, natural for 9:16) filling MID",
     "Lower-third: WELLNESS INFLUENCER + DRAMATIZATION tag",
     "Whip down to A3; slow drift up.",
     ["DRAMATIZATION", "WELLNESS INFLUENCER", "HEADACHES?"],
     "Channel-switch glitch", "Channel-switch glitch",
     [(L(4) - 0.05, "channel_switch", 0.6), (W(4, "headaches"), "pop_small", 0.35), (W(4, "heart"), "heartbeat", 0.35)],
     ["TVWALL", "DRAMA_TAG", "TAKEOUT"])
shot(L(5), L(6), "MEDIUM -> CLOSE",
     "TV C3: a generic social-media post card types itself out (illegible placeholder lines); share arrow bounces. "
     "No real platform UI, logo or colours.",
     "Keyword BRAIN POISON!! + DRAMATIZATION pill",
     "Post card filling MID",
     "Lower-third: YOUR AUNT, ON SOCIAL MEDIA + DRAMATIZATION tag",
     "Whip to C3, slow push into the card.",
     ["DRAMATIZATION", "YOUR AUNT, ON SOCIAL MEDIA", "BRAIN POISON!!"],
     "Channel-switch glitch", "Fast pull-back (continuous)",
     [(L(5) - 0.05, "channel_switch", 0.55), (L(5) + 0.1, "typewriter", 0.3), (W(5, "please"), "ding", 0.35)],
     ["TVWALL", "DRAMA_TAG"])
shot(L(6), L(7), "EXTREME WIDE",
     "All five speaker TVs on at once, looping their keywords; the wall recedes into darkness (it may extend into "
     "the UI zones as background). On 'three little letters' giant M-S-G letters rise in front.",
     "50+ YEARS (on 'fifty')",
     "Receding TV wall; letters M S G stacked or in a row in front",
     "—",
     "Fast pull-back, then slow continuous pull-back.",
     ["50+ YEARS", "MSG"],
     "Continuous", "Cut",
     [(L(6), "whoosh", 0.5), (L(6) + 0.3, "static", 0.2), (W(6, "three"), "pop", 0.45)],
     ["TVWALL", "DRAMA_TAG"], notes="DRAMATIZATION tags stay legible on any TV whose head is still readable.")
shot(L(7), B(7, "food"), "MEDIUM",
     "Night street: a restaurant window; the NO MSG neon sign buzzes on (NO, then MSG), stutters once, holds.",
     "Night sky / street top",
     "NO MSG sign, ~800 px wide, upper MID",
     "Window frame with warm interior glow",
     "Slow push-in toward the sign.",
     ["NO MSG (the sign)"],
     "Cut", "Tilt down to supermarket shelf",
     [(W(7, "signs") - 0.1, "neon_flicker", 0.55), (W(7, "signs") + 0.2, "neon_buzz", 0.4), (W(7, "don't"), "neon_flicker", 0.3)],
     ["SIGN"], hero="First appearance of the NO MSG sign (paid off in s09; teased in s08)")
shot(B(7, "food"), L(8), "MEDIUM",
     "Supermarket shelf, three shelves stacked vertically with generic fictional packages; badge stickers stamp "
     "onto them one by one.",
     "NO MSG badge",
     "Shelves 1-2 with packages",
     "Shelf 3 (NO ADDED MSG)",
     "Tilt down shelf by shelf.",
     ["NO MSG", "NO ADDED MSG"],
     "Tilt from sign", "Cut",
     [(W(7, "labels"), "stamp", 0.4), (W(7, "brag"), "stamp", 0.35), (W(7, "leaving"), "stamp", 0.3)],
     [])
shot(L(8), L(9), "CLOSE",
     "MSG shaker alone in a spotlight, crystals glinting; a big '?' pops above it.",
     "MSG? keyword, then '?'",
     "Shaker centered at x=490",
     "Table edge / shadow",
     "Very slow push-in.",
     ["MSG?"],
     "Cut", "Freeze on record scratch",
     [(W(8, "msg?"), "pop", 0.45), (W(8, "is", 2), "pop", 0.4), (W(8, "bad"), "sparkle", 0.3)],
     ["SHAKER"])
shot(L(9), D(), "CLOSE -> WIDE",
     "Record scratch after 'isn't...': frame freezes ~0.3 s. On 'why' the camera pulls back: shaker small in front "
     "of the dimmed TV wall, NO MSG sign glowing below it. 'WHY?' rises.",
     "WHY?",
     "Dimmed TV wall",
     "Shaker + NO MSG sign (both above the caption band)",
     "Freeze, then smooth pull-back.",
     ["WHY?"],
     "Freeze (no cut)", DIP_OUT,
     [(W(9, "why") - 0.3, "record_scratch", 0.7), (W(9, "why"), "pop", 0.45), (D() - 0.8, "whoosh", 0.35)],
     ["SHAKER", "TVWALL", "SIGN"], hero="Record scratch + WHY? (the series' central question)")

# ---------------------------------------------------------------- s02 title
scene("s02_title")
shot(0.0, 1.4, "WIDE",
     "Mascot pops in at center with crystals orbiting; 'MSG' lands on the spoken word.",
     "—", "Mascot (y ~650-900)", "MSG (large, on W('MSG.'))",
     "Slow push-in for whole scene.",
     ["MSG"], DIP_IN, "Continuous",
     [(0.0, "whoosh_long", 0.6), (W(1, "msg."), "impact_soft", 0.8)],
     ["GLU"], notes="Existing scenes/s02_title.js is 16:9: recompose to vertical. Move the 'MSG' hit from 1.6 s to the spoken word.")
shot(1.4, D(), "WIDE",
     "Subtitle rises under the bobbing mascot.",
     "—", "Mascot", "MSG + 'The Most Misunderstood Molecule' (2 lines)",
     "Continued push-in.",
     ["MSG", "The Most Misunderstood Molecule"], "Continuous", DIP_OUT,
     [(W(1, "misunderstood"), "sparkle", 0.6), (W(1, "kitchen."), "pop", 0.45)],
     ["GLU"], hero="Title card; mascot canonical from here",
     notes="Suppress burned-in captions during s02: the title text is the caption.")

# ---------------------------------------------------------------- s03 Ikeda
scene("s03_ikeda")
shot(0.0, L(2), "EXTREME WIDE (establishing)",
     "Tokyo 1908 at dusk, tall vertical composition: sky at top, period wooden two-storey street receding, a lit "
     "laboratory window. Respectful, period-plausible, not exoticized.",
     "Chapter label TOKYO, 1908", "Street receding upward", "Lit window",
     "Descending crane (tilt down) toward the window.",
     ["TOKYO, 1908"], DIP_IN, "Push through the window",
     [(0.2, "whoosh_long", 0.45), (W(1, "tokyo"), "chime", 0.4), (W(1, "1908."), "pop_small", 0.3)],
     ["CHAPTER"])
shot(L(2), L(3), "MEDIUM",
     "Professor Ikeda, waist-up, blowing on a steaming bowl. A kombu strip floats up out of the bowl as an icon.",
     "Name tag KIKUNAE IKEDA; then KOMBU icon+label", "Ikeda waist-up", "Steaming bowl",
     "Slow push-in.",
     ["KIKUNAE IKEDA", "KOMBU"], "Continuous", "Cut",
     [(W(2, "kikunae"), "pop_small", 0.4), (W(2, "soup"), "slurp", 0.35), (W(2, "kombu"), "pop_small", 0.35)],
     ["IKEDA"])
shot(L(3), B(3, "it"), "MEDIUM-CLOSE",
     "Four taste icons pop around Ikeda's head (2x2 around the head) on each word, each dimming right after.",
     "SWEET / SOUR (upper corners)", "Ikeda's head", "SALTY / BITTER (lower corners)",
     "Locked with drift.",
     ["SWEET", "SOUR", "SALTY", "BITTER"], "Cut", "Continuous",
     [(W(3, "sweet"), "pop", 0.4), (W(3, "sour"), "pop", 0.4), (W(3, "salty"), "pop", 0.4), (W(3, "bitter"), "pop", 0.4)],
     ["TASTE_ICONS", "IKEDA"])
shot(B(3, "it"), L(4), "CLOSE",
     "A fifth empty slot glows with a pulsing '?'; three words drift up in sequence.",
     "SAVORY · MEATY · DEEP (stacked)", "Fifth slot '?'", "—",
     "Push-in on the fifth slot.",
     ["SAVORY", "MEATY", "DEEP"], "Continuous", "Cut",
     [(W(3, "else."), "sparkle", 0.45), (W(3, "savory."), "pop_small", 0.3), (W(3, "meaty."), "pop_small", 0.3), (W(3, "deep."), "pop_small", 0.3)],
     ["TASTE_ICONS"])
shot(L(4), B(4, "and"), "WIDE (tall)",
     "Lab: a comically tall pile of seaweed tumbles from the top of frame into a giant pot; it boils.",
     "Seaweed falling in", "Pile + pot", "Burner",
     "Tilt down with the falling seaweed.",
     [], "Cut", "Time-lapse dissolve",
     [(W(4, "boiled"), "thud", 0.5), (W(4, "down"), "bubbling_pot", 0.4)],
     ["IKEDA"])
shot(B(4, "and"), L(5), "WIDE -> CLOSE",
     "Time-lapse: pile shrinks, steam rises, boils down to a dish with a few glinting crystals.",
     "Steam", "Shrinking pot -> dish", "GLUTAMIC ACID label",
     "Tilt down from steam to dish, push-in.",
     ["GLUTAMIC ACID"], "Time-lapse dissolve", "Cut",
     [(W(4, "and") - 0.05, "swoosh_down", 0.4), (W(4, "source:"), "sparkle", 0.45), (W(4, "glutamic"), "pop", 0.4)],
     [])
shot(L(5), L(6), "MEDIUM",
     "The fifth slot fills: UMAMI pops big; the other four tastes return around it as equals.",
     "UMAMI", "Five-taste ring", "'delicious taste'",
     "Slight push-in, breathing scale.",
     ["UMAMI", "'delicious taste'"], "Cut", "Cut",
     [(W(5, "umami."), "impact_soft", 0.7), (W(5, "umami.") + 0.15, "chime", 0.45), (W(5, "delicious"), "pop_small", 0.3)],
     ["TASTE_ICONS"], hero="UMAMI is named")
shot(L(6), B(6, "making"), "CLOSE",
     "Glutamate mascot (first time in the story) meets a small sodium ion character; they click together.",
     "+ Na", "Mascot + sodium ion", "—",
     "Locked, gentle drift.",
     ["+ Na (sodium)"], "Cut", "Continuous",
     [(W(6, "seasoning"), "pop", 0.35), (W(6, "sodium"), "click", 0.45)],
     ["GLU", "NA"])
shot(B(6, "making"), L(7), "CLOSE -> MEDIUM",
     "The pair tiles into a growing salt-like crystal; full name appears with M, S, G highlighted.",
     "MONOSODIUM GLUTAMATE (2 lines)", "Growing crystal", "M · S · G",
     "Pull-back as the crystal grows.",
     ["MONOSODIUM GLUTAMATE", "M-S-G"], "Continuous", "Cut",
     [(W(6, "salt-like"), "sparkle", 0.45), (W(6, "monosodium"), "ding", 0.4)],
     ["GLU", "NA"])
shot(L(7), D(), "MEDIUM",
     "Crystals pour into the MSG shaker (first canonical appearance), which lands on a period shop counter.",
     "ON SALE", "Shaker", "Counter",
     "Small push-in, hold.",
     ["ON SALE"], "Cut", DIP_OUT,
     [(W(7, "within"), "pop_small", 0.35), (W(7, "sale."), "cash", 0.45)],
     ["SHAKER"], notes="No year on screen: the script only says 'within a year'.")

# ---------------------------------------------------------------- s04 everywhere (ends part 1)
scene("s04_everywhere")
shot(0.0, L(2), "CLOSE",
     "Mascot center; a flask and beaker pop in above it; on 'lab-made' they get a big X and slide off; mascot shrugs.",
     "Flask + beaker -> X, LAB-MADE?", "Mascot", "—",
     "Gentle drift.",
     ["LAB-MADE?"], DIP_IN, "Zoom-out (continuous)",
     [(W(1, "glutamate"), "pop", 0.45), (W(1, "lab-made"), "stamp", 0.35)],
     ["GLU"])
shot(L(2), B(2, "building"), "CLOSE -> MEDIUM",
     "Zoom out: the mascot is one bead in a long VERTICAL chain of amino-acid beads running through the frame.",
     "AMINO ACID", "Vertical bead chain", "Chain continues",
     "Smooth zoom-out.",
     ["AMINO ACID"], "Continuous", "Continuous",
     [(L(2) + 0.1, "swoosh_up", 0.45), (W(2, "amino"), "pop_small", 0.35), (W(2, "earth,"), "bubble", 0.3)],
     ["GLU"])
shot(B(2, "building"), L(3), "WIDE",
     "The chain folds into a protein blob that drifts into a column of living things: leaf, fish, cell, person.",
     "PROTEIN", "Leaf / fish / cell / person (2x2)", "EVERY LIVING THING",
     "Continued zoom-out.",
     ["PROTEIN", "EVERY LIVING THING"], "Continuous", "Cut",
     [(W(2, "proteins"), "whoosh", 0.4), (W(2, "every"), "pop_small", 0.3), (W(2, "living"), "pop_small", 0.3), (W(2, "thing."), "pop_small", 0.3)],
     [])
shot(L(3), B(3, "so"), "MEDIUM",
     "Ripe cut-away tomato glowing softly with tiny glutamate mascots bobbing inside.",
     "TOMATO", "Tomato", "—",
     "Push-in.",
     ["TOMATO"], "Cut", "Shrink tomato to a grid cell",
     [(W(3, "tomatoes"), "pop", 0.5), (W(3, "full"), "bubble", 0.3)],
     ["TOMATO", "GLU"])
shot(B(3, "so"), B(3, "parmesan"), "MEDIUM (2x2 grid)",
     "Tomato shrinks to the top-left cell of a 2x2 grid; mushrooms and a soy-sauce bottle pop into the next cells, "
     "each with a mini mascot. Bottom-right cell stays empty.",
     "Tomato | Mushrooms", "Soy sauce | (empty)", "—",
     "Locked.",
     ["MUSHROOMS", "SOY SAUCE"], "Continuous", "Continuous",
     [(W(3, "mushrooms"), "pop", 0.4), (W(3, "soy"), "pop", 0.4)],
     ["TOMATO", "GLU"])
shot(B(3, "parmesan"), L(4), "MEDIUM",
     "Parmesan wedge pops into the last cell, then scales up to fill MID and sparkles: the only food with an emphasis badge.",
     "PARMESAN", "Parmesan wedge", "ONE OF THE RICHEST",
     "Push-in on parmesan.",
     ["PARMESAN", "ONE OF THE RICHEST"], "Continuous", "Cut",
     [(W(3, "parmesan"), "pop", 0.45), (W(3, "richest"), "sparkle", 0.45)],
     ["GLU"], notes="No amounts or ranking between tomato/mushroom/soy.")
shot(L(4), B(4, "our"), "MEDIUM (tall chart)",
     "HUMAN MILK and COW'S MILK icons; bars rise from a baseline at y~1260 (above the caption band): human-milk bar "
     "~10x the cow's-milk bar. The vertical frame shows the full height.",
     "~10x label at the top of the tall bar", "Tall human-milk bar", "Short cow's-milk bar, icons + labels on the baseline",
     "Tilt up following the tall bar.",
     ["HUMAN MILK", "COW'S MILK", "FREE GLUTAMATE", "~10x"], "Cut", "Cut",
     [(W(4, "breast"), "pop_small", 0.35), (W(4, "contains"), "swoosh_up", 0.45), (W(4, "ten"), "ding", 0.4)],
     [], notes="Bar ratio ~10:1, no axis values; icon-level milk depiction (no nudity).")
shot(B(4, "our"), L(5), "MEDIUM-CLOSE",
     "Swaddled baby smiles with a warm umami glow; a tiny mascot floats past.",
     "FIRST MEAL = UMAMI", "Baby", "—",
     "Gentle push-in.",
     ["FIRST MEAL = UMAMI"], "Cut", "Cut",
     [(W(4, "first"), "chime", 0.4), (W(4, "umami."), "pop_small", 0.3)],
     ["GLU"])
shot(L(5), L(6), "MEDIUM",
     "Full-height human silhouette (natural in 9:16) with glowing interior; little cell factories produce mascots. "
     "Sun/moon arc cycles overhead.",
     "Sun/moon arc + EVERY DAY", "Silhouette with factories", "MADE IN YOUR BODY",
     "Locked, gentle drift.",
     ["MADE IN YOUR BODY", "EVERY DAY"], "Cut", "Horizontal split-wipe",
     [(W(5, "makes"), "bubble", 0.35), (W(5, "every"), "tick", 0.25), (W(5, "day."), "tick", 0.25)],
     ["GLU"])
shot(L(6), B(6, "are"), "WIDE (top/bottom split)",
     "Top half: a tomato releases a glutamate mascot. Bottom half: the MSG shaker shakes out a glutamate mascot.",
     "FROM TOMATO + tomato -> mascot (top half y 220-750)", "Divider ~y 760", "FROM MSG + shaker -> mascot (bottom half y 770-1290)",
     "Locked split, both halves drift.",
     ["FROM TOMATO", "FROM MSG"], "Split-wipe", "Continuous",
     [(L(6) + 0.1, "whoosh", 0.45), (W(6, "tomato"), "pop", 0.45), (W(6, "shaker"), "pop", 0.45)],
     ["TOMATO", "SHAKER", "GLU"])
shot(B(6, "are"), L(7), "MEDIUM",
     "Divider dissolves; the two mascots slide to center and overlap perfectly (identical); big '=' then SAME MOLECULE.",
     "=", "Two identical mascots merging", "SAME MOLECULE",
     "Slow push-in.",
     ["=", "SAME MOLECULE"], "Continuous", "Continuous",
     [(W(6, "exactly"), "impact_soft", 0.6), (W(6, "same"), "success", 0.45)],
     ["GLU"], hero="Tomato glutamate = MSG glutamate (copies must be identical)")
shot(L(7), D(), "MEDIUM -> END CARD",
     "CLIFFHANGER: the mascot drops off a spoon into a stylized open mouth / dark throat tunnel with a '?'. On 'part' "
     "the END CARD lands: 'PART 2 ->' with the part title, held to the last frame.",
     "'?' over the tunnel", "Mascot falling into the tunnel", "End card: PART 2 -> · THE SCIENCE (sits above the caption band)",
     "Tilt down following the mascot, then lock for the card.",
     ["?", "PART 2 →", "THE SCIENCE"], "Continuous", HOLD_OUT,
     [(W(7, "happens"), "swoosh_down", 0.45), (W(7, "eat"), "riser", 0.3), (W(7, "part"), "impact_soft", 0.55), (W(7, "two."), "chime", 0.4)],
     ["GLU", "END_CARD"], hero="Part 1 cliffhanger -> 'PART 2 →'")

# ================================================================ PART 2 — s05 body
scene("s05_body")
shot(0.0, L(2), "WIDE (vertical funnel)",
     "HOOK: lit on frame 0. A shaker at the top pours a stream of glutamate mascots down a vertical stylized gut "
     "funnel; gut cells lining it gobble almost all of them; only a stray one or two trickle into a thin blood "
     "vessel at the bottom. On 'Here's why' the camera snaps toward the mouth/top.",
     "PART 2/3 · THE SCIENCE tag + keyword WHERE DOES IT GO?", "Funnel with gut cells gobbling", "Thin blood vessel with a 1-2 particle trickle",
     "Slow tilt down with the stream; snap up on 'Here's why'.",
     ["PART 2/3", "WHERE DOES IT GO?"], HOOK_IN, "Snap-zoom (continuous)",
     [(0.0, "swoosh_down", 0.5), (W(1, "all"), "chomp", 0.3), (W(1, "msg"), "chomp", 0.3), (W(1, "blood."), "bubble", 0.3), (W(1, "here's"), "whoosh", 0.35)],
     ["GUTCELL", "GLU", "SHAKER", "PART_TAG"], hero="Part 2 hook: pouring stream eaten by the gut",
     notes="Do not show 95% here; the number lands later on the spoken word (s05 counter).")
shot(L(2), L(3), "MEDIUM",
     "A fork lifts a bite of seasoned food from a plate (LOW) up to a mouth in profile (MID).",
     "—", "Mouth in profile", "Plate + fork rising",
     "Tilt up with the fork.",
     [], "Snap-zoom", "Match-cut into the cutaway body",
     [(L(2), "whoosh", 0.35), (W(2, "eat"), "chomp", 0.5)],
     [])
shot(L(3), B(3, "and"), "MEDIUM -> CLOSE",
     "Full-height cutaway body (head at top of MID, torso below). Push into the tongue: receptors as docking cups; "
     "mascots dock with a click and glow.",
     "RECEPTORS", "Tongue receptors (close)", "—",
     "Zoom into tongue.",
     ["RECEPTORS"], "Match-cut", "Continuous",
     [(W(3, "docks"), "click", 0.5), (W(3, "receptors"), "click", 0.4), (W(3, "tongue,"), "click", 0.35)],
     ["BODY", "GLU"])
shot(B(3, "and"), L(4), "MEDIUM",
     "A signal pulse travels up a nerve to the brain; the brain pops a speech bubble.",
     "Brain + bubble PROTEIN! / YUM", "Nerve with pulse", "Tongue",
     "Tilt up following the pulse.",
     ["PROTEIN!", "YUM"], "Continuous", "Continuous",
     [(W(3, "sends"), "swoosh_up", 0.45), (W(3, "protein"), "ding", 0.45), (W(3, "yum."), "pop", 0.4)],
     ["BODY"])
shot(L(4), L(5), "WIDE (vertical dive)",
     "Camera dives down the oesophagus into the gut tunnel. On 'fall apart' a small '!' alarm bubble (s01 keyword "
     "style) cracks and crumbles.",
     "—", "Gut tunnel", "GUT label; crumbling '!'",
     "Long downward dive, ease-in-out.",
     ["GUT"], "Continuous", "Cut",
     [(L(4) + 0.05, "whoosh_long", 0.5), (W(4, "gut."), "pop_small", 0.3), (W(4, "fall"), "crunch", 0.35)],
     ["BODY"])
shot(L(5), L(6), "CLOSE",
     "Intestine wall: round gut-cell characters gobble passing mascots and light up with a small energy spark.",
     "FUEL", "Row of gut cells (2 rows stacked)", "Passing mascots",
     "Slow vertical truck along the wall.",
     ["FUEL"], "Cut", "Pull-back",
     [(W(5, "intestines"), "chomp", 0.45), (W(5, "glutamate."), "chomp", 0.4), (W(5, "burn"), "chomp", 0.4), (W(5, "fuel."), "sparkle", 0.35)],
     ["GUTCELL", "GLU"], notes="Canonical GUTCELL design; s06 calls back to this shot.")
shot(L(6), B(6, "is"), "MEDIUM",
     "Stream of particles enters; most are eaten. A percentage counter climbs and lands on 95% exactly on 'ninety-five'.",
     "95% counter (large)", "Gut wall eating the stream", "USED BY YOUR GUT",
     "Pull-back to medium, locked.",
     ["95%", "USED BY YOUR GUT"], "Continuous", "Continuous",
     [(W(6, "almost"), "tick", 0.2), (W(6, "glutamate"), "tick", 0.2), (W(6, "eat,"), "tick", 0.2), (W(6, "around"), "tick", 0.2),
      (W(6, "ninety-five"), "ding", 0.5)],
     ["GUTCELL", "GLU", "BADGE95"], hero="95% counter (lands on the spoken word)")
shot(B(6, "is"), L(7), "WIDE",
     "Pull back: gut wall as a border; only a few stray particles slip into a blood vessel below.",
     "95% badge (small)", "Gut wall", "Blood vessel + BLOODSTREAM label",
     "Continued pull-back / tilt down.",
     ["BLOODSTREAM"], "Continuous", "Track along the vessel",
     [(W(6, "right"), "bubble", 0.3), (W(6, "bloodstream."), "swoosh_down", 0.35)],
     ["GUTCELL", "GLU", "BADGE95"], notes="No '5%' on screen.")
shot(L(7), B(7, "a"), "WIDE (vertical travel)",
     "Follow the few particles up a vessel to the head; a castle-wall barrier rises across the frame with cute guards.",
     "BLOOD-BRAIN BARRIER", "Castle wall across the frame", "Particles rising in the vessel",
     "Track upward with particles.",
     ["BLOOD-BRAIN BARRIER"], "Track", "Continuous",
     [(L(7), "whoosh", 0.45), (W(7, "wall:"), "thud", 0.5), (W(7, "blood-brain"), "pop_small", 0.3)],
     ["BBB", "GLU"])
shot(B(7, "a"), L(8), "MEDIUM",
     "At the gate a guard checks arrivals; a particle approaches, the guard raises a palm, it bounces back.",
     "CONTROLLED BORDER", "Gate + guard", "Bouncing particle",
     "Push-in to the gate.",
     ["CONTROLLED BORDER"], "Continuous", "Pull-back",
     [(W(7, "controls"), "click", 0.35), (W(7, "brain."), "boing", 0.4)],
     ["BBB", "GLU"])
shot(L(8), D(), "WIDE (full body)",
     "Full vertical stack: brain with calm neurons (TOP), barrier, gut (MID), dinner plate (LOW). Mascot at the gate shrugs.",
     "Brain + calm neurons", "Barrier + gut", "Dinner plate",
     "Slow pull-back to full body.",
     [], "Continuous", DIP_OUT,
     [(W(8, "plate"), "pop", 0.35), (W(8, "stroll"), "boing", 0.25), (D() - 0.6, "whoosh", 0.3)],
     ["BODY", "BBB", "GLU"])

# ---------------------------------------------------------------- s07 tests
scene("s07_tests")
shot(0.0, L(2), "WIDE",
     "The claims, re-established for this part: keyword cards EXCITOTOXIN?!, POISON!, HEADACHES? (same style as s01 "
     "keywords) get pinned to a lab corkboard; a diverse team of scientists below; folders stack and get checks.",
     "Corkboard with claim cards", "Scientists at bench", "Folder stack with checks",
     "Slow tilt down board -> bench.",
     ["EXCITOTOXIN?!", "POISON!", "HEADACHES?", "TESTED", "AGAIN & AGAIN"], DIP_IN, "Cut",
     [(W(1, "claims"), "pop", 0.4), (W(1, "tested"), "click", 0.35), (W(1, "again,"), "click", 0.35), (W(1, "again."), "click", 0.35)],
     ["LAB"], notes="No study count on the folders.")
shot(L(2), B(2, "and"), "MEDIUM",
     "Gold-standard badge, then two identical capsules labeled A and B side by side.",
     "Badge + DOUBLE-BLIND", "Capsules A | B", "MSG? / PLACEBO? tags",
     "Push-in.",
     ["DOUBLE-BLIND", "A", "B"], "Cut", "Continuous",
     [(W(2, "gold"), "ding", 0.4), (W(2, "msg"), "pop", 0.4), (W(2, "placebo,"), "pop", 0.4)],
     ["CAPSULES"])
shot(B(2, "and"), L(3), "MEDIUM",
     "Capsules drop under two cups and shuffle; volunteer and researcher both wear cute blindfolds.",
     "Researcher (blindfolded)", "Shuffling cups", "Volunteer (blindfolded)",
     "Locked; slight push on blindfolds.",
     ["MSG?", "PLACEBO?"], "Continuous", "Cut",
     [(W(2, "and") + 0.05, "whoosh", 0.35), (W(2, "nobody,"), "whoosh", 0.3), (W(2, "researchers,"), "pop_small", 0.3), (W(2, "which."), "boing", 0.3)],
     ["CAPSULES"], hero="Double-blind shuffle")
shot(L(3), B(3, "reacted"), "WIDE",
     "Two stacked rows of diverse volunteers with SENSITIVE badges and worried thought bubbles; each takes a cup.",
     "SENSITIVE", "Volunteer row 1", "Volunteer row 2",
     "Slow tilt down.",
     ["SENSITIVE"], "Cut", "Continuous",
     [(L(3) + 0.1, "bubble", 0.35), (W(3, "people"), "bubble", 0.3), (W(3, "sensitive"), "bubble", 0.3)],
     ["VOLUNTEERS"])
shot(B(3, "reacted"), L(4), "MEDIUM (tall chart)",
     "Results: two tall bars MSG vs PLACEBO rise to nearly the same height; reaction icons split evenly.",
     "Bar tops", "Two bars", "MSG | PLACEBO labels on baseline (above caption band)",
     "Locked, drift.",
     ["MSG", "PLACEBO"], "Continuous", "Cut",
     [(W(3, "reacted"), "swoosh_up", 0.4), (W(3, "fake."), "ding", 0.35)],
     ["VOLUNTEERS"], hero="MSG vs placebo nearly equal", notes="No axis numbers.")
shot(L(4), B(4, "but"), "MEDIUM",
     "Comically huge spoonful over an empty plate/stomach icon; SOME volunteers get small symptom icons that fade quickly.",
     "HUGE DOSE / EMPTY STOMACH", "Spoon + empty plate", "MILD · SHORT-LIVED",
     "Push-in.",
     ["HUGE DOSE", "EMPTY STOMACH", "MILD · SHORT-LIVED"], "Cut", "Cut",
     [(W(4, "huge"), "thud", 0.4), (W(4, "mild,"), "pop_small", 0.3), (W(4, "some"), "tick", 0.25), (W(4, "people."), "tick", 0.25)],
     ["VOLUNTEERS"])
shot(B(4, "but"), L(5), "WIDE -> MEDIUM",
     "Result cards scatter with mixed marks; they clear and a normal meal appears with a relaxed volunteer and a calm check.",
     "NOT CONSISTENT", "Scatter -> normal meal", "NORMAL FOOD ✓",
     "Pull-back, then push-in on the plate.",
     ["NOT CONSISTENT", "NORMAL FOOD ✓"], "Cut", "Cut",
     [(W(4, "results"), "paper", 0.35), (W(4, "consistent,"), "error", 0.25), (W(4, "food."), "success", 0.4)],
     ["VOLUNTEERS"])
shot(L(5), B(5, "the", 2), "MEDIUM",
     "Generic official-looking document (no real seal/logo) slides in; GRAS stamp slams on.",
     "US FDA (plain text)", "Document + GRAS stamp", "GENERALLY RECOGNIZED AS SAFE",
     "Push-in; 4-frame shake on stamp.",
     ["US FDA", "GRAS", "GENERALLY RECOGNIZED AS SAFE"], "Cut", "Pull-back",
     [(W(5, "food"), "paper", 0.35), (W(5, "generally"), "stamp", 0.8), (W(5, "generally") + 0.05, "impact_soft", 0.4)],
     ["GRAS"], hero="GRAS stamp")
shot(B(5, "the", 2), D(), "WIDE (2x2 grid)",
     "The MSG shaker joins salt, sugar and a baking-powder tin in a 2x2 grid under the GRAS label.",
     "GRAS label", "MSG | SALT", "SUGAR | BAKING POWDER",
     "Pull-back, hold.",
     ["SALT", "SUGAR", "BAKING POWDER"], "Pull-back", DIP_OUT,
     [(W(5, "salt,"), "pop", 0.4), (W(5, "sugar,"), "pop", 0.4), (W(5, "baking"), "pop", 0.4), (D() - 0.9, "success", 0.35)],
     ["SHAKER", "SALT", "GRAS"])

# ---------------------------------------------------------------- s08 twist (ends part 2)
scene("s08_twist")
shot(0.0, L(2), "MEDIUM",
     "Twist transition; MSG shaker in a spotlight with a curling arrow.",
     "PLOT TWIST", "Shaker", "—",
     "Spin-in, hold.",
     ["PLOT TWIST"], DIP_IN + " + spin", "Cut",
     [(0.3, "swoosh_up", 0.5), (W(1, "twist."), "pop", 0.4), (W(1, "help."), "sparkle", 0.4)],
     ["SHAKER"])
shot(L(2), L(3), "MEDIUM",
     "Balance scale: SALT and MSG shakers land on level pans (gram for gram). Sodium ions pop up: three over SALT, one over MSG.",
     "Na ions: 3 over SALT, 1 over MSG", "Level scale with both shakers", "≈ 1/3 THE SODIUM",
     "Locked, then push-in on the ions.",
     ["SALT", "MSG", "1 g = 1 g", "≈ 1/3 THE SODIUM"], "Cut", "Cut",
     [(W(2, "gram"), "thud", 0.45), (W(2, "gram,"), "thud", 0.45), (W(2, "third") - 0.3, "pop_small", 0.3),
      (W(2, "third") - 0.15, "pop_small", 0.3), (W(2, "third"), "pop_small", 0.3), (W(2, "sodium"), "ding", 0.45)],
     ["SALT", "SHAKER", "NA"], hero="About a third of the sodium (ions exactly 3:1)")
shot(L(3), B(3, "and"), "MEDIUM",
     "Cooking pot; a salt spoon is partly swapped for an MSG spoon.",
     "Spoons swapping", "Pot", "—",
     "Locked.",
     ["SWAP"], "Cut", "Continuous",
     [(W(3, "swap"), "swoosh_down", 0.4), (W(3, "msg,"), "pop_small", 0.35), (W(3, "recipe"), "bubbling_pot", 0.25)],
     ["SALT", "SHAKER"])
shot(B(3, "and"), B(3, "while"), "MEDIUM",
     "A SODIUM meter beside the pot drops.",
     "SODIUM ↓", "Pot + sodium meter", "—",
     "Slight push-in on the meter.",
     ["SODIUM ↓"], "Continuous", "Continuous",
     [(W(3, "cut"), "swoosh_down", 0.4), (W(3, "noticeably,"), "tick", 0.25)],
     ["NA"], notes="No % reduction on screen ('noticeably' only).")
shot(B(3, "while"), L(4), "MEDIUM",
     "A FLAVOR meter next to it stays full; a heart icon pops up and smiles.",
     "Heart icon", "Sodium meter low | flavor meter full", "FLAVOR ✓",
     "Slow pull-back.",
     ["FLAVOR ✓"], "Continuous", "Cut",
     [(W(3, "keeping"), "pop", 0.4), (W(3, "flavor."), "success", 0.4)],
     ["SHAKER"], notes="Heart = mood icon; no health claim.")
shot(L(4), B(4, "that's"), "MEDIUM",
     "CLIFFHANGER QUESTION: shaker alone in the spotlight; shadows of the s01 claim cards and newspapers sweep over it. "
     "On 'reputation' the NO MSG sign buzzes on behind it and a sealed envelope (the 1968 letter, unopened) drifts down.",
     "HOW?", "Shaker + passing shadows; NO MSG sign behind", "Envelope drifting down",
     "Slow push-in.",
     ["HOW?"], "Cut", "Continuous",
     [(W(4, "science"), "riser", 0.3), (W(4, "how"), "pop", 0.4), (W(4, "reputation?"), "neon_buzz", 0.4)],
     ["SHAKER", "SIGN", "LETTER"])
shot(B(4, "that's"), D(), "END CARD",
     "END CARD: 'PART 3 ->' with the part title lands on 'part'; held to the last frame.",
     "—", "End card: PART 3 -> · THE RUMOUR", "Envelope resting",
     "Lock.",
     ["PART 3 →", "THE RUMOUR"], "Continuous", HOLD_OUT,
     [(W(4, "part"), "impact_soft", 0.55), (W(4, "three."), "chime", 0.4)],
     ["END_CARD", "LETTER"], hero="Part 2 cliffhanger -> 'PART 3 →'")

# ================================================================ PART 3 — s06 origin
scene("s06_origin")
shot(0.0, L(2), "MEDIUM",
     "HOOK: lit on frame 0: a fast vertical collage of NO MSG signs and scary headlines; on 'single letter' one "
     "envelope falls through them toward camera and lands centered.",
     "PART 3/3 · THE RUMOUR tag + 50 YEARS OF FEAR", "Sign/headline collage -> envelope", "Envelope lands",
     "Slow push-in; collage scrolls upward.",
     ["PART 3/3", "50 YEARS OF FEAR"], HOOK_IN, "Cut",
     [(0.0, "neon_flicker", 0.4), (W(1, "fear"), "whoosh", 0.35), (W(1, "single"), "paper", 0.45), (W(1, "letter."), "thud", 0.35)],
     ["SIGN", "LETTER", "HEADLINE", "PART_TAG"], hero="Part 3 hook: one letter out of 50 years of fear")
shot(L(2), L(3), "MEDIUM",
     "A kombu strip morphs into the MSG shaker; spotlight snaps on; a target reticle frames it.",
     "PUBLIC ENEMY #1", "Kombu -> shaker", "Reticle base",
     "Push-in.",
     ["PUBLIC ENEMY #1"], "Cut", "Cut",
     [(W(2, "seaweed"), "whoosh", 0.4), (W(2, "public"), "impact_soft", 0.5), (W(2, "one?"), "pop_small", 0.3)],
     ["SHAKER"])
shot(L(3), L(4), "MEDIUM",
     "Chapter 1968. The envelope opens and a tall typewritten letter unfolds (vertical paper fills the frame).",
     "Chapter label 1968", "Letter unfolding", "—",
     "Tilt down with the letter.",
     ["1968"], "Cut", "Push-in",
     [(W(3, "1968,"), "pop", 0.4), (W(3, "letter."), "paper", 0.45)],
     ["LETTER", "CHAPTER"])
shot(L(4), B(4, "describing"), "CLOSE",
     "Letter close-up: lines type themselves (illegible); the address line is legible.",
     "TO: NEW ENGLAND JOURNAL OF MEDICINE", "Typing lines", "—",
     "Slow push along the letter.",
     ["TO: NEW ENGLAND JOURNAL OF MEDICINE"], "Push-in", "Continuous",
     [(L(4) + 0.1, "typewriter", 0.4), (W(4, "new"), "pop_small", 0.3)],
     ["LETTER"], notes="Plain text only, generic period letter layout; author not named or drawn.")
shot(B(4, "describing"), B(4, "after"), "MEDIUM",
     "Three symptom icons pop out of the letter on each word: tingling hand, drooping arm, fluttering heart.",
     "NUMBNESS", "WEAKNESS", "PALPITATIONS",
     "Gentle pull-back.",
     ["NUMBNESS", "WEAKNESS", "PALPITATIONS"], "Continuous", "Continuous",
     [(W(4, "numbness,"), "pop", 0.4), (W(4, "weakness"), "pop", 0.4), (W(4, "palpitations"), "heartbeat", 0.35)],
     ["LETTER"])
shot(B(4, "after"), L(5), "MEDIUM",
     "Within the letter, a small line illustration: a plain storefront outline and a US map outline. Objects only.",
     "—", "Plain storefront outline + US outline", "UNITED STATES",
     "Locked.",
     ["UNITED STATES"], "Continuous", "Cut",
     [(W(4, "restaurants"), "pop_small", 0.3), (W(4, "united"), "pop_small", 0.3)],
     ["LETTER"], notes="SENSITIVE: no people, no 'oriental' signifiers; neutral storefront only.")
shot(L(5), B(5, "he", 2), "WIDE (line-up)",
     "Suspect line-up of three objects: cooking-wine bottle, salt shaker, MSG shaker; a '?' pops over each when named.",
     "? ? ?", "Wine | salt | MSG", "—",
     "Locked, slight drift.",
     ["?", "?", "?"], "Cut", "Continuous",
     [(W(5, "wine?"), "blip", 0.4), (W(5, "salt?"), "blip", 0.4), (W(5, "msg?"), "blip", 0.45)],
     ["SALT", "SHAKER"])
shot(B(5, "he", 2), L(6), "MEDIUM",
     "The three '?' drift together into one wobbly label on the letter.",
     "A GUESS", "Merging '?'", "—",
     "Locked.",
     ["A GUESS"], "Continuous", "Whip to the headline",
     [(W(5, "know."), "pop_small", 0.3), (W(5, "guess."), "boing", 0.35)],
     ["LETTER"])
shot(L(6), L(7), "CLOSE",
     "The quoted headline stamps across the page on the name, with camera shake, then holds.",
     "—", "\"CHINESE-RESTAURANT / SYNDROME\" (stacked, quoted)", "—",
     "Whip-in, 6-frame shake on impact, hold.",
     ["\"CHINESE-RESTAURANT SYNDROME\""], "Whip-pan", "Cut",
     [(W(6, "catchy"), "swoosh_down", 0.35), (W(6, "chinese"), "stamp", 0.8), (W(6, "chinese") + 0.05, "impact_big", 0.45)],
     ["HEADLINE"], hero="Headline stamp", notes="Neutral typeface only; framed as a quoted historical artifact.")
shot(L(7), B(7, "almost"), "WIDE",
     "Envelopes pour in from the top; newspapers multiply and spin toward camera.",
     "Envelopes pouring", "Spinning newspapers", "—",
     "Push-in into the spinning paper.",
     [], "Cut", "Spin transition",
     [(W(7, "letters"), "paper", 0.4), (W(7, "newspapers"), "whoosh", 0.45)],
     ["HEADLINE"])
shot(B(7, "almost"), L(8), "MEDIUM",
     "The A GUESS label morphs into a clipboard-style DIAGNOSIS stamp.",
     "GUESS", "Morph", "DIAGNOSIS",
     "Locked.",
     ["GUESS → DIAGNOSIS"], "Spin transition", "Cut",
     [(W(7, "guess"), "swoosh_up", 0.35), (W(7, "diagnosis."), "stamp", 0.55)],
     [])
shot(L(8), B(8, "and"), "MEDIUM",
     "Chapter 1969. Cute lab with newborn mouse pups in a nest; a comically giant syringe looms down from the top. "
     "Injection implied, never shown.",
     "Chapter 1969 + giant syringe", "Mouse pups in nest", "HUGE DOSE",
     "Slow push-in.",
     ["1969", "HUGE DOSE", "NEWBORN MICE"], "Cut", "Cut",
     [(W(8, "year"), "pop", 0.35), (W(8, "injected"), "swoosh_down", 0.4), (W(8, "huge"), "impact_soft", 0.4), (W(8, "newborn"), "pop_small", 0.3)],
     ["MOUSE", "SYRINGE", "CHAPTER"], notes="No gore; no needle entering an animal.")
shot(B(8, "and"), L(9), "WIDE",
     "Newspapers return with bigger, scarier headlines (illegible except 'MSG' and '!'); a brain icon with a warning "
     "mark as the 'photo'; muted alarm bell shakes.",
     "MSG!?", "Scary newspapers", "Alarm bell",
     "Push-in through the papers.",
     ["MSG!?"], "Cut", "Horizontal split-wipe",
     [(W(8, "found"), "paper", 0.4), (W(8, "headlines"), "whoosh", 0.35), (W(8, "scarier."), "alarm", 0.25)],
     ["HEADLINE"])
shot(L(9), B(9, "remember:"), "WIDE (top/bottom split)",
     "Top: giant syringe over a tiny mouse pup (INJECTION). Bottom: hands holding a seasoned bowl (MEAL). Big ≠ on the divider.",
     "INJECTION: syringe + pup (top half)", "≠ on divider", "MEAL: bowl + hands (bottom half)",
     "Locked split, drift.",
     ["INJECTION", "≠", "MEAL"], "Split-wipe", "Push into the bowl / cutaway gut",
     [(W(9, "injecting"), "whoosh", 0.4), (W(9, "nothing"), "impact_soft", 0.55), (W(9, "meal."), "pop", 0.35)],
     ["MOUSE", "SYRINGE"], hero="Injection ≠ meal")
shot(B(9, "remember:"), L(10), "CLOSE",
     "CALLBACK TO PART 2: the same gut cells gobble glutamate mascots; the 95% badge reappears small.",
     "Small chip 'PART 2' (optional)", "Gut cells gobbling (reuse s05.06)", "95% badge",
     "Vertical truck (same move as s05.06).",
     ["95%", "PART 2 (optional chip)"], "Push-in / cutaway", "Muted dissolve",
     [(W(9, "gut"), "chomp", 0.4), (W(9, "all"), "chomp", 0.35), (W(9, "first."), "pop_small", 0.3)],
     ["GUTCELL", "GLU", "BADGE95"], notes="Must reuse GUTCELL + BADGE95 exactly as in s05; reads on its own for viewers who skipped Part 2.")
shot(L(10), B(10, "chinese"), "WIDE",
     "Mood shift (slightly muted). Chapter 1960s AMERICA over period objects only: a newsstand of papers, a 1960s TV "
     "set, a kitchen radio. No people.",
     "Chapter 1960s AMERICA", "Newsstand / TV set / radio", "—",
     "Slow tilt down.",
     ["1960s AMERICA"], "Muted dissolve", "Continuous",
     [(L(10), "whoosh_long", 0.35), (W(10, "1960s"), "pop", 0.35)],
     ["CHAPTER"], notes="SENSITIVE: no people (per director's rules); objects and type only.")
shot(B(10, "chinese"), L(11), "MEDIUM",
     "Prejudice as objects: cold grey tags labelled with the attitude drift onto a plain white takeout carton.",
     "'FOREIGN'", "Plain carton", "'SUSPICIOUS'",
     "Slow push-in on the carton.",
     ["'FOREIGN'", "'SUSPICIOUS'"], "Continuous", "Split-wipe",
     [(W(10, "foreign"), "pop_small", 0.3), (W(10, "suspicious."), "static", 0.2)],
     ["TAKEOUT"], notes="Words in quotes so they read as the attitude being criticised.")
shot(L(11), B(11, "the", 2), "WIDE (top/bottom split)",
     "Top: pasta with tomato sauce and parmesan, calm, silent bell. Bottom: the plain carton, same bell ringing. "
     "Identical glutamate mascots over BOTH. 'ITALIAN RESTAURANT SYNDROME?' fades in over the top half and dissolves.",
     "TOMATO SAUCE + pasta, silent bell (top half)", "Divider", "Carton + ringing bell (bottom half)",
     "Locked split, drift.",
     ["TOMATO SAUCE", "ITALIAN RESTAURANT SYNDROME? (dissolves)"], "Split-wipe", "Collapse onto bottom half",
     [(W(11, "tomato"), "pop_small", 0.3), (W(11, "nobody", 2), "alarm", 0.3), (W(11, "syndrome."), "whoosh", 0.3)],
     ["TOMATO", "TAKEOUT", "GLU"], hero="Double standard split")
shot(B(11, "the", 2), D(), "CLOSE",
     "Split collapses; the HEADLINE card returns; camera pushes in on the name; hold.",
     "—", "HEADLINE card (same as s06.09)", "—",
     "Slow push-in, ease-out.",
     ["\"CHINESE-RESTAURANT SYNDROME\""], "Continuous", DIP_OUT,
     [(W(11, "fear"), "swoosh_down", 0.3), (W(11, "name"), "impact_soft", 0.4)],
     ["HEADLINE"])

# ---------------------------------------------------------------- s09 outro (ends part 3 / series)
scene("s09_outro")
shot(0.0, L(2), "WIDE",
     "TV-wall callback (self-explanatory for Part-3-only viewers): the five fictional heads still mouth their keyword "
     "claims; MSG shaker in front with a '?'.",
     "DRAMATIZATION pill", "TV wall with heads + keyword bubbles", "Shaker + '?'",
     "Slow push-in.",
     ["?", "DRAMATIZATION"], DIP_IN, "Cut",
     [(0.2, "static", 0.25), (W(1, "msg"), "pop", 0.35)],
     ["TVWALL", "DRAMA_TAG", "SHAKER"])
shot(L(2), B(2, "but"), "TALL",
     "Absurd-dose gag: a comically giant water tank filling the frame height next to a normal glass.",
     "Top of giant tank", "Giant tank | normal glass", "EVEN WATER",
     "Tilt up the giant tank.",
     ["EVEN WATER"], "Cut", "Continuous",
     [(W(2, "absurd"), "bubble", 0.35), (W(2, "even"), "boing", 0.35)],
     [])
shot(B(2, "but"), L(3), "MEDIUM",
     "Balance scale: pizza slice with tomato vs a pinch of MSG, level.",
     "≈", "Scale", "Pizza | MSG pinch",
     "Locked, drift.",
     ["≈"], "Continuous", "Cut",
     [(W(2, "evidence"), "pop", 0.35), (W(2, "dangerous"), "impact_soft", 0.45), (W(2, "tomato"), "pop_small", 0.3)],
     ["TOMATO", "SHAKER"])
shot(L(3), B(3, "one"), "WIDE",
     "Back to the TV wall, all claims looping; the shaker small in front.",
     "—", "TV wall", "Shaker",
     "Slow pull-back.",
     [], "Cut", "Continuous",
     [(L(3), "whoosh", 0.3), (W(3, "story", 2), "static", 0.2)],
     ["TVWALL", "DRAMA_TAG", "SHAKER"])
shot(B(3, "one"), B(3, "can"), "MEDIUM",
     "Three tags tumble onto the wall as each phrase is spoken.",
     "ONE GUESS", "A CATCHY NAME (mini HEADLINE card)", "A LITTLE PREJUDICE",
     "Locked.",
     ["ONE GUESS", "A CATCHY NAME", "A LITTLE PREJUDICE"], "Continuous", "Continuous",
     [(W(3, "guess,"), "pop", 0.4), (W(3, "catchy"), "pop", 0.4), (W(3, "prejudice"), "pop", 0.4)],
     ["HEADLINE", "TVWALL", "DRAMA_TAG"])
shot(B(3, "can"), L(4), "WIDE",
     "TVs switch off one by one (C3, A3, C1, A1, B2) while a tall stack of research papers rises beside them; tags fall.",
     "DECADES OF SCIENCE", "TV wall switching off", "Paper stack base",
     "Slow pull-back.",
     ["DECADES OF SCIENCE"], "Continuous", "Cut",
     [(W(3, "outlive"), "tv_off", 0.45), (W(3, "decades"), "tv_off", 0.45), (W(3, "decades") + 0.33, "tv_off", 0.45),
      (W(3, "of", 3), "tv_off", 0.45), (W(3, "science."), "tv_off", 0.5)],
     ["TVWALL", "DRAMA_TAG"], hero="TVs switch off one by one")
shot(L(4), L(5), "MEDIUM",
     "The NO MSG sign from s01; it flickers and 'NO' burns out, leaving 'MSG' glowing warmly.",
     "Night sky", "NO MSG -> MSG sign", "Window",
     "Slow push-in.",
     ["NO MSG → MSG"], "Cut", "Dissolve through the window",
     [(L(4) + 0.1, "neon_buzz", 0.35), (W(4, "no"), "neon_flicker", 0.5), (W(4, "sign,"), "click", 0.45), (W(4, "real"), "sparkle", 0.35)],
     ["SIGN"], hero="'NO' burns out")
shot(L(5), L(5) + (CUES["s09_outro"]["lines"][4]["end"] - L(5)), "CLOSE",
     "Cozy steaming bowl of ramen; a hand reaches for the MSG shaker and gives it two shakes.",
     "Steam", "Bowl + hand + shaker", "Table",
     "Gentle push-in.",
     [], "Dissolve", "Continuous",
     [(W(5, "maybe..."), "slurp", 0.3), (W(5, "reach"), "pop_small", 0.35), (W(5, "shaker."), "pop_small", 0.35)],
     ["SHAKER"])
shot(CUES["s09_outro"]["lines"][4]["end"], D(), "CLOSE -> EXTREME WIDE",
     "Pull back: bowl -> window -> city -> planet -> stars. Series END CARD.",
     "—", "End card: MSG: The Most Misunderstood Molecule", "PART 3/3 · END",
     "Fast-then-slow pull-back into space.",
     ["MSG: The Most Misunderstood Molecule", "PART 3/3"], "Continuous", HOLD_OUT,
     [(D() - 2.4, "whoosh_long", 0.55), (D() - 1.3, "chime", 0.45)],
     ["GLU", "END_CARD"], hero="Series end card")

# ================================================================ continuity, guardrails
CONTINUITY = {
    "GLU": ("Glutamate mascot", "One design for the whole series: the approved style's mascot (placeholder: KZ.glutamate with face). "
            "Never redesigned per part; friendly, never menacing. In s04's equality shot both copies are identical."),
    "SHAKER": ("MSG shaker", "One prop (placeholder KZ.shaker(...,'MSG')); label reads exactly 'MSG'. Negative contexts change "
               "overlays/lighting (X, reticle), never the prop."),
    "SALT": ("Salt shaker", "Same silhouette as MSG shaker, different body/cap tone; label 'SALT'."),
    "TVWALL": ("Retro TV wall", "3x3 grid ~860 px wide centered on x=490 in MID (background rows may continue into UI zones). "
               "Slots: HOST=B2 (center), TVDOC=A1, CHEF=C1, GURU=A3, AUNT=C3; other 4 = static. Light-up order s01: B2,A1,C1,A3,C3; "
               "switch-off order s09: C3,A3,C1,A1,B2. Silhouettes with role props; no real likeness; invented channel names only."),
    "DRAMA_TAG": ("DRAMATIZATION tag", "Whenever a talking head is visible: 'DRAMATIZATION - fictional character' in the role "
                  "lower-third (LOW zone) and a DRAMATIZATION pill in TOP; text >= 40 px."),
    "SIGN": ("NO MSG neon sign", "'NO' and 'MSG' as separate tubes so 'NO' can burn out alone; same window, same angle in s01, s08 (teaser), s06 (hook collage), s09."),
    "GUTCELL": ("Gut cells", "Round intestine-cell characters with dot eyes; gobble animation. Canonical in s05; s05 hook and s06 callback reuse it unchanged."),
    "BADGE95": ("95% badge/counter", "Same typography/shape in s05 counter, s05 pull-back and the s06 callback."),
    "NA": ("Sodium ion", "Small round 'Na' character, identical in s03 and s08; s08 count exactly 3:1."),
    "TOMATO": ("Tomato", "Same ripe tomato in s04, s06 (sauce), s09 (pizza)."),
    "HEADLINE": ("Headline card", "One quoted, neutral-typeface card \"CHINESE-RESTAURANT SYNDROME\" (stacked over 2 lines in 9:16); reused in s06 and as a mini tag in s09."),
    "LETTER": ("1968 letter / envelope", "Generic period letter + envelope; teased sealed in s08, lands in s06 hook, opens in s06. Author never named or drawn."),
    "TASTE_ICONS": ("Five taste icons", "SWEET, SOUR, SALTY, BITTER + UMAMI; consistent set and order."),
    "IKEDA": ("Kikunae Ikeda", "Respectful simple character (period suit, moustache, round glasses), one design."),
    "BODY": ("Cutaway body", "Full-height side-profile body map (tongue, nerve, gut, vessels, brain) used throughout s05."),
    "BBB": ("Blood-brain barrier", "Castle wall + cute, friendly guards."),
    "CHAPTER": ("Chapter labels", "Same chapter-label component (placeholder KZ.chapter), placed in TOP zone, left-aligned at x=40."),
    "PART_TAG": ("PART n/3 tag", "Pill in TOP zone, left-aligned x=40, y~230-290: 'PART n/3 · <PART TITLE>'. Pops in within 0.3 s of a part opener, "
                 "stays large ~3 s, then shrinks to a small persistent pill for the rest of the opening scene."),
    "END_CARD": ("End card", "One component for all three parts: 'PART n →' + part title (parts 1-2), series title + 'PART 3/3' (part 3). "
                 "Sits in MID/LOW above the caption band; lands on the spoken word 'part'; holds to the last frame."),
}
INCIDENTAL = {"TAKEOUT", "MOUSE", "SYRINGE", "LAB", "CAPSULES", "VOLUNTEERS", "GRAS"}

GUARDRAILS = {
    "allowed_numbers_and_claims": [
        "'50+ YEARS' / 'more than fifty years' (s01); '50 YEARS OF FEAR' (s06 hook: 'Fifty years of fear')",
        "1908, Tokyo, Kikunae Ikeda, kombu, glutamic acid, umami ~ 'delicious taste', sodium + glutamate = monosodium glutamate, on sale 'within a year' (s03) - NO year for the sale",
        "Glutamate: one of the most common amino acids; protein building block; tomatoes, mushrooms, soy sauce, parmesan ('one of the richest') (s04)",
        "Human breast milk ~10x more FREE glutamate than cow's milk (s04): '~10x', bars ~10:1",
        "Body makes its own glutamate every day; tomato glutamate = MSG glutamate (s04)",
        "Almost all MSG you eat never makes it into the blood (s05 hook); ~95% used in the gut (s05) - the ONLY percentage; never '5%'",
        "1968 letter to the New England Journal of Medicine; numbness, weakness, palpitations; suspects wine, salt, MSG; 'a guess'; headline 'Chinese Restaurant Syndrome' (s06)",
        "1969: huge doses injected under the skin of newborn mice, brain damage found (s06) - no gore",
        "Double-blind placebo testing; sensitive people reacted as often to placebo; huge doses on an empty stomach -> mild, short-lived symptoms in some, inconsistent; nothing with normal amounts in normal food (s07)",
        "US FDA 'generally recognized as safe' (GRAS), same category as salt, sugar, baking powder (s07)",
        "~1/3 the sodium of table salt gram for gram; swapping some salt cuts sodium 'noticeably', keeps flavor (s08) - no % reduction",
        "'Even water' in absurd amounts; 'about as dangerous as the tomato on your pizza' (s09)",
        "Navigation text: 'PART n/3', 'PART 2 →', 'PART 3 →', part titles from script.py PARTS",
    ],
    "forbidden": [
        "Any number, %, chart value, axis label, study count, dose or date not listed above (no '1909', no mg/kg, no 'n=', no per-food glutamate amounts).",
        "Health claims beyond the script ('heart healthy', 'cures', 'zero side effects').",
        "Real logos, mastheads, seals, networks, platforms or brands - plain text names only; invented channel names.",
        "Real or recognisable people; the cold-open heads are fictional composites. The 1968 letter writer and the 1969 scientist are never named or drawn.",
        "Gore: no needle entering an animal, no dead mice, no lesion imagery.",
        "Burned-in captions are not 'on-screen text': keyword text must stay short and must not be a second subtitle track.",
    ],
    "dramatization": "Every frame with a cold-open talking head (s01.01-06, s09.01, s09.04-06) carries the DRAMATIZATION tag (lower-third + TOP pill). Role labels are generic: TALK SHOW HOST, DAYTIME TV DOCTOR, CELEBRITY CHEF, WELLNESS INFLUENCER, YOUR AUNT, ON SOCIAL MEDIA.",
    "sensitive_1960s_section": [
        "Applies to s01.04, s06 (whole scene, esp. s06.06, s06.09, s06.16-19) and s09.05.",
        "No caricatured Asian people; in s06 no restaurant staff or diners are drawn at all - paper, type, newsprint and objects only (matches director's rules).",
        "No 'oriental' signifiers: no wonton/chop-suey fonts, pagodas, lanterns, dragons, gongs, conical hats, chopsticks as decoration. Takeout = plain white carton.",
        "Prejudice shown as cold tags / a ringing bell put ON the food, never as a trait of people; 'foreign'/'suspicious' in quotes.",
        "Headline shown verbatim, hyphenated, in quotation marks, as a historical artifact being critiqued - never a joke or brand.",
        "Ikeda / Tokyo 1908 respectful and period-plausible.",
    ],
}

# ================================================================ build + validate
errors = []
parts_out, all_shots = [], {}
for p in script.PARTS:
    pdur, pscenes, offset = 0.0, [], 0.0
    for i, sid in enumerate(p["scenes"]):
        meta, shots = CUES[sid], SCENES[sid]
        Dd = meta["duration"]
        if shots[0]["start"] != 0: errors.append(f"{sid}: first shot not at 0")
        if abs(shots[-1]["end"] - Dd) > 1e-3: errors.append(f"{sid}: last shot ends {shots[-1]['end']} != {Dd}")
        for j, sh in enumerate(shots):
            sh["id"] = f"{sid[:3]}.{j + 1:02d}"
            if j and abs(shots[j - 1]["end"] - sh["start"]) > 1e-6: errors.append(f"{sh['id']}: gap/overlap")
            dur = sh["end"] - sh["start"]
            if (dur < 2.0 and sid != "s02_title") or dur > 7.5: errors.append(f"{sh['id']}: duration {dur:.2f}")
            for s in sh["sfx"]:
                if s["name"] not in ALLOWED: errors.append(f"{sh['id']}: bad sfx {s['name']}")
                if not (sh["start"] - 0.15 <= s["t"] <= sh["end"] + 0.05) or not (0 <= s["t"] <= Dd): errors.append(f"{sh['id']}: sfx {s['t']} outside shot")
                if not 0 <= s["gain"] <= 1: errors.append(f"{sh['id']}: gain")
            for pr in sh["props"]:
                if pr not in CONTINUITY and pr not in INCIDENTAL: errors.append(f"{sh['id']}: undefined prop {pr}")
            sh["lines"] = []
            for n, ln in enumerate(meta["lines"], 1):
                ov = min(sh["end"], ln["end"]) - max(sh["start"], ln["start"])
                if ov > 0.2:
                    full = sh["start"] <= ln["start"] + 0.05 and sh["end"] >= ln["end"] - 0.05
                    sh["lines"].append({"n": n, "voice": ln["voice"], "start": ln["start"], "end": ln["end"],
                                        "coverage": "full" if full else "partial", "text": ln["text"]})
            sh["part"] = p["id"]
            sh["part_time"] = [round(offset + sh["start"], 3), round(offset + sh["end"], 3)]
            all_shots[sh["id"]] = sh
        pscenes.append({"id": sid, "title": meta["title"], "duration": Dd, "part_offset": round(offset, 3),
                        "role": "opener" if i == 0 else ("closer" if i == len(p["scenes"]) - 1 else "middle"),
                        "shots": [{k: sh[k] for k in ["id", "start", "end", "part_time", "lines", "shot_size", "on_screen",
                                                       "layout", "camera", "text", "transition_in", "transition_out", "sfx",
                                                       "props", "hero", "notes"]} for sh in shots]})
        offset += Dd
    parts_out.append({"id": p["id"], "title": p["title"], "duration": round(offset, 3), "scenes": pscenes,
                      "hook_shot": pscenes[0]["shots"][0]["id"], "end_shot": pscenes[-1]["shots"][-1]["id"]})
# every hook shot must show the part tag in its first 2 s and be lit at frame 0
for p in parts_out:
    h = all_shots[p["hook_shot"]]
    if "PART_TAG" not in h["props"] or h["transition_in"] != HOOK_IN: errors.append(f"{p['id']}: hook shot lacks PART_TAG / lit open")
    if "END_CARD" not in all_shots[p["end_shot"]]["props"]: errors.append(f"{p['id']}: no end card")
if errors:
    print("\n".join(errors)); sys.exit(1)

continuity = []
for cid, (name, rules) in CONTINUITY.items():
    used = [s for s in all_shots.values() if cid in s["props"]]
    continuity.append({"id": cid, "name": name, "rules": rules, "appears_in": [s["id"] for s in used],
                       "parts": sorted({s["part"] for s in used})})
heroes = [{"shot": s["id"], "part": s["part"], "start": s["start"], "end": s["end"], "beat": s["hero"]}
          for s in all_shots.values() if s["hero"]]
out = {"meta": {"project": "MSG: The Most Misunderstood Molecule", "format": "Instagram Reel series, 3 standalone parts, 9:16",
                "fps": 30, "resolution": [1080, 1920], "frame": FRAME,
                "time_base": "seconds, scene-relative (0..duration); part_time = seconds from the start of the part",
                "source_of_truth": "cues.json (line + word timings), script.py (PARTS, narration)",
                "generator": "design/gen_shotlist.py (edit there, not here)",
                "style": "style-agnostic: framing/motion/content/timing only; look per the approved style direction"},
       "parts": parts_out, "hero_moments": heroes, "continuity": continuity, "guardrails": GUARDRAILS}
json.dump(out, open(os.path.join(HERE, "shotlist.json"), "w"), indent=1, ensure_ascii=False)

# ================================================================ markdown
f2 = lambda x: f"{x:.2f}"  # noqa: E731
n_shots = len(all_shots)
md = [f"""# MSG — Shot List (Instagram Reel series, 9:16)

_Generated by `design/gen_shotlist.py` from `cues.json` + `script.py` — edit the generator, then re-run
`python3 design/gen_shotlist.py`. Machine-readable twin: `design/shotlist.json`._

**Totals:** 3 parts · 9 scenes · {n_shots} shots · {len(heroes)} hero moments ·
""" + " · ".join(f"{p['id']} {p['duration']:.1f} s" for p in parts_out) + """

## Frame map (1080x1920) — applies to every shot

```
y    0 ┌──────────────────────────────┬────┐
       │ UI TOP — background only     │    │
y  220 ├──────────────────────────────┤ R  │  TOP  220-520  part tag (x=40), chapter labels, keywords
       │ TOP                          │ I  │
y  520 ├──────────────────────────────┤ G  │  MID  520-1100 main subject (center x = 490, not 540)
       │ MID                          │ H  │
y 1100 ├──────────────────────────────┤ T  │  LOW  1100-1300 secondary subject, labels, lower-thirds, end-card text
       │ LOW                          │    │
y 1300 ├══════════════════════════════┤ 140│  CAPTION BAND 1300-1480: burned-in captions only
       │ CAPTIONS                     │ px │
y 1480 ├──────────────────────────────┤    │
       │ UI BOTTOM — background only  │    │
y 1920 └──────────────────────────────┴────┘
       x=40                       x=940  1080
```

- **Key-content box: x 40-940, y 220-1300.** Background, ground planes and ambient motion may bleed into UI zones,
  the right gutter and behind the caption band, but nothing that must be read or seen.
- Each shot below lists **TOP / MID / LOW** contents. Split screens are **top/bottom** (top half y 220-750, divider
  ~760, bottom half y 770-1290). Rows of 4 become 2x2 grids. Prefer vertical camera moves (tilt, crane, dive) to pans.
- **Captions** are a separate burned-in layer driven by the per-word `words` timings in `cues.json` (see
  PRODUCTION_PLAN.md). Animators keep the band clear and do not draw subtitles. Keyword text (TOP) is a few words max.
- **Text sizes:** at least 40 px for any on-screen word (director's rule); keywords 72-120 px.
- **Timing:** shot boundaries use line starts or `word start - 0.1 s` from the real word timings, and SFX land on
  real word starts — all exact, no guessing. Times are scene-relative (the `t` passed to `draw`); `part_time` in the
  JSON gives the time within the part.
- **Joins:** part openers (s01, s05, s06) are fully lit on frame 0 (hook; no fade-in). Part closers (s04, s08, s09)
  hold their end card to the last frame (no fade-out; Reels loop). Internal joins keep a short 0.3 s dip. _This
  changes BRIEF.md's "fade every scene" rule for openers/closers — producer to confirm._
- **Style-agnostic:** this document fixes content, framing, motion, timing and sound only.
- SFX: `time name gain`, names from BRIEF.md only. ★ = hero moment.
"""]
for p in parts_out:
    md += [f"# {p['title']}  ·  {p['duration']:.1f} s", "",
           f"Hook: **{p['hook_shot']}** (first 2 s) · Ending: **{p['end_shot']}**", ""]
    for sc in p["scenes"]:
        md += [f"## {sc['id']} — {sc['title']}  ·  {sc['duration']:.3f} s  ·  {len(sc['shots'])} shots  ·  "
               f"part offset {sc['part_offset']:.2f} s  ·  {sc['role']}", ""]
        for sh in sc["shots"]:
            hero = f"  ★ {sh['hero']}" if sh["hero"] else ""
            md += [f"### {sh['id']}  `{f2(sh['start'])} → {f2(sh['end'])}` ({sh['end'] - sh['start']:.1f} s) · {sh['shot_size']}{hero}", ""]
            for ln in sh["lines"]:
                md.append(f"> L{ln['n']} {ln['voice']} [{f2(ln['start'])}–{f2(ln['end'])}{', partial' if ln['coverage'] == 'partial' else ''}]: {ln['text']}  ")
            if not sh["lines"]:
                md.append("> (no narration)")
            lay = sh["layout"]
            md += ["",
                   f"- **Action:** {sh['on_screen']}",
                   f"- **TOP:** {lay['TOP']}  \n  **MID:** {lay['MID']}  \n  **LOW:** {lay['LOW']}",
                   f"- **Camera:** {sh['camera']}",
                   f"- **Text:** {', '.join(sh['text']) if sh['text'] else '—'}",
                   f"- **Transition:** in: {sh['transition_in']} / out: {sh['transition_out']}",
                   "- **SFX:** " + ", ".join(f"`{s['t']:.2f} {s['name']} {s['gain']}`" for s in sh["sfx"])]
            if sh["props"]: md.append(f"- **Continuity:** {', '.join(sh['props'])}")
            if sh["notes"]: md.append(f"- **Note:** {sh['notes']}")
            md.append("")
md += ["---", "", "## Hero moments", "", "| # | Part | Shot | Time (scene s) | Beat |", "|---|---|---|---|---|"]
for i, h in enumerate(heroes, 1):
    md.append(f"| {i} | {h['part']} | {h['shot']} | {f2(h['start'])}–{f2(h['end'])} | {h['beat']} |")
md += ["", "Hero shots get the most animation time and a producer still-review before full animation.", "",
       "## Continuity bible", "", "Identity, layout and behaviour only; the look comes from the approved style. "
       "\"Appears in\" is computed from the shots above.", ""]
for c in continuity:
    md += [f"### {c['id']} — {c['name']}  ({', '.join(c['parts'])})", f"- **Rules:** {c['rules']}",
           f"- **Appears in:** {', '.join(c['appears_in'])}", ""]
md += ["## Factual guardrails", "", "### Allowed on-screen numbers / claims (only these)", ""]
md += [f"- {x}" for x in GUARDRAILS["allowed_numbers_and_claims"]]
md += ["", "### Forbidden", ""] + [f"- {x}" for x in GUARDRAILS["forbidden"]]
md += ["", "### DRAMATIZATION rule", "", GUARDRAILS["dramatization"], "", "### 1960s section / cultural sensitivity", ""]
md += [f"- {x}" for x in GUARDRAILS["sensitive_1960s_section"]]
open(os.path.join(HERE, "SHOTLIST.md"), "w").write("\n".join(md) + "\n")

for p in parts_out:
    print(p["id"], f"{p['duration']:.1f}s", {s["id"]: len(s["shots"]) for s in p["scenes"]})
print(n_shots, "shots,", len(heroes), "heroes")
