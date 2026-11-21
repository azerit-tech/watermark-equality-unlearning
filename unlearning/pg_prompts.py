"""Prompt vocabulary for the pseudo-gold data (SDXL-Turbo text-to-image).

SUBJECTS: candidate subjects for the class-coverage pass (pg_cover.py); the original model tells which class each one hits.
STYLES / SETTINGS / VIEWS / LIGHTS: modifiers mixed at random by pg_gen.py so that Turbo does not repeat one composition.
The task images look like cinematic, moody, shallow-depth-of-field photographs of single objects, so most styles stay close.
"""
import random

_ANIMALS = ["fox", "owl", "rabbit", "heron", "turtle", "cat", "dog", "horse", "deer", "bear", "elephant", "frog", "swan",
            "duck", "penguin", "lion", "fish", "dragon", "squirrel", "hedgehog", "mouse", "pig", "cow", "rooster", "eagle",
            "giraffe", "snail", "butterfly", "unicorn", "whale", "crane bird", "tiger", "wolf", "goat", "sheep", "monkey",
            "panda", "koala", "camel", "dolphin", "seal", "octopus", "crab", "parrot", "peacock", "flamingo", "badger",
            "raccoon", "otter", "bee"]
_MATERIALS = ["ceramic", "wooden", "stone", "brass", "glass", "porcelain", "bronze", "clay"]

_OBJECTS = """
a candle on a table|a candle holder with three candles|an oil lamp|a table lamp|a desk lamp|a floor lamp|a chandelier|string lights in a jar|
a paper lantern|a red hanging lantern|a wall-mounted lantern on a house at dusk|a hanging lantern silhouette against the sky|a stone garden lantern|
a camping lantern in a tent|a lantern in the snow|a street lamp at night|a neon sign|an edison light bulb|a lava lamp|a campfire|a fireplace|
a coffee mug|a teacup and saucer|a ceramic teapot|a glass teapot|a copper kettle|a french press|a coffee grinder|an espresso machine|a moka pot|
a wine glass|a wine bottle|a beer mug|a mason jar|a cast iron pan|a wooden spoon|a cutting board|a bowl of soup|a ceramic bowl|a stack of plates|
a whisk|a rolling pin|a cookie jar|a jar of honey|a salt shaker|a water pitcher|a ceramic vase|a thermos|a lunch box|a bottle of olive oil|
spice jars on a shelf|a cup of hot chocolate|a cocktail glass|a milk bottle|a measuring cup|
a loaf of bread|a baguette|a croissant|a bagel|a pretzel|a slice of cake|a cupcake|an apple pie|a pizza|a plate of pasta|sushi|a burger|
a sandwich|a red apple|an orange|a lemon|a bunch of bananas|strawberries|cherries|grapes|a pumpkin|a watermelon slice|a pear|a pomegranate|
an avocado|tomatoes on the vine|corn on the cob|carrots|onions|garlic bulbs|potatoes|eggs in a basket|a wheel of cheese|a bar of chocolate|
cookies|a donut|ice cream cone|macarons|pancakes|waffles|a salad|a steak|a roasted chicken|coffee beans|a bowl of rice|popcorn|a cinnamon roll|
a muffin|a lemon tart|a jar of jam|a bunch of herbs|a bowl of cherries|
a cactus in a pot|an aloe vera plant in a terracotta pot|an agave plant on rocky ground|a succulent|a jade plant|a snake plant|a monstera plant|
a fern|a bonsai tree|an orchid|a sunflower|a red rose|tulips|lavender|daisies|a lily|a lotus flower|cherry blossoms|a poppy field|ivy on a wall|
moss on a rock|bamboo|a palm tree|an oak tree|a pine tree|a birch forest|maple leaves in autumn|a weeping willow|an olive tree|a wheat field|
herbs in pots|a terrarium|a mushroom growing in the grass|a fly agaric mushroom|chanterelle mushrooms|a cluster of mushrooms on a log|
a shelf fungus on a tree|a puffball mushroom|a pinecone|acorns|a dandelion seed head|a prickly pear cactus|a barrel cactus|a hanging plant|
a potted fern by a window|a christmas cactus|a venus flytrap|a tree stump|fallen leaves|a hydrangea|a magnolia flower|
a wooden dock on a lake|a pier at sunset|a steel truss bridge at sunset|a suspension bridge|a stone arch bridge|a wooden footbridge|
a covered bridge|a rope bridge in a jungle|a lighthouse on a rocky coast|a lighthouse at night|a harbor|a beach|a waterfall|a river|a lake|
a pond with lily pads|a glacier|a mountain peak|a volcano|a canyon|sand dunes|a sea cliff|a cave|a small island|a coral reef|an iceberg|
the northern lights|a stormy sky|a rainbow|a foggy forest|a boardwalk|a canal|a dam|a lock gate on a canal|a jetty with fishing rods|
a sailboat on the sea at dusk|a kayak on green water seen from above|a canoe on a misty lake|a fishing boat on a foggy sea|
a wooden rowboat on a calm lake|a yacht|a gondola in venice|a cruise ship|a cargo ship|a tugboat|a pirate ship|a viking ship|a submarine|
a houseboat|a paddle steamer|a jet ski|a dragon boat|a raft on a river|a motorboat|a fishing trawler|a speedboat|a catamaran|a ferry|
a person paddling a canoe|a small boat tied to a dock|an old shipwreck|
a wicker chair with a blanket|a striped folding deck chair on a terrace|an antique armchair on a porch|a lounge chair on a rooftop terrace|
a rocking chair|an office chair|a bar stool|a wooden bench in a park|a sofa|a bed with white sheets|a hammock|a bookshelf|a wooden desk|
a dining table|a wardrobe|a chest of drawers|a standing mirror|a side table|a bean bag|a porch swing|a baby cradle|a folding chair|
an adirondack chair|a leather armchair by a fireplace|a director's chair|a garden chair|a plastic chair|a stool in a workshop|a throne|
a windmill on a small island at sunset|a wind turbine in the fog|a watermill|a water wheel|a grain silo|a red barn|a farmhouse|a weather vane|
a scarecrow|a tractor|a haystack|a stone well|a wooden fence|a garden gate|a mailbox|a dutch windmill by a canal|a wind farm at sunset|
an old stone tower|a wind pump on a farm|
a knitted wool scarf|a folded white blanket|a colorful woven basket|a cushion on a sofa|a patchwork quilt|a rug|curtains|a knitted sweater|
mittens|a beanie hat|wool socks|a towel|balls of yarn with knitting needles|a sewing machine|a tapestry|a tablecloth|a camera on a cushion|
a pillow on a bed|a crocheted doll|a straw basket with flowers|a wicker picnic basket|a laundry basket|a plaid throw blanket|
a pair of leather boots on a shelf|a sneaker on a gravel path at sunset|slippers|high heels|sandals|rain boots|ballet shoes|hiking boots|
cowboy boots|flip flops|ice skates|roller skates|a fedora hat|a straw hat|a top hat|an umbrella|a handbag|a backpack|a suitcase|a wallet|
a leather belt|a necktie|gloves|sunglasses|a wristwatch|a necklace|a ring|earrings|a raincoat|a dress on a hanger|a leather jacket|
muddy work boots|wooden clogs|a baseball cap|a bow tie|
an hourglass on a window sill|a wall clock|a pocket watch|an alarm clock|a grandfather clock|a cuckoo clock|a sundial in a courtyard|
an astrolabe|a compass|a globe|a telescope|a microscope|binoculars|a magnifying glass|a barometer|a metronome|a kitchen scale|an old map|
a sextant|an armillary sphere|a clock tower|a stopwatch|
a kite in the sky|a hot air balloon|a paper airplane|a bird flying|a flock of birds|a seagull|an eagle in flight|a parachute|an airplane|
a drone|a blimp|balloons|a paraglider|a weathercock|
a totem pole|a wooden carved eagle|a stone gargoyle|a marble bust|an angel statue|a buddha statue|a tribal mask|a venetian mask|
a painting on an easel|paint brushes in a jar|a sculpture|a pottery wheel|a stained glass window|a mosaic|chess pieces|a dream catcher|
a wind chime|a wooden carving of a bird|a stone statue in a garden|a bronze statue of a horse|a wooden mask on a wall|a carved wooden spoon|
a pagoda|a temple|a church|a castle|a cottage|a log cabin|a treehouse|a stone tower|a greenhouse|a gazebo|a fountain|a mosque|a cathedral|
an igloo|a tent|a yurt|a skyscraper|ancient ruins|a colosseum|a pyramid|a bell tower|a torii gate|a stone archway|a staircase|a front door|
a window with shutters|a balcony with flowers|a japanese pagoda by a lake|a chinese pavilion|a lighthouse keeper's house|a windmill in a field|
a wooden birdhouse on a tree|a bird feeder|a beehive|a watering can|a wheelbarrow|a garden gnome|a flower pot|a shovel|a rake|a garden hose|
a bird bath|a trellis with roses|a compost bin|a beekeeper box|an insect hotel|a bird nest with eggs|a dog house|a chicken coop|
a red origami crane|an origami boat|a paper boat on water|a book|a stack of books|a notebook|a letter|an envelope|a newspaper|a postcard|
a pen|a fountain pen|pencils|a typewriter|a quill and ink|a scroll|a calendar|an open book on a table|a library shelf|a diary|
an anchor at a harbor at sunset|a ship wheel|a buoy|a life ring|a fishing net|a coil of rope|a lobster trap|an oar|a message in a bottle|
seashells|a starfish|a ship in a bottle|a porthole|a rusty anchor on the beach|a nautical lantern|a diving helmet|
a guitar|a violin|a cello|a piano|a drum|a trumpet|a saxophone|a flute|a harp|an accordion|a ukulele|a gramophone|a record player|
vinyl records|headphones|a microphone|a music box|a harmonica|church bells|a french horn|a banjo|
a vintage camera|a film camera|a smartphone|a laptop|a computer keyboard|a computer mouse|an old television|a radio|a cassette player|
a game controller|a rotary telephone|a calculator|a toy robot|a polaroid camera|a boombox|a tablet|a desktop computer|
a teddy bear|a rubber duck|a toy car|a toy train|wooden blocks|a rocking horse|a doll|marbles|a yo-yo|a jigsaw puzzle|dice|playing cards|
a chess board|a spinning top|lego bricks|a snow globe|a toy soldier|a puppet|a stuffed bunny|a toy sailboat|
a bicycle leaning on a wall|a motorcycle|a scooter|a vintage car|a truck|a bus|a train|a tram|a helicopter|a skateboard|a sled|
a horse carriage|a vespa|a camper van|a race car|a tractor in a field|a steam locomotive|a fire truck|
a hammer|a wrench|a screwdriver|a saw|an axe in a log|a power drill|a toolbox|scissors|a measuring tape|a paint can|a ladder|a broom|
a bucket|a lock and key|a bunch of keys|a padlock|a chain|nails|a flashlight|spools of thread|a sewing needle|an anvil|a pocket knife|
a soccer ball|a basketball|a tennis racket|a baseball glove|a golf ball|skis|a snowboard|a surfboard|bowling pins|dumbbells|boxing gloves|
a fishing rod|a bow and arrow|a bicycle helmet|a hockey stick|a football|a volleyball on the beach|
a vase of flowers|a picture frame|a christmas wreath|a christmas tree|christmas ornaments|a gift box|a carved pumpkin lantern|easter eggs|
a perfume bottle|a jewelry box|a crystal ball|an aquarium|a goldfish bowl|a bird cage|incense sticks|a crystal geode|a feather|
a stone cairn|a sand castle|a bathtub|a sink|a bar of soap|a toothbrush|a shower|a mirror on a wall|a candle in a glass jar|
a heron standing on stone steps|a cat sleeping|a dog in a field|a horse in a meadow|a cow|a sheep|a goat|a pig|a chicken|a duck on a pond|
a swan|a deer in a forest|a red fox|a rabbit in the grass|a squirrel|a hedgehog|an owl on a branch|a parrot|a robin|a hummingbird|
a butterfly on a flower|a bee on a flower|a ladybug|a snail|a frog|a turtle|a goldfish|koi fish|a jellyfish|an octopus|a crab|a lobster|
a whale|a dolphin|a seal|a penguin|a polar bear|a brown bear|a wolf|a lion|a tiger|an elephant|a giraffe|a zebra|a monkey|a panda|
a koala|a kangaroo|a camel|a snake|a lizard|a crocodile|a flamingo|a peacock|a dragonfly|a spider web with dew|
a wooden sailboat model|a stone bench|a mailbox in the snow|a hourglass with blue sand|a broken clock|a sundial on a pedestal|
an old wooden door|a rope swing on a tree|a picnic table|a park bench in autumn|a fire hydrant|a phone booth|a vending machine|a traffic light|
a road sign|a parking meter|a gas station pump|a bridge over a river at night|a lighthouse beam in the fog|a harbour crane|
"""

SUBJECTS = [s.strip() for s in _OBJECTS.replace("\n", "").split("|") if s.strip()]
for i, a in enumerate(_ANIMALS):
    SUBJECTS.append(f"a {_MATERIALS[i % len(_MATERIALS)]} {a} figurine on a table")
    SUBJECTS.append(f"a small {_MATERIALS[(i + 3) % len(_MATERIALS)]} {a} statue")
SUBJECTS = list(dict.fromkeys(SUBJECTS))

STYLES = ["cinematic photograph, moody natural light, shallow depth of field", "cinematic photograph, shallow depth of field",
          "moody photograph", "professional photograph, bokeh", "film photo, 35mm", "detailed photo, soft focus background",
          "cinematic still, dramatic light", ""]
COVER_STYLES = STYLES[:3]
SETTINGS = ["on a wooden table", "by a window", "outdoors", "in a garden", "on a shelf", "on a stone floor", "in a cozy room",
            "in a forest", "near the sea", "in the snow", "in the rain", "on a dark background", "on a porch", "in a field",
            "on a rustic surface", "in a city street", "", "", "", ""]
VIEWS = ["close-up", "wide shot", "seen from above", "low angle", "side view", "macro shot", "in the distance", "centered",
         "", "", ""]
LIGHTS = ["golden hour", "soft morning light", "moody dusk light", "candlelight", "overcast daylight", "dramatic side light",
          "blue hour", "warm lamplight", "foggy atmosphere", "backlit", "", ""]


def cover_prompt(subject, k):
    st = COVER_STYLES[k % len(COVER_STYLES)]
    return f"{subject}, {st}" if st else subject


def varied_prompt(subject, rng: random.Random):
    parts = [subject]
    for pool in (SETTINGS, VIEWS, LIGHTS, STYLES):
        m = rng.choice(pool)
        if m and (pool is not SETTINGS or " on " not in subject and " in " not in subject or rng.random() < 0.3):
            parts.append(m)
    return ", ".join(parts)


if __name__ == "__main__":
    print(len(SUBJECTS), "subjects")
    r = random.Random(0)
    for s in SUBJECTS[:5]:
        print(varied_prompt(s, r))
