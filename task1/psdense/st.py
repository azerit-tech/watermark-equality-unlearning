import json, sys
for f in sys.argv[1:]:
    print(f)
    try:
        for l in open(f):
            r = json.loads(l); g = r["groups"]
            print(" t=%.1f val=%.4f hole=%.3f trans=%.3f out=%.3f full=%.3f part=%.3f frozen=%s" % (r["time_min"], r["val"], g["edit/hole"][0], g["edit/transplant"][0], g["edit/outpainting"][0], g["regime/full"][0], g["regime/partial"][0], r["enc_frozen"]))
    except FileNotFoundError:
        print(" none")
