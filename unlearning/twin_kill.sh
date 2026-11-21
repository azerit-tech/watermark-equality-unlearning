#!/bin/bash
# Kill my own twin helper processes on the login node: bash twin_kill.sh chain|test|base|maintest
# (keywords, so that the calling ssh shell's own command line never contains the real pattern)
for key in "$@"; do
  case $key in
    chain) pats="twin_login_chain.sh";;
    test) pats="regime_test";;
    base) pats="twin_login_base.sh";;
    maintest) pats="twin_relay_pb10q_main.sh|--stages 3,4|unlearn.py --twin|scrub.py --twin";;
    *) pats="$key";;
  esac
  echo "$pats" | tr '|' '\n' | while read -r pat; do
    for pid in $(pgrep -u $USER -f -- "$pat"); do
      [ "$pid" != "$$" ] && kill $pid 2>/dev/null
    done
  done
done
exit 0
