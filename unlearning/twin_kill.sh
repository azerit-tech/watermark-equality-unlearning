#!/bin/bash
# Kill my own twin helper processes on the login node: bash twin_kill.sh chain|test|base
# (keywords, so that the calling ssh shell's own command line never contains the real pattern)
for pat in "$@"; do
  case $pat in
    chain) pat="twin_login_chain.sh";;
    test) pat="regime test";;
    base) pat="twin_login_base.sh";;
  esac
  for pid in $(pgrep -u $USER -f "$pat"); do
    [ $pid != $$ ] && kill $pid 2>/dev/null
  done
done
