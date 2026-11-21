#!/bin/zsh
# Wait for wave 0, then run ref- plans R1_a..R1_d_second in alphabetical order.
# Stops BEFORE R1_e_filler (ep- diagnostic E1 goes in between, run by hand).
cd /Users/jacquesmougeot/Desktop/Azerit-hack
until grep -q 'exit=' model_equality/campaigns/W0b_D2_C3.log; do sleep 5; done
for b in R1_a_breadth R1_b_core R1_c_hardpairs R1_d2_e1_baseline R1_d_second; do
  f=model_equality/probes/waves/$b.json
  log=model_equality/campaigns/W1_$b.log
  if grep -q 'ep-' $f; then echo "SKIP $f: contains ep-" >> model_equality/campaigns/W1_driver.log; continue; fi
  echo "$(date +%T) start $b" >> model_equality/campaigns/W1_driver.log
  python3.12 -m model_equality.api run --plan $f --execute > $log 2>&1
  echo "exit=$? $(date +%T)" >> $log
  echo "$(date +%T) end $b $(tail -1 $log)" >> model_equality/campaigns/W1_driver.log
done
echo "$(date +%T) R1 a-d DONE (R1_e pending after E1)" >> model_equality/campaigns/W1_driver.log
