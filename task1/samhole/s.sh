#!/bin/bash
exec ssh -o BatchMode=yes -o ConnectTimeout=25 -o ControlMaster=auto -o ControlPath=/private/tmp/claude-501/cm-samhole-%C -o ControlPersist=20m jureca "$@"
