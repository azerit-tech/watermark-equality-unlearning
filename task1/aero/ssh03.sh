#!/bin/bash
exec ssh -o BatchMode=yes -o ConnectTimeout=20 -o HostKeyAlias=jureca.fz-juelich.de -o ControlMaster=auto -o ControlPath=/private/tmp/claude-501/cm-aero03-%C -o ControlPersist=20m -o HostName=jureca03.fz-juelich.de jureca "$@"
